import torch
import torch.nn.functional as F

from mmdet.core import (bbox2roi, get_uncertain_point_coords,
                        get_uncertain_point_coords_with_randomness, roi2poi)
from mmdet.ops.point_sample import point_sample
from .. import builder
from ..registry import HEADS
from .base_roi_head import BaseRoIHead


@HEADS.register_module
class PointRoIHead(BaseRoIHead):

    def __init__(self, point_extractor, point_head, *args, **kwargs):
        super(PointRoIHead, self).__init__(*args, **kwargs)
        self.point_extractor = builder.build_point_extractor(point_extractor)
        self.point_head = builder.build_head(point_head)
        self.point_head.init_weights()

    def forward_dummy(self, x, proposals):
        raise NotImplementedError

    def forward_train(self,
                      x,
                      img_meta,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_meta)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        loss_bbox, bbox_feats = self._bbox_forward_train(
            x, sampling_results, gt_bboxes, gt_labels, img_meta)
        losses.update(loss_bbox)

        # mask head forward and loss
        loss_mask, mask_pred = self._mask_forward_train(
            x, sampling_results, bbox_feats, gt_masks, img_meta)
        if loss_mask is not None:
            losses.update(loss_mask)

        loss_mask_point = self._mask_point_forward_train(
            x, sampling_results, mask_pred, gt_masks)

        if loss_mask_point is not None:
            losses.update(loss_mask_point)

        return losses

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_meta):
        mask_feats = self.extract_mask_feats(x, sampling_results, bbox_feats)

        if mask_feats.shape[0] > 0:
            mask_pred = self.mask_head(mask_feats)
            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks, self.train_cfg)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            return loss_mask, mask_pred
        else:
            return None, None

    def _mask_point_forward_train(self, x, sampling_results, mask_pred,
                                  gt_masks):

        if mask_pred is not None:
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            roi_coords = get_uncertain_point_coords_with_randomness(
                mask_pred,
                pos_labels,
                self.train_cfg.num_points,
                self.train_cfg.oversample_ratio,
                self.train_cfg.importance_sample_ratio,
            )
            rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            coarse_feats, fine_grained_feats = self.extract_point_feats(
                x, rois, roi_coords, mask_pred)
            mask_point_pred = self.point_head(coarse_feats, fine_grained_feats)
            pos_pois = roi2poi(rois, roi_coords)
            mask_point_target = self.point_head.get_target(
                pos_pois, sampling_results, gt_masks, self.train_cfg)
            loss_mask_point = self.point_head.loss(mask_point_pred,
                                                   mask_point_target,
                                                   pos_labels)

            return loss_mask_point
        else:
            return None

    def extract_point_feats(self, x, rois, roi_coords, mask_pred):
        coarse_feats = point_sample(mask_pred, roi_coords, align_corners=False)
        pois = roi2poi(rois, roi_coords)

        fine_grained_feats = self.point_extractor(
            [x[i] for i in self.point_extractor.in_indices], pois)
        return coarse_feats, fine_grained_feats

    def refine_mask(self, x, mask_rois, label_pred, mask_pred):

        refined_mask_pred = mask_pred.clone()
        for subdivision_step in range(self.test_cfg.subdivision_steps):
            refined_mask_pred = F.interpolate(
                refined_mask_pred,
                scale_factor=self.test_cfg.scale_factor,
                mode='bilinear',
                align_corners=False)
            # If `subdivision_num_points` is larger or equal to the
            # resolution of the next step, then we can skip this step
            H, W = refined_mask_pred.shape[-2:]
            if (self.test_cfg.subdivision_num_points >= 4 * H * W and
                    subdivision_step < self.test_cfg.subdivision_steps - 1):
                continue
            point_indices, roi_coords = get_uncertain_point_coords(
                refined_mask_pred, label_pred,
                self.test_cfg.subdivision_num_points)
            coarse_feats, fine_grained_feats = self.extract_point_feats(
                x, mask_rois, roi_coords, mask_pred)
            point_logits = self.point_head(coarse_feats, fine_grained_feats)

            R, C, H, W = refined_mask_pred.shape
            point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
            refined_mask_pred = (
                refined_mask_pred.reshape(R, C, H * W).scatter_(
                    2, point_indices, point_logits).view(R, C, H, W))

            return refined_mask_pred

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            refined_mask_pred = self.refine_mask(x, mask_rois, det_labels,
                                                 mask_pred)
            segm_result = self.mask_head.get_seg_masks(refined_mask_pred,
                                                       _bboxes, det_labels,
                                                       self.test_cfg,
                                                       ori_shape, scale_factor,
                                                       rescale)
        return segm_result
