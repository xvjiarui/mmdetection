import torch
import torch.nn.functional as F

from mmdet.core import (bbox2roi, build_assigner, build_sampler,
                        get_uncertain_point_coords,
                        get_uncertain_point_coords_with_randomness,
                        point_sample)
from .. import builder
from ..registry import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module
class PointRend(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 point_extractor,
                 point_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(PointRend, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.point_extractor = builder.build_point_extractor(point_extractor)
        self.point_extractor.init_weights()
        self.point_head = builder.build_head(point_head)
        self.point_head.init_weights()

    def init_weights(self, pretrained=None):
        super(PointRend, self).init_weights(pretrained)
        # TODO refactor init_weights
        if hasattr(self, 'point_head'):
            self.point_head.init_weights()
        if hasattr(self, 'point_extractor'):
            self.point_extractor.init_weights()

    def forward_dummy(self, img):
        raise NotImplementedError

    # TODO decouple two stage training
    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(self.train_cfg.rcnn.sampler, context=self)
        num_imgs = img.size(0)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = bbox_assigner.assign(proposal_list[i],
                                                 gt_bboxes[i],
                                                 gt_bboxes_ignore[i],
                                                 gt_labels[i])
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        # bbox head forward and loss
        rois = bbox2roi([res.bboxes for res in sampling_results])
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            [x[i] for i in self.bbox_roi_extractor.in_indices], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_targets = self.bbox_head.get_target(sampling_results, gt_bboxes,
                                                 gt_labels,
                                                 self.train_cfg.rcnn)
        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
        losses.update(loss_bbox)

        # mask head forward and loss
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_feats = self.mask_roi_extractor(
                [x[i] for i in self.mask_roi_extractor.in_indices], pos_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            pos_rois = rois[pos_inds]
            mask_feats = bbox_feats[pos_inds]

        if mask_feats.shape[0] > 0:
            mask_pred = self.mask_head(mask_feats)
            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks,
                                                     self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

            point_coords = get_uncertain_point_coords_with_randomness(
                mask_pred,
                pos_labels,
                self.train_cfg.point.num_points,
                self.train_cfg.point.oversample_ratio,
                self.train_cfg.point.importance_sample_ratio,
            )

            fine_grained_feats = self.point_extractor(
                [x[i] for i in self.point_extractor.in_indices], pos_rois,
                point_coords)
            coarse_feats = point_sample(
                mask_pred, point_coords, align_corners=False)
            point_pred = self.point_head(fine_grained_feats, coarse_feats)
            point_targets = self.point_head.get_target(point_coords,
                                                       mask_targets,
                                                       self.train_cfg.point)
            loss_point = self.point_head.loss(point_pred, point_targets,
                                              pos_labels)
            losses.update(loss_point)

        return losses

    def refine_mask(self, x, mask_rois, label_pred, mask_pred):

        refined_mask_pred = mask_pred.clone()
        for subdivision_step in range(self.test_cfg.point.subdivision_steps):
            refined_mask_pred = F.interpolate(
                refined_mask_pred,
                scale_factor=self.test_cfg.point.scale_factor,
                mode='bilinear',
                align_corners=False)
            # If `mask_point_subdivision_num_points` is larger or equal to the
            # resolution of the next step, then we can skip this step
            H, W = refined_mask_pred.shape[-2:]
            if (self.test_cfg.point.subdivision_num_points >= 4 * H * W
                    and subdivision_step <
                    self.test_cfg.point.subdivision_steps - 1):
                continue
            point_indices, point_coords = get_uncertain_point_coords(
                refined_mask_pred, label_pred,
                self.test_cfg.point.subdivision_num_points)
            fine_grained_feats = self.point_extractor(
                [x[i] for i in self.point_extractor.in_indices], mask_rois,
                point_coords)
            coarse_feats = point_sample(
                mask_pred, point_coords, align_corners=False)
            point_logits = self.mask_point_head(fine_grained_feats,
                                                coarse_feats)

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
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
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
                [x[i] for i in self.mask_roi_extractor.in_indices], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)

            refined_mask_pred = self.refine_mask(x, mask_rois, det_labels + 1,
                                                 mask_pred)

            segm_result = self.mask_head.get_seg_masks(refined_mask_pred,
                                                       _bboxes, det_labels,
                                                       self.test_cfg.rcnn,
                                                       ori_shape, scale_factor,
                                                       rescale)
        return segm_result
