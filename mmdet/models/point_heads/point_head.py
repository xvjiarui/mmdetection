# Modified from https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend  # noqa

import torch
import torch.nn as nn

from mmdet.core import mask_point_target, poi2list
from mmdet.ops import point_sample
from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module
class SharedFCPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with
    kernel 1. The head takes both fine-grained and coarse prediction
    features as its input.
    """

    def __init__(self,
                 num_fcs=3,
                 in_channels=256,
                 fc_channels=256,
                 num_classes=81,
                 class_agnostic=False,
                 coarse_pred_each_layer=True,
                 loss_point=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(SharedFCPointHead, self).__init__()
        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_channles = fc_channels
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.coarse_pred_each_layer = coarse_pred_each_layer
        self.loss_point = build_loss(loss_point)

        fc_channels_in = in_channels + num_classes
        self.fcs = nn.ModuleList()
        for k in range(num_fcs):
            fc = nn.Conv1d(
                fc_channels_in,
                fc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True)
            self.fcs.append(fc)
            fc_channels_in = fc_channels
            fc_channels_in += num_classes if self.coarse_pred_each_layer else 0

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.fc_logits = nn.Conv1d(
            fc_channels_in, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.fcs:
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fc_logits.weight, std=0.001)
        if self.fc_logits.bias is not None:
            nn.init.constant_(self.fc_logits.bias, 0)

    def forward(self, mask_pred, roi_coords, fine_grained_features):
        coarse_features = point_sample(
            mask_pred, roi_coords, align_corners=False)
        x = torch.cat((coarse_features, fine_grained_features), dim=1)
        for fc in self.fcs:
            x = self.relu(fc(x))
            if self.coarse_pred_each_layer:
                x = torch.cat((x, coarse_features), dim=1)
        return self.fc_logits(x)

    def get_target(self, pos_pois, sampling_results, gt_masks,
                   point_train_cfg):
        pos_poi_list = poi2list(pos_pois)
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]

        mask_point_targets = mask_point_target(pos_poi_list,
                                               pos_assigned_gt_inds, gt_masks,
                                               point_train_cfg)

        return mask_point_targets

    def loss(self, point_pred, point_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_point = self.loss_point(point_pred, point_targets,
                                         torch.zeros_like(labels))
        else:
            loss_point = self.loss_point(point_pred, point_targets, labels)
        loss['loss_point'] = loss_point
        return loss

    def _get_uncertainty(self, mask_pred, labels):
        """
        We estimate uncertainty as L1 distance between 0.0 and the logit
        prediction in 'mask_pred' for the foreground class in `classes`.

        Args:
            mask_pred (Tensor): A tensor of shape (R, C, ...) or (R, 1,
            ...) for class-specific or class-agnostic, where R is the total
            number of predicted masks in all images and C is the number of
            foreground classes. The values are logits.

            labels (list): A list of length R that contains either predicted of
            ground truth class for each predicted mask.

        Returns:
            scores (Tensor): A tensor of shape (R, 1, ...) that contains
            uncertainty scores with the most uncertain locations having the
            highest uncertainty score.
        """
        if mask_pred.shape[1] == 1:
            gt_class_logits = mask_pred.clone()
        else:
            inds = torch.arange(mask_pred.shape[0], device=mask_pred.device)
            gt_class_logits = mask_pred[inds, labels].unsqueeze(1)
        return -torch.abs(gt_class_logits)

    def get_roi_coords_train(self, mask_pred, labels, cfg):
        """
        Sample points in [0, 1] x [0, 1] coordinate space based on their
        uncertainty. The uncertainties are calculated for each point using
        'uncertainty_func' function that takes point's logit
            prediction as input.
        See PointRend paper for details.

        Args:
            mask_pred (Tensor): A tensor of shape (N, C, mask_height,
            mask_width) for
            class-specific or class-agnostic prediction.
            labels (list): A list of length R that contains either predicted of
            ground truth class for each predicted mask.
        Returns:
            roi_coords (Tensor): A tensor of shape (N, P, 2) that contains the
            coordinates of P sampled points.
        """
        num_points = cfg.num_points
        oversample_ratio = cfg.oversample_ratio
        importance_sample_ratio = cfg.importance_sample_ratio
        assert oversample_ratio >= 1
        assert 0 <= importance_sample_ratio <= 1
        num_rois = mask_pred.shape[0]
        num_sampled = int(num_points * oversample_ratio)
        roi_coords = torch.rand(
            num_rois, num_sampled, 2, device=mask_pred.device)
        point_logits = point_sample(mask_pred, roi_coords, align_corners=False)
        # It is crucial to calculate uncertainty based on the sampled
        # prediction value for the points. Calculating uncertainties of the
        # coarse predictions first and sampling them for points leads to
        # incorrect results.  To illustrate this: assume uncertainty func(
        # logits)=-abs(logits), a sampled point between two coarse
        # predictions with -1 and 1 logits has 0 logits, and therefore 0
        # uncertainty value. However, if we calculate uncertainties for the
        # coarse predictions first, both will have -1 uncertainty,
        # and sampled point will get -1 uncertainty.
        point_uncertainties = self._get_uncertainty(point_logits, labels)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(
            point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_sampled * torch.arange(
            num_rois, dtype=torch.long, device=mask_pred.device)
        idx += shift[:, None]
        roi_coords = roi_coords.view(-1, 2)[idx.view(-1), :].view(
            num_rois, num_uncertain_points, 2)
        if num_random_points > 0:
            rand_roi_coords = torch.rand(
                num_rois, num_random_points, 2, device=mask_pred.device)
            roi_coords = torch.cat((roi_coords, rand_roi_coords), dim=1)
        return roi_coords

    def get_roi_coords_test(self, mask_pred, pred_label, cfg):
        """
        Find `num_points` most uncertain points from `uncertainty_map` grid.

        Args:
            mask_pred (Tensor): A tensor of shape (N, C, mask_height,
            mask_width) for
            class-specific or class-agnostic prediction.

        Returns:
            point_indices (Tensor): A tensor of shape (N, P) that contains
            indices from [0, mask_height x mask_width) of the most uncertain
            points.
            point_coords (Tensor): A tensor of shape (N, P, 2) that contains
            [0, 1] x [0, 1] normalized coordinates of the most uncertain
            points from the mask_height x mask_width grid .
            """
        num_points = cfg.subdivision_num_points
        uncertainty_map = self._get_uncertainty(mask_pred, pred_label)
        num_rois, _, mask_height, mask_width = uncertainty_map.shape
        h_step = 1.0 / float(mask_height)
        w_step = 1.0 / float(mask_width)

        uncertainty_map = uncertainty_map.view(num_rois, -1)
        num_points = min(mask_height * mask_width, num_points)
        point_indices = uncertainty_map.topk(num_points, dim=1)[1]
        point_coords = torch.zeros(
            num_rois,
            num_points,
            2,
            dtype=torch.float,
            device=mask_pred.device)
        point_coords[:, :, 0] = w_step / 2.0 + (point_indices %
                                                mask_width).float() * w_step
        point_coords[:, :, 1] = h_step / 2.0 + (point_indices //
                                                mask_width).float() * h_step
        return point_indices, point_coords
