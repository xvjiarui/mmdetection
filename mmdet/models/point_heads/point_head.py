import torch
import torch.nn as nn

from mmdet.core import mask_point_target, poi2list
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

    def forward(self, coarse_features, fine_grained_features):
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
