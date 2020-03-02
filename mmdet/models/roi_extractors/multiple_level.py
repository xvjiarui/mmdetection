from __future__ import division

import torch
import torch.nn as nn

from mmdet import ops
from mmdet.core import force_fp32
from ..registry import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module
class MultipleRoIExtractor(nn.Module):
    """Extract RoI features from multiple level feature maps.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 in_indices=None):
        super(MultipleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.fp16_enabled = False
        self._in_indices = in_indices

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    @property
    def in_indices(self):
        if self._in_indices is not None:
            return self._in_indices
        else:
            return list(range(self.num_inputs))

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5 + 0.5
        x2 = cx + new_w * 0.5 - 0.5
        y1 = cy - new_h * 0.5 + 0.5
        y2 = cy + new_h * 0.5 - 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        if len(rois) > 0:
            offset = 0
            for i in range(num_levels):
                roi_feats_t = self.roi_layers[i](feats[i], rois)
                roi_feats[:, offset:offset + roi_feats_t.size(1)] = roi_feats_t
                offset += roi_feats_t.size(1)
        return roi_feats
