import torch
import torch.nn as nn

from mmdet.core import force_fp32, point_sample, roi_point2img_coord
from ..registry import POINT_EXTRACTORS


@POINT_EXTRACTORS.register_module
class MultiplePointExtractor(nn.Module):
    """Extract RoI features from multiple level feature maps.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
    """

    def __init__(self, out_channels, featmap_strides, in_indices=None):
        super(MultiplePointExtractor, self).__init__()
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

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, points):
        batch_size = feats[0].size(0)
        num_levels = len(feats)
        point_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, points.size(1))
        for batch_ind in range(batch_size):
            offset = 0
            for i in range(num_levels):
                height, width = feats[i].shape[-2:]
                scale = torch.tensor(
                    [width, height],
                    device=feats[i].device) * self.featmap_strides[i]
                roi_inds = rois[:, 0] == batch_ind
                if roi_inds.any():
                    coords = roi_point2img_coord(rois[roi_inds],
                                                 points[roi_inds])
                    grid = coords / scale
                    point_feats_t = point_sample(
                        feats[i][batch_ind], grid,
                        align_corners=False).transpose(0, 1)
                    point_feats[roi_inds, offset:offset +
                                point_feats_t.size(1)] = point_feats_t
                    offset += point_feats_t.size(1)
        return point_feats
