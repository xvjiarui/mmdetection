import torch.nn as nn
from torch.nn.modules.utils import _pair

from .point_sample import generate_grid, poi_align


class SimpleRoIAlign(nn.Module):

    def __init__(self, out_size, spatial_scale):
        super(SimpleRoIAlign, self).__init__()

        self.out_size = _pair(out_size)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):

        from mmdet.core import roi2poi
        num_rois = rois.size(0)
        grid = generate_grid(num_rois, self.out_size, device=rois.device)
        output = poi_align(
            features,
            roi2poi(rois, grid),
            scale_factor=1. / self.spatial_scale)

        channels = features.size(1)
        roi_feats = output.reshape(num_rois, channels, *self.out_size)

        return roi_feats

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}'.format(
            self.out_size, self.spatial_scale)
        return format_str
