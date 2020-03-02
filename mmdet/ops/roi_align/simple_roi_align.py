import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


class SimpleRoIAlign(nn.Module):

    def __init__(self, out_size, spatial_scale):
        super(SimpleRoIAlign, self).__init__()

        self.out_size = _pair(out_size)
        self.spatial_scale = float(spatial_scale)

    def roi_crop_resize(self, feature, rois):
        num_rois = rois.size(0)
        channels = feature.size(0)
        from mmdet.core import roi2point, point_sample
        point_coord = roi2point(rois, self.out_size)
        height, width = feature.shape[-2:]
        scale = torch.tensor([width, height],
                             device=feature.device) / self.spatial_scale
        grid = point_coord / scale
        output = point_sample(feature, grid, align_corners=False)
        output = output.transpose(0, 1).reshape(num_rois, channels,
                                                *self.out_size)

        return output

    def forward(self, features, rois):

        batch_size = features.size(0)
        roi_feats = features.new_zeros(
            rois.size(0), features.size(1), *self.out_size)
        for batch_ind in range(batch_size):
            roi_ind = rois[:, 0] == batch_ind
            if roi_ind.any():
                roi_feats[roi_ind] = self.roi_crop_resize(
                    features[batch_ind], rois[roi_ind])

        return roi_feats

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}'.format(
            self.out_size, self.spatial_scale)
        return format_str
