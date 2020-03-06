import torch.nn as nn
from torch.nn.modules.utils import _pair

# from .point_sample import point_sample, roi2point
from .point_sample import generate_grid, poi_align


class SimpleRoIAlign(nn.Module):

    def __init__(self, out_size, spatial_scale):
        super(SimpleRoIAlign, self).__init__()

        self.out_size = _pair(out_size)
        self.spatial_scale = float(spatial_scale)

    # def roi_crop_resize(self, feature, rois):
    #     num_rois = rois.size(0)
    #     channels = feature.size(0)
    #     point_coord = roi2point(rois, self.out_size)
    #     output = point_sample(
    #         feature,
    #         point_coord,
    #         scale_factor=1. / self.spatial_scale,
    #         align_corners=False).transpose(0, 1)
    #
    #     output = output.reshape(num_rois, channels, *self.out_size)
    #
    #     return output

    def forward(self, features, rois):

        # batch_size = features.size(0)
        # roi_feats = features.new_zeros(
        #     rois.size(0), features.size(1), *self.out_size)
        # for batch_ind in range(batch_size):
        #     roi_ind = rois[:, 0] == batch_ind
        #     if roi_ind.any():
        #         roi_feats[roi_ind] = self.roi_crop_resize(
        #             features[batch_ind], rois[roi_ind])

        from mmdet.core import roi2poi
        num_rois = rois.size(0)
        output = poi_align(
            features,
            roi2poi(rois,
                    generate_grid(num_rois, self.out_size,
                                  device=rois.device)),
            scale_factor=1. / self.spatial_scale)

        channels = features.size(1)
        roi_feats = output.reshape(num_rois, channels, *self.out_size)

        return roi_feats

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}'.format(
            self.out_size, self.spatial_scale)
        return format_str
