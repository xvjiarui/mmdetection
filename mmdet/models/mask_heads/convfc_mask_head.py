import torch.nn as nn

from mmdet.core import auto_fp16
from ..registry import HEADS
from .fcn_mask_head import FCNMaskHead


@HEADS.register_module
class ConvFCMaskHead(FCNMaskHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *arg, **kwarg):
        super(ConvFCMaskHead, self).__init__(*arg, **kwarg)
        self.num_fcs = num_fcs
        assert self.num_fcs > 0
        self.fc_out_channels = fc_out_channels
        self.conv_logits = None

        self.output_size = (int(self.roi_feat_size[0] * self.scale_factor),
                            int(self.roi_feat_size[1] * self.scale_factor))
        self.output_area = self.output_size[0] * self.output_size[1]

        last_layer_dim = self.conv_out_channels * self.output_area

        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            fc_in_channels = (
                last_layer_dim if i == 0 else self.fc_out_channels)
            self.fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        last_layer_dim = self.fc_out_channels
        output_channels = self.num_classes * self.output_area
        self.fc_logit = nn.Linear(last_layer_dim, output_channels)

    def init_weights(self):
        super(ConvFCMaskHead, self).init_weights()
        for m in self.fcs.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fc_logit.weight, 0, 0.001)
        nn.init.constant_(self.fc_logit.bias, 0)

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)

        if self.interpolate is not None:
            x = self.interpolate(x)
            if 'conv' in self.interpolate_method:
                x = self.relu(x)

        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_pred = self.fc_logit(x).view(
            x.size(0), self.num_classes, *self.output_size)
        return mask_pred
