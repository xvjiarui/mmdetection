import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init, xavier_init

from ..builder import NECKS, build_backbone
from .fpn import FPN


class ASPP(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling)

    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by this module
        dilations (tuple): Dilations of the four branches.
            Default: (1, 3, 6, 1)
    """

    def __init__(self, in_channels, out_channels, dilations=(1, 3, 6, 1)):
        super().__init__()
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for aspp_idx in range(len(dilations)):
            dilation = dilations[aspp_idx]
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out


@NECKS.register_module(name='RFP')
class RFP(FPN):
    """RFP (Recursive Feature Pyramid)

    This is an implementation of RFP in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)

    Args:
        rfp_steps (int): Number of unrolled steps of RFP
        rfp_backbone (dict): Configuration of the backbone
        kwargs: Args for FPN
        aspp_dilations (tuple): Dilations of the four branches.
            Default: (1, 3, 6, 1)
    """

    def __init__(self,
                 rfp_steps,
                 rfp_backbone,
                 aspp_dilations=(1, 3, 6, 1),
                 **kwargs):
        super().__init__(**kwargs)
        self.rfp_steps = rfp_steps
        neck_out_channels = kwargs['out_channels']
        self.rfp_modules = nn.ModuleList()
        for rfp_idx in range(1, rfp_steps):
            rfp_module = build_backbone(rfp_backbone)
            self.rfp_modules.append(rfp_module)
        self.rfp_aspp = ASPP(neck_out_channels, neck_out_channels // 4,
                             aspp_dilations)
        self.rfp_weight = nn.Conv2d(
            neck_out_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        constant_init(self.rfp_weight, 0)

    def init_weights(self):
        for convs in [self.lateral_convs, self.fpn_convs]:
            for m in convs.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
        for rfp_idx in range(self.rfp_steps - 1):
            self.rfp_modules[rfp_idx].init_weights(
                self.rfp_modules[rfp_idx].pretrained)

    def forward(self, inputs):
        inputs = list(inputs)
        assert len(inputs) == len(self.in_channels) + 1
        img = inputs.pop(0)
        x = super().forward(tuple(inputs))
        for rfp_idx in range(self.rfp_steps - 1):
            rfp_feats = [x[0]] + list(
                self.rfp_aspp(x[i]) for i in range(1, len(x)))
            x_idx = self.rfp_modules[rfp_idx].rfp_forward(img, rfp_feats)
            x_idx = super().forward(x_idx)
            x_new = []
            for ft_idx in range(len(x_idx)):
                add_weight = torch.sigmoid(self.rfp_weight(x_idx[ft_idx]))
                x_new.append(add_weight * x_idx[ft_idx] +
                             (1 - add_weight) * x[ft_idx])
            x = x_new
        return x
