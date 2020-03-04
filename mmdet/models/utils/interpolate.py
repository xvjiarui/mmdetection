import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.ops.carafe import CARAFEPack


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        scale_factor (int): Upsample ratio
        upsample_kernel (int): Kernel size of Conv layer to expand the channels

    Returns:
        upsampled feature map
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super(PixelShufflePack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        xavier_init(self.upsample_conv, distribution='uniform')

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class Interpolate(nn.Module):
    """A wrapper of F.interpolate.

    """

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode,
                             self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


interpolate_cfg = {
    # format: layer_type: (abbreviation, module)
    'nearest': Interpolate,
    'bilinear': Interpolate,
    'deconv': nn.ConvTranspose2d,
    'conv': nn.Conv2d,
    'pixel_shuffle': PixelShufflePack,
    'carafe': CARAFEPack
}


def build_interpolate_layer(cfg):
    """ Build interpolate layer

    Args:
        cfg (dict): cfg should contain:
            type (str): Identify interpolate layer type.
            interpolate ratio (int): Interpolate ratio
            layer args: args needed to instantiate a interpolate layer.

    Returns:
        layer (nn.Module): Created interpolate layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in interpolate_cfg:
        raise KeyError('Unrecognized interpolate type {}'.format(layer_type))
    else:
        interpolate = interpolate_cfg[layer_type]
        if interpolate is None:
            raise NotImplementedError

    layer = interpolate(**cfg_)
    return layer
