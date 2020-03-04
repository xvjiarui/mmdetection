from ..registry import DETECTORS
from .mask_rcnn import MaskRCNN


@DETECTORS.register_module
class PointRend(MaskRCNN):
    """PointRend

    http://arxiv.org/abs/1912.08193
    """

    def __init__(self, *args, **kwargs):
        super(PointRend, self).__init__(*args, **kwargs)
