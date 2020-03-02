import torch

from . import point_sample


def point_target(point_coords, mask_targets, cfg):
    point_targets = point_sample(
        mask_targets.to(torch.float32).unsqueeze(1),
        point_coords,
        align_corners=False).squeeze(1)
    return point_targets
