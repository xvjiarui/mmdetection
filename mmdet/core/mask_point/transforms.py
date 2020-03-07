import torch

from mmdet.ops.point_sample import roi_coord2img_coord


def roi2poi(rois, points):
    """Convert roi format to poi format.

    Args:
        rois (Tensor): shape (n, 5)
        points (Tensor): shape (n, p, 2)

    Returns:
        Tensor: shape (n, p, 3), [batch_ind, x, y]
    """
    assert rois.dim() == 2 and rois.size(1) == 5
    assert points.dim() == 3 and points.size(2) == 2
    assert rois.size(0) == points.size(0)
    num_points = points.size(1)
    pois = roi_coord2img_coord(rois, points)
    # add batch dim
    pois = torch.cat(
        (rois[:, 0].view(-1, 1, 1).expand(-1, num_points, -1), pois), dim=2)

    return pois


def poi2list(pois):
    poi_list = []
    img_ids = torch.unique(pois[:, 0, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (pois[:, 0, 0] == img_id.item())
        poi_list.append(pois[inds])
    return poi_list
