import torch

from mmdet.ops import affine_grid, grid_sample


def normalize(grid):
    """
    convert from range [-1, 1] to [0, 1]
    """
    return (grid + 1.0) / 2.0


def denormalize(grid):
    """
    convert from range [0, 1] to [-1, 1]
    """
    return grid * 2.0 - 1


def generate_grid(num_grid, size, device):
    """
    Generate regular square grid of points in [0, 1] x [0, 1] coordinate space.

    Args:
        num_grid (int): The number of grids to sample, one for each region.
        size (tuple(int, int)): The side size of the regular grid.
        device (torch.device): Desired device of returned tensor.

    Returns:
        (Tensor): A tensor of shape (num_grid, size[0]*size[1], 2) that
        contains coordinates for the regular grids.
    """
    affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], device=device)
    grid = affine_grid(
        affine_trans, torch.Size((1, 1, *size)), align_corners=False)
    grid = normalize(grid)
    return grid.view(1, -1, 2).expand(num_grid, -1, -1)


def roi_point2img_coord(rois, roi_points):
    with torch.no_grad():
        assert roi_points.size(0) == rois.size(0)
        img_coords = roi_points.clone()
        img_coords[:, :, 0] = img_coords[:, :, 0] * (
            rois[:, None, 2] - rois[:, None, 0])
        img_coords[:, :, 1] = img_coords[:, :, 1] * (
            rois[:, None, 3] - rois[:, None, 1])
        img_coords[:, :, 0] += rois[:, None, 0]
        img_coords[:, :, 1] += rois[:, None, 1]
    return img_coords


def roi2point(rois, out_size):
    """
    Convert rois to image-level point coordinates with out_size.

    Args:
        rois (Tensor, [batch_ind, x1, y1, x2, y2]): roi coordinate.
        out_size (tuple(int, int): The side size of the regular grid.

    Returns:
        point_coord (Tensor): shape (num_rois, out_size[0] * out_size[1], 2)
        that contains image-normalized coordinates of grid points.
    """
    with torch.no_grad():
        roi_point = generate_grid(rois.size(0), out_size, rois.device)
        img_coord = roi_point2img_coord(rois, roi_point)
    return img_coord


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`grid_sample` to support 3D point_coords tensors
    Unlike :function:`torch.nn.functional.grid_sample` it assumes point_coords
    to lie inside [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features
        map on a H x W grid. point_coords (Tensor): A tensor of shape (N, P, 2)
        or (N, Hgrid, Wgrid, 2) that contains [0, 1] x [0, 1] normalized point
        coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid)
        that contains features for points in `point_coords`. The features are
        obtained via bilinear interplation from `input` the same way as
        :function:`torch.nn.functional.grid_sample`.
    """
    add_batch = False
    if input.dim() == 3:
        add_batch = True
        input = input.unsqueeze(0)
        point_coords = point_coords.unsqueeze(0)
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = grid_sample(input, denormalize(point_coords), **kwargs)
    if add_dim:
        output = output.squeeze(3)
    if add_batch:
        output = output.squeeze(0)
    return output
