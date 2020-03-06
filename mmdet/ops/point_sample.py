import torch

from .affine_grid import affine_grid
from .grid_sampler import grid_sample


def normalize(grid):
    """
    convert from range [-1, 1] to [0, 1]
    """
    return (grid + 1.0) / 2.0


def denormalize(grid):
    """
    convert from range [0, 1] to [-1, 1]
    """
    return grid * 2.0 - 1.0


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


def roi_coord2img_coord(rois, points):
    with torch.no_grad():
        assert points.size(0) == rois.size(0)
        assert rois.dim() == 2
        assert points.dim() == 3
        # remove batch idx
        if rois.size(1) == 5:
            rois = rois[:, 1:]
        img_coords = points.clone()
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
        grid = generate_grid(rois.size(0), out_size, rois.device)
        img_coord = roi_coord2img_coord(rois, grid)
    return img_coord


def point_sample(input, point_coords, scale_factor=None, **kwargs):
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
    if scale_factor is not None:
        h, w = input.shape[-2:]
        scale = torch.tensor([w, h], dtype=torch.float,
                             device=input.device) * scale_factor
        scale = scale.view(1, 1, 1, 2)
        point_coords = point_coords / scale
    output = grid_sample(input, denormalize(point_coords), **kwargs)
    if add_dim:
        output = output.squeeze(3)
    if add_batch:
        output = output.squeeze(0)
    return output


def poi_align(input, pois, scale_factor=1.):
    assert input.dim() == 4
    assert pois.dim() == 3 and pois.size(2) == 3
    num_rois = pois.size(0)
    num_points = pois.size(1)
    batch_size = input.size(0)
    output = input.new_zeros((num_rois, input.size(1), num_points))
    if num_rois > 0:
        # pad to fit batch
        padded_pois = pois.new_zeros((batch_size, num_rois, num_points, 3))
        for batch_ind in range(batch_size):
            inds = pois[:, 0, 0].long() == batch_ind
            num_batch_rois = inds.sum().long()
            # TODO: inds.any() triger CUDA error
            # if inds.any():
            if num_batch_rois > 0:
                padded_pois[batch_ind, :num_batch_rois] = pois[inds]
        padded_output = point_sample(
            input, padded_pois[:, :, :, 1:], scale_factor=scale_factor)
        for batch_ind in range(batch_size):
            inds = pois[:, 0, 0].long() == batch_ind
            num_batch_rois = inds.sum().long()
            if num_batch_rois > 0:
                # if inds.any():
                output[inds] = padded_output[
                    batch_ind, :, :num_batch_rois].transpose(0,
                                                             1).contiguous()

    return output
