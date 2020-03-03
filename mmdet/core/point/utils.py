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


def calculate_uncertainty(logits, classes):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction
    in 'logits' for the foreground class in `classes`.

    Args:
        logits (Tensor): A tensor of shape (R, C, ...) or (R, 1, ...) for
        class-specific or class-agnostic, where R is the total number of
        predicted masks in all images and C is the number of foreground classes
        The values are logits. classes (list):
        A list of length R that contains either predicted of ground truth class
        for eash predicted mask.

    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contain uncertainty
        scores with the most uncertain locations having the highest uncertainty
        score.
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device),
            classes].unsqueeze(1)
    return -torch.abs(gt_class_logits)


def get_uncertain_point_coords_with_randomness(logits, gt_labels, num_points,
                                               oversample_ratio,
                                               importance_sample_ratio):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their
    uncertainty. The uncertainties are calculated for each point using
    'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or
        (N, 1, Hmask, Wmask) for class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P)
        or (N, 1, P) that contains logit predictions for P points and returns
        their uncertainties as a Tensor of shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via
        importance sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the
        coordinates of P sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=logits.device)
    point_logits = point_sample(logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction
    # value for the points. # Calculating uncertainties of the coarse
    # predictions first and sampling them for points leads # to incorrect
    # results.  To illustrate this: assume uncertainty
    # func(logits)=-abs(logits), a sampled point between two coarse predictions
    # with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and sampled point will get -1 uncertainty.
    point_uncertainties = calculate_uncertainty(point_logits, gt_labels)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(
        point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        num_boxes, dtype=torch.long, device=logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = torch.cat([
            point_coords,
            torch.rand(num_boxes, num_random_points, 2, device=logits.device)
        ],
                                 dim=1)
    return point_coords


def get_uncertain_point_coords(logits, pred_label, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains
        uncertainty values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices
        from [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1]
        x [0, 1] normalized coordinates of the most uncertain points
        from the H x W grid. """
    uncertainty_map = calculate_uncertainty(logits, pred_label)
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(
        uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(
        R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(
        torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(
        torch.float) * h_step
    return point_indices, point_coords
