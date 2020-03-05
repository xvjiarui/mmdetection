import torch

from mmdet.ops.point_sample import poi_align


def mask_point_target(pos_pois_list, pos_assigned_gt_inds_list, gt_masks_list,
                      cfg):
    cfg_list = [cfg for _ in range(len(gt_masks_list))]
    mask_targets = map(mask_point_target_single, pos_pois_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


def mask_point_target_single(pos_pois, pos_assigned_gt_inds, gt_masks, cfg):
    num_pos = pos_pois.size(0)
    num_points = cfg.num_points
    if num_pos > 0:
        device = pos_pois.device
        gt_masks_th = (
            torch.from_numpy(gt_masks).to(device).index_select(
                0, pos_assigned_gt_inds).to(dtype=pos_pois.dtype))
        targets = poi_align(gt_masks_th.unsqueeze(1), pos_pois).squeeze(1)
        mask_point_targets = (targets >= 0.5).float()
    else:
        mask_point_targets = pos_pois.new_zeros((0, num_points))
    return mask_point_targets
