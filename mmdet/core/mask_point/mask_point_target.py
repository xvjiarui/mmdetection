import torch

from mmdet.ops.point_sample import point_sample, roi_point2img_coord


def mask_point_target(point_coords_list, rois_list, pos_assigned_gt_inds_list,
                      gt_masks_list, cfg):
    cfg_list = [cfg for _ in range(len(gt_masks_list))]
    mask_targets = map(mask_point_target_single, point_coords_list, rois_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


def mask_point_target_single(pos_point_coords, pos_rois, pos_assigned_gt_inds,
                             gt_masks, cfg):
    num_pos = pos_rois.size(0)
    num_points = cfg.num_points
    mask_point_targets = []
    if num_pos > 0:
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        gt_masks = torch.from_numpy(gt_masks).to(
            dtype=torch.float, device=pos_rois.device)
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            coords = roi_point2img_coord(pos_rois[i:i + 1],
                                         pos_point_coords[i:i + 1])
            target = point_sample(
                gt_mask.unsqueeze(0),
                coords,
                scale_factor=1.,
                align_corners=False).squeeze(0)
            mask_point_targets.append(target)
        mask_point_targets = torch.cat(mask_point_targets)
    else:
        mask_point_targets = pos_rois.new_zeros((0, num_points))
    return mask_point_targets
