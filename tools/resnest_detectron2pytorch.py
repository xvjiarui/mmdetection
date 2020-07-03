import argparse
from collections import OrderedDict

import mmcv
import numpy as np
import torch


def convert(src, dst, avg_down=False):

    if src.endswith('pth'):
        src_model = torch.load(src)
    else:
        print('latin1')
        src_model = mmcv.load(src, encoding='latin1')

    dst_state_dict = OrderedDict()
    for k, v in src_model.items():
        key_name_split = k.split('.')
        if 'backbone.fpn_lateral' in k:
            if 'norm' in k:
                m_type = 'bn'
            else:
                m_type = 'conv'
            lateral_id = int(key_name_split[1][-1])
            name = f'neck.lateral_convs.{lateral_id-2}.{m_type}.{key_name_split[-1]}'
        elif 'backbone.fpn_output' in k:
            if 'norm' in k:
                m_type = 'bn'
            else:
                m_type = 'conv'
            lateral_id = int(key_name_split[1][-1])
            name = f'neck.fpn_convs.{lateral_id-2}.{m_type}.{key_name_split[-1]}'
        elif 'backbone.bottom_up.stem.conv1_' in k:
            conv_id = int(key_name_split[3][-1])
            if 'norm' in k:
                stem_id = (conv_id - 1) * 3 + 1
            else:
                stem_id = (conv_id - 1) * 3
            name = f'backbone.stem.{stem_id}.{key_name_split[-1]}'
        elif 'backbone.bottom_up.stem.conv1.norm.' in k:
            name = f'backbone.bn1.{key_name_split[-1]}'
        elif 'backbone.bottom_up.stem.conv1.' in k:
            name = f'backbone.conv1.{key_name_split[-1]}'
        elif 'backbone.bottom_up.res' in k:
            weight_type = key_name_split[-1]
            res_id = int(key_name_split[2][-1]) - 1
            # deal with short cut
            if 'shortcut' in key_name_split[4]:
                if 'shortcut' == key_name_split[-2]:
                    if avg_down:
                        name = f'backbone.layer{res_id}.{key_name_split[3]}.downsample.1.{key_name_split[-1]}'
                    else:
                        name = f'backbone.layer{res_id}.{key_name_split[3]}.downsample.0.{key_name_split[-1]}'
                elif 'norm' == key_name_split[-2]:
                    if avg_down:
                        name = f'backbone.layer{res_id}.{key_name_split[3]}.downsample.2.{key_name_split[-1]}'
                    else:
                        name = f'backbone.layer{res_id}.{key_name_split[3]}.downsample.1.{key_name_split[-1]}'
                else:
                    print(f'Unvalid key {k}')
            # deal with conv
            elif 'conv' in key_name_split[
                    4] and not 'norm' == key_name_split[-2]:
                # deal with ResNeSt
                conv_id = int(key_name_split[4][-1])
                if conv_id == 2 and 'conv.' in k:
                    name = f'backbone.layer{res_id}.{key_name_split[3]}.conv{conv_id}.conv.{key_name_split[-1]}'
                elif conv_id == 2 and 'bn' in k:
                    bn_id = int(key_name_split[-2][-1])
                    name = f'backbone.layer{res_id}.{key_name_split[3]}.conv{conv_id}.bn{bn_id}.{key_name_split[-1]}'
                elif conv_id == 2 and 'fc' in k:
                    fc_id = int(key_name_split[-2][-1])
                    name = f'backbone.layer{res_id}.{key_name_split[3]}.conv{conv_id}.fc{fc_id}.{key_name_split[-1]}'
                else:
                    name = f'backbone.layer{res_id}.{key_name_split[3]}.conv{conv_id}.{key_name_split[-1]}'
            # deal with BN
            elif key_name_split[-2] == 'norm':
                conv_id = int(key_name_split[-3][-1])
                name = f'backbone.layer{res_id}.{key_name_split[3]}.bn{conv_id}.{key_name_split[-1]}'
            else:
                print(f'{k} is invalid')
        elif 'proposal_generator.anchor_generator' in k:
            continue
        elif 'rpn' in k:
            if 'conv' in key_name_split[2]:
                name = f'rpn_head.rpn_conv.{key_name_split[-1]}'
            elif 'objectness_logits' in key_name_split[2]:
                name = f'rpn_head.rpn_cls.{key_name_split[-1]}'
            elif 'anchor_deltas' in key_name_split[2]:
                name = f'rpn_head.rpn_reg.{key_name_split[-1]}'
            else:
                print(f'{k} is invalid')
        elif 'roi_heads' in k:
            if key_name_split[1] == 'box_head':
                if 'fc' in key_name_split[2]:
                    fc_id = int(key_name_split[2][-1]) - 1
                    name = f'roi_head.bbox_head.shared_fcs.{fc_id}.{key_name_split[-1]}'
                elif 'conv' in key_name_split[2]:
                    if 'norm' in k:
                        m_type = 'bn'
                    else:
                        m_type = 'conv'
                    conv_id = int(key_name_split[2][-1]) - 1
                    name = f'roi_head.bbox_head.shared_convs.{conv_id}.{m_type}.{key_name_split[-1]}'
                else:
                    print(f'{k} is invalid')
            elif 'cls_score' == key_name_split[2]:
                name = f'roi_head.bbox_head.fc_cls.{key_name_split[-1]}'
            elif 'bbox_pred' == key_name_split[2]:
                name = f'roi_head.bbox_head.fc_reg.{key_name_split[-1]}'
            elif 'mask_fcn' in key_name_split[2]:
                if 'norm' in k:
                    m_type = 'bn'
                else:
                    m_type = 'conv'
                conv_id = int(key_name_split[2][-1]) - 1
                name = f'roi_head.mask_head.convs.{conv_id}.{m_type}.{key_name_split[-1]}'
            elif 'deconv' in key_name_split[2]:
                name = f'roi_head.mask_head.upsample.{key_name_split[-1]}'
            elif 'roi_heads.mask_head.predictor' in k:
                name = f'roi_head.mask_head.conv_logits.{key_name_split[-1]}'
            elif 'roi_heads.mask_coarse_head.reduce_spatial_dim_conv' in k:
                name = f'roi_head.mask_head.downsample_conv.conv.{key_name_split[-1]}'
            elif 'roi_heads.mask_coarse_head.prediction' in k:
                name = f'roi_head.mask_head.fc_logits.{key_name_split[-1]}'
            elif key_name_split[1] == 'mask_coarse_head':
                fc_id = int(key_name_split[2][-1]) - 1
                name = f'roi_head.mask_head.fcs.{fc_id}.{key_name_split[-1]}'
            elif 'roi_heads.mask_point_head.predictor' in k:
                name = f'roi_head.point_head.fc_logits.{key_name_split[-1]}'
            elif key_name_split[1] == 'mask_point_head':
                fc_id = int(key_name_split[2][-1]) - 1
                name = f'roi_head.point_head.fcs.{fc_id}.conv.{key_name_split[-1]}'
            else:
                print(f'{k} is invalid')
        else:
            print(f'{k} is not converted!!')

        if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
            raise ValueError(
                'Unsupported type found in checkpoint! {}: {}'.format(
                    k, type(v)))
        if name.startswith('backbone.'):
            print(f'{name} -> {name[9:]}')
            name = name[9:]
            if not isinstance(v, torch.Tensor):
                dst_state_dict[name] = torch.from_numpy(v)
            else:
                dst_state_dict[name] = v
        else:
            print(f'{name} not in backbone')

    mmdet_model = dict(state_dict=dst_state_dict, meta=dict())
    torch.save(mmdet_model, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    parser.add_argument(
        '--avg_down', help='used avg_down or not', action='store_true')
    args = parser.parse_args()
    convert(args.src, args.dst, args.avg_down)


if __name__ == '__main__':
    main()
