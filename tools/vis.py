import argparse
import math
import os.path as osp
from collections import defaultdict

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet.core import tensor2imgs
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('template_dir')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.show or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--show" or "'
         '--show-dir"')

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    assert not distributed
    model = MMDataParallel(model, device_ids=[0])
    coord_map = defaultdict(list)
    for files in mmcv.scandir(args.template_dir):
        filename, coord = osp.splitext(files)[0].split('_')
        j, i = coord.split('-')
        coord_map[filename].append((int(j), int(i)))
    single_gpu_vis(model, data_loader, args.show, args.show_dir, coord_map)


hidden_outputs = {}


def attn_mask_hook(name):

    def hook(self, input, output):
        x = input[0]
        # Assume `reduction = 1`, then `inter_channels = C`
        # or `inter_channels = C` when `mode="gaussian"`

        # NonLocal1d x: [N, C, H]
        # NonLocal2d x: [N, C, H, W]
        # NonLocal3d x: [N, C, T, H, W]
        n = x.size(0)

        # NonLocal1d g_x: [N, H, C]
        # NonLocal2d g_x: [N, HxW, C]
        # NonLocal3d g_x: [N, TxHxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # NonLocal1d theta_x: [N, H, C], phi_x: [N, C, H]
        # NonLocal2d theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        # NonLocal3d theta_x: [N, TxHxW, C], phi_x: [N, C, TxHxW]
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, self.mode)
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        hidden_outputs[name] = pairwise_weight.view(n, x.size(2), x.size(3),
                                                    x.size(2), x.size(3))

    return hook


def register_context_mask_hook(model):
    for module_name, module in model.module.named_modules():
        if 'NonLocal' in str(module.__class__):
            module.register_forward_hook(attn_mask_hook(module_name))
            print(f'{module_name} is registered')


def single_gpu_vis(model,
                   data_loader,
                   show=False,
                   out_dir=None,
                   coord_map=None):
    model.eval()
    register_context_mask_hook(model)

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        img_metas = data['img_metas'][0].data[0]
        if not osp.splitext(img_metas[0]['ori_filename'])[0] in coord_map:
            prog_bar.update()
            continue
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            for input_j, input_i in coord_map[osp.splitext(
                    img_metas[0]['ori_filename'])[0]]:
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for img, img_meta in zip(imgs, img_metas):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]
                    img_show = mmcv.bgr2rgb(img_show)

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    for hidden_name, hidden_output in hidden_outputs.items():
                        plt.imshow(img_show)
                        sattn = hidden_output.detach().cpu().numpy()[0]
                        fact = 2**round(math.log2(w / sattn.shape[-1]))
                        idx = (input_i // fact, input_j // fact)
                        sattn_map = sattn[idx[0], idx[1], ...]
                        print('sattn_map sum:', sattn_map.sum())
                        print('sattn map shape:', sattn_map.shape)
                        attn_map_copy = sattn_map.copy()
                        sattn_map[attn_map_copy < (np.percentile(
                            attn_map_copy, 80))] = attn_map_copy.min()
                        sattn_map = mmcv.imresize_like(sattn_map, img_show)
                        plt.imshow(
                            sattn_map,
                            cmap='jet',
                            interpolation='bilinear',
                            alpha=0.3)
                        # how much was the original image upsampled before
                        # feeding it to the model
                        scale = ori_h / h
                        # round the position at the downsampling factor
                        x = ((input_j // fact) + 0.5) * fact
                        y = ((input_i // fact) + 0.5) * fact
                        plt.plot([x * scale], [y * scale],
                                 marker='s',
                                 color='r')

                        if out_dir is not None:
                            dst_dir = osp.join(out_dir, hidden_name)
                            mmcv.mkdir_or_exist(dst_dir)
                            img_id, ext = osp.splitext(
                                img_meta['ori_filename'])
                            filename = osp.join(
                                dst_dir, f'{img_id}_{input_j}-{input_i}{ext}')
                            print(f'saving {filename}')
                            plt.savefig(filename)
                        else:
                            plt.title(img_meta['ori_filename'] + hidden_name)
                            plt.show()
                        plt.clf()

        hidden_outputs.clear()

        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()


if __name__ == '__main__':
    main()
