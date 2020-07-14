import argparse
import os.path as osp
import os

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import init_dist, load_checkpoint

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.core import encode_mask_results, tensor2imgs
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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
         'results / save the results) with the argument "--show" or "--show-dir"')

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
        workers_per_gpu=cfg.data.workers_per_gpu,
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
    single_gpu_vis(model, data_loader, args.show, args.show_dir)


hidden_outputs = {}


def context_mask_hook(name):
    def hook(module, input, output):
        x = input[0]
        batch, channel, height, width = x.size()
        # [N, 1, H, W]
        context_mask = module.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, H, W]
        context_mask = module.softmax(context_mask)

        hidden_outputs[name] = context_mask.view(batch, height, width)

    return hook


def register_context_mask_hook(model):
    for module_name, module in model.module.named_modules():
        if 'ContextBlock' in str(
                module.__class__) and 'layer4.1' in module_name:
            module.register_forward_hook(context_mask_hook(module_name))
            print(f'{module_name} is registered')


def single_gpu_vis(model,
                   data_loader,
                   show=False,
                   out_dir=None):
    model.eval()
    register_context_mask_hook(model)

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
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

                    hidden_output_np = hidden_output.detach().cpu().numpy()[0]
                    att_map = hidden_output_np.copy()
                    att_map[hidden_output_np < (np.percentile(hidden_output_np,
                                                              80))] = hidden_output_np.min()
                    att_map[hidden_output_np > (
                        np.percentile(hidden_output_np, 95))] = np.percentile(
                        hidden_output_np, 95)
                    hidden_output_show = mmcv.imresize_like(hidden_output_np,
                                                            img_show)
                    # plt.imshow(hidden_output_show / hidden_output_show.max(), cmap='viridis',
                    #            interpolation='bilinear', vmin=0., vmax=1., alpha=0.5)
                    plt.imshow(hidden_output_show, cmap='jet',
                               interpolation='bilinear', alpha=0.3)
                    if out_dir is not None:
                        dst_dir = osp.join(out_dir, hidden_name)
                        mmcv.mkdir_or_exist(dst_dir)
                        print(
                            f"saving {osp.join(dst_dir, img_meta['ori_filename'])}")
                        plt.savefig(
                            osp.join(dst_dir, img_meta['ori_filename']))
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
