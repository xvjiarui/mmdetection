import argparse
import os.path as osp

import mmcv
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('img_dir', help='img config directory')
    parser.add_argument('out_dir', help='output config directory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.out_dir)
    new_width = 400
    new_height = 300
    for img_file in mmcv.scandir(args.img_dir):
        im = Image.open(osp.join(args.img_dir, img_file))
        width, height = im.size  # Get dimensions

        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        # Crop the center of the image
        im = im.crop((left, top, right, bottom))
        im.save(osp.join(args.out_dir, img_file))


if __name__ == '__main__':
    main()
