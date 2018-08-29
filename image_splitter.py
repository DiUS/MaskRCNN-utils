#
# python -u image_splitter.py -id path/to/labebox_parser/output/ -od path/to/output/
#

from pathlib import Path
import os
import imageio
from skimage import io
import numpy as np
import shutil
import argparse

parser = argparse.ArgumentParser(description='Split images and masks into squares')
parser.add_argument('-id', '--input_dir', type=str, help='Base directory containing original images and masks', required=True)
parser.add_argument('-od', '--output_dir', type=str, help='Base directory of the output where split images and masks will be saved to.', required=True)
args = parser.parse_args()

all_output_dir = os.path.join(args.output_dir, 'all')
with_labels_only_output_dir = os.path.join(args.output_dir, 'with_labels_only')

if os.path.exists(all_output_dir):
    shutil.rmtree(all_output_dir)

if os.path.exists(with_labels_only_output_dir):
    shutil.rmtree(with_labels_only_output_dir)

os.makedirs(os.path.join(all_output_dir, 'images'))
os.makedirs(os.path.join(all_output_dir, 'masks'))
os.makedirs(os.path.join(with_labels_only_output_dir, 'images'))
os.makedirs(os.path.join(with_labels_only_output_dir, 'masks'))

def crop(filepath):
    img = io.imread(filepath)
    h, w = img.shape[:2]

    cropped_images = []
    for j in range(0, w, h):
        cropped_images.append(img[:, j:j+h])

    return cropped_images

def copy_labelled_images_and_masks(from_dir, to_dir):
    from_images_dir = os.path.join(from_dir, 'images')
    from_masks_dir = os.path.join(from_dir, 'masks')

    to_images_dir = os.path.join(to_dir, 'images')
    to_masks_dir = os.path.join(to_dir, 'masks')

    for filename in os.listdir(os.path.join(from_masks_dir)):
        from_image_file_path = '{}/{}'.format(from_images_dir, filename)
        from_mask_file_path = '{}/{}'.format(from_masks_dir, filename)

        mask = imageio.imread(from_mask_file_path)
        if np.count_nonzero(mask) > 0:
            to_image_file_path = '{}/{}'.format(to_images_dir, filename)
            to_mask_file_path = '{}/{}'.format(to_masks_dir, filename)
            shutil.copy(from_image_file_path, to_image_file_path)
            shutil.copy(from_mask_file_path, to_mask_file_path)

for images_or_masks in ['images', 'masks']:
    dir = os.path.join(args.input_dir, images_or_masks)
    if not os.path.exists(dir) or not os.path.isdir(dir):
        continue

    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)

        cropped_images = crop(filepath)

        for i, cropped_image in enumerate(cropped_images):
            new_filename = '{}_seg_{}.png'.format(os.path.splitext(filename)[0], i)
            image_output_path = os.path.join(all_output_dir, images_or_masks, new_filename)
            imageio.imwrite(image_output_path, cropped_image)
        # break

input_masks_dir = os.path.join(args.input_dir, 'masks')
if os.path.exists(input_masks_dir) and os.listdir(input_masks_dir):
    copy_labelled_images_and_masks(all_output_dir, with_labels_only_output_dir)
