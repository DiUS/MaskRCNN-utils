import numpy as np
import imageio
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='Seperate out images with labels.')
parser.add_argument('-id','--input_dir', type=str, help='Base directory containing images and masks with and without labels.', required=True)
parser.add_argument('-od','--output_dir', type=str, help='Base directory of the output where images and masks with labels will be saved into.', required=True)
args = parser.parse_args()

input_images_dir = os.path.join(args.input_dir, 'images')
input_masks_dir = os.path.join(args.input_dir, 'masks')

output_images_dir = os.path.join(args.output_dir, 'images')
output_masks_dir = os.path.join(args.output_dir, 'masks')

if os.path.exists(output_images_dir):
    shutil.rmtree(output_images_dir)

if os.path.exists(output_masks_dir):
    shutil.rmtree(output_masks_dir)

os.makedirs(output_images_dir)
os.makedirs(output_masks_dir)

for filename in os.listdir(input_masks_dir):
    input_image_file_path = '{}/{}'.format(input_images_dir, filename)
    input_mask_file_path = '{}/{}'.format(input_masks_dir, filename)

    mask = imageio.imread(input_mask_file_path)
    if np.count_nonzero(mask) > 0:
        output_image_file_path = '{}/{}'.format(output_images_dir, filename)
        output_mask_file_path = '{}/{}'.format(output_masks_dir, filename)
        shutil.move(input_image_file_path, output_image_file_path)
        shutil.move(input_mask_file_path, output_mask_file_path)

