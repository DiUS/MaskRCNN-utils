#
# python -u setup_stage2_test_images.py \
#   --input_dir path/to/split/images \
#   --output_dir path/to/output
#

import os
import shutil
import glob
import argparse

parser = argparse.ArgumentParser(description='Setup stage2 test images for Mask R-CNN inference')
parser.add_argument('-id', '--input_dir', type=str, help='Directory containing split images', required=True)
parser.add_argument('-od', '--output_dir', type=str, help='Base directory where the stage2 test images will be saved to', required=True)
args = parser.parse_args()

stage2_test_dir = os.path.join(args.output_dir, 'stage2_test')
if os.path.exists(stage2_test_dir):
    shutil.rmtree(stage2_test_dir)

os.makedirs(stage2_test_dir)

image_file_list = glob.glob("{}/*_inst_0_class_*.png".format(args.input_dir))
if len(image_file_list) < 1:
    image_file_list = glob.glob("{}/*.png".format(args.input_dir))

for idx in range(len(image_file_list)):
    full_name = image_file_list[idx]
    base_name = os.path.basename(full_name)

    if idx % 500 == 0:
        print("processing {} ({} of {})".format(base_name, idx+1, len(image_file_list)))

    images_dir = os.path.join(stage2_test_dir, base_name, 'images')

    os.makedirs(images_dir)
    shutil.copy(full_name, os.path.join(images_dir, base_name))
