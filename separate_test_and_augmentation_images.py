#
# python -u separate_test_and_augmentation_images.py \
#   --labelbox_output_dir path/to/labelbox_parser/output/ \
#   --image_splitter_output_dir path/to/image_splitter/output/with_labels_only \
#   --output_test_dir path/to/output/stage1_test \
#   --output_augmentation_dir path/to/output/augmentation_raw \
#   --labelbox_class_names "Sulphide/Partial Sulphide" --labelbox_class_names "Pure Quartz Carbonate"
#

import os
import numpy as np
import shutil
import math
import glob
import re
import argparse

TEST_DATASET_PERCENT = 5

parser = argparse.ArgumentParser(description='Separate 5% of total images and masks to be used as test dataset on the trained Mask R-CNN model')
parser.add_argument('-lod', '--labelbox_output_dir', type=str, help='Directory containing Labelbox original images and masks', required=True)
parser.add_argument('-isod', '--image_splitter_output_dir', type=str, help='Directory containing images and masks that are split into squares', required=True)
parser.add_argument('-otd', '--output_test_dir', type=str, help='Base directory of the output where test dataset (images and masks) will be saved to', required=True)
parser.add_argument('-oad', '--output_augmentation_dir', type=str, help='Base directory of the output where non-test dataset (images and masks) will be saved to for augmentation', required=True)
parser.add_argument('-lcn','--labelbox_class_names', action='append', help='Labelbox class names', required=True)
args = parser.parse_args()

def get_splitted_test_files(image_class):
    all_labelbox_files =  list(map(
        (lambda x: os.path.basename(x)),
        glob.glob('{}/images/*_class_{}*'.format(args.labelbox_output_dir, image_class))
    ))

    labelbox_test_files = set(np.random.choice(
        all_labelbox_files, int(math.ceil(TEST_DATASET_PERCENT / 100 * len(all_labelbox_files)))
    ))

    splitted_test_files = []
    for labelbox_test_file in labelbox_test_files:
        splitted_test_files.extend(
            map(
                (lambda x: os.path.basename(x)),
                glob.glob('{}/images/{}*'.format(args.image_splitter_output_dir, os.path.splitext(labelbox_test_file)[0]))
            )
        )
    return splitted_test_files

def normalise_class_name(class_name):
    class_name = re.sub('/', '-', class_name)
    class_name = re.sub(' ', '_', class_name)
    return class_name

splitted_all_files = os.listdir(os.path.join(args.image_splitter_output_dir, 'images'))

splitted_test_files =  []
for class_name in args.labelbox_class_names:
    splitted_test_files.extend(get_splitted_test_files(
        normalise_class_name(class_name)
    ))

# remove any test dataset from previous run
if os.path.exists(args.output_test_dir):
    shutil.rmtree(args.output_test_dir)

# and create clean slate dir
os.makedirs(args.output_test_dir)
for splitted_test_file in splitted_test_files:
    target_image_dir = os.path.join(args.output_test_dir, splitted_test_file, 'images')
    os.makedirs(target_image_dir)

    shutil.copy(
        os.path.join(args.image_splitter_output_dir, 'images', splitted_test_file),
        os.path.join(target_image_dir, splitted_test_file)
    )

    target_mask_dir = os.path.join(args.output_test_dir, splitted_test_file, 'masks')
    os.makedirs(target_mask_dir)

    shutil.copy(
        os.path.join(args.image_splitter_output_dir, 'masks', splitted_test_file),
        os.path.join(target_mask_dir, splitted_test_file)
    )

target_augmentation_images_dir = os.path.join(args.output_augmentation_dir, 'images')
target_augmentation_masks_dir = os.path.join(args.output_augmentation_dir, 'masks')

# remove any augmentation input dirs for images and masks from previous run
if os.path.exists(target_augmentation_images_dir):
    shutil.rmtree(target_augmentation_images_dir)
if os.path.exists(target_augmentation_masks_dir):
    shutil.rmtree(target_augmentation_masks_dir)

# and create clean slate dirs
os.makedirs(target_augmentation_images_dir)
os.makedirs(target_augmentation_masks_dir)

for augmentation_file in (set(splitted_all_files) - set(splitted_test_files)):
    shutil.copy(
        os.path.join(args.image_splitter_output_dir, 'images', augmentation_file),
        os.path.join(target_augmentation_images_dir, augmentation_file)
    )

    shutil.copy(
        os.path.join(args.image_splitter_output_dir, 'masks', augmentation_file),
        os.path.join(target_augmentation_masks_dir, augmentation_file)
    )

