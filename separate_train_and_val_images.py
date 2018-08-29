#
# python -u separate_train_and_val_images.py \
#   --input_dir path/to/augmentation/output \
#   --output_dir path/to/output \
#   --labelbox_class_names "Sulphide/Partial Sulphide" --labelbox_class_names "Pure Quartz Carbonate"
#

import os
import numpy as np
import shutil
import math
import glob
import re
import argparse

VAL_DATASET_PERCENT = 5

parser = argparse.ArgumentParser(description='Separate 5% of total augmented images and masks to be used as validation dataset and the rest 95% as training dataset')
parser.add_argument('-id', '--input_dir', type=str, help='Base directory containing all augmented images and masks', required=True)
parser.add_argument('-od', '--output_dir', type=str, help='Base directory of the output', required=True)
parser.add_argument('-lcn','--labelbox_class_names', action='append', help='Labelbox class names', required=True)
args = parser.parse_args()

def get_val_files(class_name, dir, val_percent):
    class_files = list(map(
        (lambda x: os.path.basename(x)),
        glob.glob('{}/images/*_class_{}*'.format(dir, class_name))
    ))

    return set(np.random.choice(
        class_files,
        int(math.ceil(val_percent/ 100 * len(class_files)))
    ))

def normalise_class_name(class_name):
    class_name = re.sub('/', '-', class_name)
    class_name = re.sub(' ', '_', class_name)
    return class_name

train_dir = os.path.join(args.output_dir, 'stage1_train')
val_dir = os.path.join(args.output_dir, 'val')

# remove any dataset from previous run
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(val_dir):
    shutil.rmtree(val_dir)

os.makedirs(train_dir)
os.makedirs(val_dir)

all_files = os.listdir(os.path.join(args.input_dir, 'images'))

val_files = []
for class_name in args.labelbox_class_names:
    val_files.extend(get_val_files(
        normalise_class_name(class_name),
        args.input_dir,
        VAL_DATASET_PERCENT
    ))

for val_file in val_files:
    target_image_dir = os.path.join(val_dir, val_file, 'images')
    os.makedirs(target_image_dir)

    shutil.copy(
        os.path.join(args.input_dir, 'images', val_file),
        os.path.join(target_image_dir, val_file)
    )

    target_mask_dir = os.path.join(val_dir, val_file, 'masks')
    os.makedirs(target_mask_dir)

    shutil.copy(
        os.path.join(args.input_dir, 'masks', val_file),
        os.path.join(target_mask_dir, val_file)
    )

for train_file in (set(all_files) - set(val_files)):
    target_image_dir = os.path.join(train_dir, train_file, 'images')
    os.makedirs(target_image_dir)

    shutil.copy(
        os.path.join(args.input_dir, 'images', train_file),
        os.path.join(target_image_dir, train_file)
    )

    target_mask_dir = os.path.join(train_dir, train_file, 'masks')
    os.makedirs(target_mask_dir)

    shutil.copy(
        os.path.join(args.input_dir, 'masks', train_file),
        os.path.join(target_mask_dir, train_file)
    )
