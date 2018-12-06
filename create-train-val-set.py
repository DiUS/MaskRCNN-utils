import os
import numpy as np
import shutil
import math
import glob
import re
import argparse
import random

VALIDATION_DATASET_PERCENT = 5 # taken from the non-test set

parser = argparse.ArgumentParser(description='Take 5 per cent of the non-test set to be used as validation dataset on the trained Mask R-CNN model')
parser.add_argument('-id', '--input_dir', type=str, help='Directory containing the images and masks', required=True)
parser.add_argument('-otd', '--output_training_dir', type=str, help='Base directory where the training dataset (images and masks) will be saved.', required=True)
parser.add_argument('-ovd', '--output_validation_dir', type=str, help='Base directory where the validation dataset (images and masks) will be saved.', required=True)
parser.add_argument('-lcn','--labelbox_class_names', action='append', help='Labelbox class names', required=True)
args = parser.parse_args()

def get_image_ids_for_class(class_name):
    class_image_id_list = glob.glob("{}/*_class_{}".format(args.input_dir, class_name))
    number_to_select = int(round(VALIDATION_DATASET_PERCENT/100.0 * len(class_image_id_list)))
    print("selecting {} images from the class {} for the validation set".format(number_to_select, class_name))
    image_id_list = list(random.sample(class_image_id_list, number_to_select))
    return image_id_list

all_image_ids = glob.glob("{}/*_class_*".format(args.input_dir))

validation_image_ids = []
for class_name in args.labelbox_class_names:
    validation_image_ids += get_image_ids_for_class(class_name)

# remove datasets from a previous run
if os.path.exists(args.output_training_dir):
    shutil.rmtree(args.output_training_dir)
if os.path.exists(args.output_validation_dir):
    shutil.rmtree(args.output_validation_dir)

# copy each validation set image and its mask to the validation set directory
for image_id in validation_image_ids:
    shutil.copytree("{}/{}".format(args.input_dir, image_id), "{}/{}".format(args.output_validation_dir, image_id))

# copy each training set image and its mask to the training set directory
for image_id in (set(all_image_ids) - set(validation_image_ids)):
    shutil.copytree("{}/{}".format(args.input_dir, image_id), "{}/{}".format(args.output_training_dir, image_id))
