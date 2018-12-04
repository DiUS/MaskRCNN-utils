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

def get_files_for_class(class_name):
    class_file_list = glob.glob("{}/images/*_class_{}.png".format(args.input_dir, class_name))
    number_to_select = int(round(VALIDATION_DATASET_PERCENT/100.0 * len(class_file_list)))
    print("selecting {} images from the class {} for the validation set".format(number_to_select, class_name))
    file_list = [os.path.basename(item) for item in random.sample(class_file_list, number_to_select)]
    return file_list

all_files = [os.path.basename(item) for item in glob.glob("{}/images/*.png".format(args.input_dir))]

validation_files = []
for class_name in args.labelbox_class_names:
    validation_files += get_files_for_class(class_name)

# remove datasets from a previous run
if os.path.exists(args.output_training_dir):
    shutil.rmtree(args.output_training_dir)
if os.path.exists(args.output_validation_dir):
    shutil.rmtree(args.output_validation_dir)

# and create clean directories
os.makedirs("{}/images".format(args.output_training_dir))
os.makedirs("{}/masks".format(args.output_training_dir))
os.makedirs("{}/images".format(args.output_validation_dir))
os.makedirs("{}/masks".format(args.output_validation_dir))

# copy each validation set image and its mask to the validation set directory
for f in validation_files:
    shutil.copy("{}/images/{}".format(args.input_dir, f), "{}/images/{}".format(args.output_validation_dir, f))
    shutil.copy("{}/masks/{}".format(args.input_dir, f), "{}/masks/{}".format(args.output_validation_dir, f))

# copy each training set image and its mask to the training set directory
for f in (set(all_files) - set(validation_files)):
    shutil.copy("{}/images/{}".format(args.input_dir, f), "{}/images/{}".format(args.output_training_dir, f))
    shutil.copy("{}/masks/{}".format(args.input_dir, f), "{}/masks/{}".format(args.output_training_dir, f))
