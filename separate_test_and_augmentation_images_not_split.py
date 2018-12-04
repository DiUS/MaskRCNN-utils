import os
import numpy as np
import shutil
import math
import glob
import re
import argparse
import random

TEST_DATASET_PERCENT = 5

parser = argparse.ArgumentParser(description='Separate 5% of total images and masks to be used as test dataset on the trained Mask R-CNN model')
parser.add_argument('-lod', '--labelbox_output_dir', type=str, help='Directory containing Labelbox original images and masks', required=True)
parser.add_argument('-otd', '--output_test_dir', type=str, help='Base directory of the output where test dataset (images and masks) will be saved to', required=True)
parser.add_argument('-oad', '--output_augmentation_dir', type=str, help='Base directory of the output where non-test dataset (images and masks) will be saved to for augmentation', required=True)
parser.add_argument('-lcn','--labelbox_class_names', action='append', help='Labelbox class names', required=True)
args = parser.parse_args()

def get_test_files(class_name):
    class_file_list = glob.glob("{}/images/*_class_{}.png".format(args.labelbox_output_dir, class_name))
    number_to_select = int(TEST_DATASET_PERCENT/100.0 * len(class_file_list))
    print("selecting {} images from the class {} for the test set".format(number_to_select, class_name))
    test_files = [os.path.basename for item in random.sample(class_file_list, number_to_select)]
    return test_files

all_files = os.listdir(os.path.join(args.labelbox_output_dir, 'images'))

test_files = []
for class_name in args.labelbox_class_names:
    test_files.append(get_test_files(class_name))

# remove any test dataset from previous run
if os.path.exists(args.output_test_dir):
    shutil.rmtree(args.output_test_dir)
if os.path.exists(args.output_augmentation_dir):
    shutil.rmtree(args.output_augmentation_dir)

# and create clean directories
os.makedirs("{}/images".format(args.output_test_dir))
os.makedirs("{}/masks".format(args.output_test_dir))
os.makedirs("{}/images".format(args.output_augmentation_dir))
os.makedirs("{}/masks".format(args.output_augmentation_dir))

# copy each test image and its mask to the test directory
for test_file in test_files:
    shutil.copy("{}/images/{}".format(args.labelbox_output_dir, test_file), "{}/images/{}".format(args.output_test_dir, test_file))
    shutil.copy("{}/masks/{}".format(args.labelbox_output_dir, test_file), "{}/masks/{}".format(args.output_test_dir, test_file))
