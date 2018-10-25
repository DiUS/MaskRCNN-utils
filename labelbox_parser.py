#
# python -u labelbox_parser.py -ljf path/to/labelbox.json -od path/to/output/ -lcn "Sulphide/Partial Sulphide" -lcn "Pure Quartz Carbonate"
#

import json
import imageio
import numpy as np
import os
import re
from skimage import draw, io
import shutil
import argparse

parser = argparse.ArgumentParser(description='Parse Labelbox json export to generate images and their masks.')
parser.add_argument('-ljf','--labelbox_json_file', type=str, help='Path to Labelbox JSON file.', required=True)
parser.add_argument('-lcn','--labelbox_class_names', action='append', help='Labelbox class names', required=True)
parser.add_argument('-lid','--local_image_dir', type=str, help='Base directory of local images.', required=False)
parser.add_argument('-od','--output_dir', type=str, help='Base directory of the output.', required=True)
parser.add_argument('--resize_images', dest='resize_images', action='store_true', help='Resize images.')
parser.set_defaults(resize_images=False)
args = parser.parse_args()

def resize_image_to_height(image, height=400):
    image_dtype = image.dtype

    h, w = image.shape[:2]

    top_pad = (height - h) // 2
    bottom_pad = height - h - top_pad
    left_pad = 0
    right_pad = 0
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)

    return image.astype(image_dtype), padding

def resize_mask(mask, padding):
    return np.pad(mask, padding, mode='constant', constant_values=0)

def normalise_label_name(label_name):
    label_name = re.sub('/', '-', label_name)
    label_name = re.sub(' ', '_', label_name)
    return label_name

labelbox_json = json.load(open(args.labelbox_json_file))

output_images_dir = os.path.join(args.output_dir, 'images')
output_masks_dir = os.path.join(args.output_dir, 'masks')

# remove any previous labelbox parsed data
if os.path.exists(output_images_dir):
    shutil.rmtree(output_images_dir)

if os.path.exists(output_masks_dir):
    shutil.rmtree(output_masks_dir)

os.makedirs(output_images_dir)
os.makedirs(output_masks_dir)

for item in labelbox_json:
    labels = item['Label']
    if labels == 'Skip':
        continue

    image_name = item['External ID']
    image_url = item['Labeled Data']
    if args.local_image_dir:
        image_ref = os.path.join(args.local_image_dir, image_name)
    else:
        image_ref = image_url
    print("Image ref: {}".format(image_ref))

    image = io.imread(image_ref)
    height, width = image.shape[:2]

    if args.resize_images:
        image, padding = resize_image_to_height(image)

    for label_name, label_regions in labels.items():
        if label_name not in args.labelbox_class_names:
            continue

        for index, label_region in enumerate(label_regions):
            assert 'geometry' in label_region
            label_polygon = label_region['geometry']
            mask = np.zeros((height, width), dtype=np.uint8)
            all_points_x = []
            all_points_y = []
            for coords in label_polygon:
                all_points_x.append(coords['x'])
                all_points_y.append(coords['y'])

            rr, cc = draw.polygon(all_points_y, all_points_x)

            try:
                mask[rr, cc] = 1
            except IndexError:
                print('IndexError image ({})'.format(image_ref))
                continue

            mask = np.flipud(mask)

            if args.resize_images:
                mask = resize_mask(mask, padding[:2])

            new_filename = '{}_inst_{}_class_{}.png'.format(
                os.path.splitext(image_name)[0],
                index,
                normalise_label_name(label_name)
            )

            image_output_path = os.path.join(output_images_dir, new_filename)
            imageio.imwrite(image_output_path, image)

            instance_mask_output_path = os.path.join(output_masks_dir, new_filename)
            imageio.imwrite(instance_mask_output_path, mask)
#     break

