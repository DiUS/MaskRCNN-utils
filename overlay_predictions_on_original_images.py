#
# python -u overlay_predictions_on_original_images.py \
#   --original_images_dir path/to/original/png/images \
#   --predictions_file path/to/predictions.json \
#   --output_dir path/to/output
#
import json
import os
import random
import colorsys
import argparse
import gc
import shutil
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

parser = argparse.ArgumentParser(description='Overlay Mask R-CNN predictions result on the orignal images')
parser.add_argument('-oid', '--original_images_dir', type=str, help='Base directory containing original images', required=True)
parser.add_argument('-pf', '--predictions_file', type=str, help='JSON file containing Mask R-CNN predictions result', required=True)
parser.add_argument('-od', '--output_dir', type=str, help='Base directory where the overlayed images will be saved to', required=True)
args = parser.parse_args()

if os.path.exists(args.output_dir):
    shutil.rmtree(args.output_dir)

os.makedirs(args.output_dir)

def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

predictions_json = json.load(open(args.predictions_file))

for prediction in predictions_json:
    image_name =  prediction["original_image_id"]

    image = imread(os.path.join(args.original_images_dir, image_name))

    colors = random_colors(len(prediction["vein_instances"]))
    height, width = image.shape[:2]

    _, ax = plt.subplots(figsize=(18, 18))
    ax.set_title('Prediction')
    ax.axis('off')

    for i, instance in enumerate(prediction["vein_instances"]):
        color = colors[i]
        for verts in instance['polygons']:
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

        ## show label
        x1, y1, x2, y2 = instance['bounding_box']
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        caption = instance['class'][0]
        if 'score' in instance.keys():
            caption += " {:.3f}".format(instance['score'])

        ax.text(x1, y1 - 10, caption, color=color, size=11, backgroundcolor="none")

    ax.imshow(image)
    plt.show()
    plt.savefig("{}/{}".format(args.output_dir, image_name))

    del image
    plt.clf()
    plt.close()
    gc.collect()

