#!/bin/bash

echo "Splitting images and masks"
rm -rf output/all/{images,masks}
mkdir -p output/all/{images,masks}

for images_or_masks in masks images; do
  for file_full_path in ../labelbox_parser/output/${images_or_masks}/*.png; do
    filename=$(basename $file_full_path .png)
    convert -crop 1300x1300 $file_full_path "output/all/${images_or_masks}/${filename}_seg_%d.png"
  done
done

echo "Separating images and masks with labels only"
python separate_out_images_with_labels.py -id output/all -od output/with_labels_only

