#!/bin/bash

rm -rf output/{images,masks}
mkdir -p output/{images,masks}

for images_or_masks in masks images; do
  for file_full_path in ../labelbox_parser/output/${images_or_masks}/*.png; do
    filename=$(basename $file_full_path .png)
    convert -crop 1300x1300 $file_full_path "output/${images_or_masks}/${filename}_seg_%d.png"
  done
done
