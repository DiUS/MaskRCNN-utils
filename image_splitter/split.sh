#!/bin/bash

mkdir -p output/{images,masks}

for images_or_masks in masks images; do
  for file_full_path in ../labelbox_parser/output/${images_or_masks}/*.png; do
    filename=$(basename $file_full_path)
    convert -crop 1300x1300 $file_full_path "output/${images_or_masks}/${filename}_cropped_%d.png"
  done
done
