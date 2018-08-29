Utilities for working with Mask R-CNN, a neural network for object instance segmentation.

The data preparation pipeline to train a model to segment vein instances using Mask R-CNN is as follows:
  ```
  Labelbox -> ImageSplitter -> ImageDistributer (test vs rest) -> Augmentor -> ImageDistributer (train vs val)-> MaskRCNN
  ```

The steps are explained as follows:
* Use [Labelbox](https://labelbox.com) to label classes on the image.
* Export the Labelbox project as JSON.
* Generate images their single instance masks using the coordinates in the json:
  ```
  python -u labelbox_parser.py \
  --labelbox_json_file path/to/labelbox.json \
  --labelbox_class_names "Sulphide/Partial Sulphide" --labelbox_class_names "Pure Quartz Carbonate" \
  --output_dir path/to/labelbox_parser/output \
  --resize_images
  ```
* Split the large images and masks from the output above into squares:
  ```
  python -u image_splitter.py --input_dir path/to/labebox_parser/output/ --output_dir path/to/image_splitter/output/
  ```
* Separate 5% of images and masks, splitted above, to be used for testing the trained Mask R-CNN model later and remaining 95% for augmentation for training and validation of new model using:
  ```
  python -u separate_test_and_augmentation_images.py \
  --labelbox_output_dir path/to/labelbox_parser/output/ \
  --image_splitter_output_dir path/to/image_splitter/output/with_labels_only \
  --output_test_dir path/to/mrcnn/dataset/stage1_test \
  --output_augmentation_dir path/to/output/augmentation_raw \
  --labelbox_class_names "Sulphide/Partial Sulphide" --labelbox_class_names "Pure Quartz Carbonate"
  ```
* Apply augmentation to generate more samples:
  ```
  python -u augmentation.py \
  --input_dir path/to/augmentation_raw/ \
  --output_dir path/to/augmenation/output/ \
  --number_of_augmented_images_per_original 20 \
  --no-augment_colour
  ```
* Separate 5% of images and masks, augmented above, to be used as validation dataset and the rest 95% as training dataset:
  ```
  python -u separate_train_and_val_images.py \
  --input_dir path/to/augmentation/output \
  --output_dir path/to/mrcnn/dataset/ \
  --labelbox_class_names "Sulphide/Partial Sulphide" --labelbox_class_names "Pure Quartz Carbonate"
  ```
* Now the `path/to/mrcnn/dataset/` contains `stage1_train`, `val` and `stage1_test` dataset directories which can used in Mask R-CNN training and inference.
