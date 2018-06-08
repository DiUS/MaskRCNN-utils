Utilities for working with Mask R-CNN, a neural network for object instance segmentation.

The data preparation pipeline to train a model to segment vein instances using Mask R-CNN is as follows:
  ```
  Labelbox -> ImageSplitter -> ImageDistributer (test vs rest) -> Augmentor -> ImageDistributer (train vs val)-> MaskRCNN
  ```

The steps are explained as follows:
* Use [Labelbox](https://labelbox.com) to label rock veins on the image.
* Export the Labelbox project as JSON.
* Generate single instance mask using the coordinates in the json using [MaskRCNN-utils/labelbox_parser](labelbox_parser/Labelbox%20JSON%20to%20Instance%20Mask.ipynb) notebook. It generates the results in `MaskRCNN-utils/labelbox_parser/output/`.
* Split large images and masks into 1300x1300 squares by using [MaskRCNN-utils/image_splitter](image_splitter/split.sh). It generates the results in `MaskRCNN-utils/image_splitter/output/`.
* Separate 5% of images+masks from `MaskRCNN-utils/image_splitter/output` to be used for testing the trained Mask R-CNN model later and remaining 95% for augmentation for training and validation of new model using [MaskRCNN-utils/images_distributer](MaskRCNN-utils/images_distributer/separate_test_and_augmentation_images.ipynb) notebook. It generates test dataset in `MaskRCNN-utils/images_distributer/output/stage1_test` and the rest in `MaskRCNN-utils/augmentor/input` for augmentation.
* Apply augmentation to generate more samples using [MaskRCNN-utils/augmentor](augmentor/augmentor.ipynb) notebook. It generates the augmentations in `MaskRCNN-utils/augmentor/output` directory.
* Separate 5% of augmented images+masks from `MaskRCNN-utils/augmentor/output` to be used as validation dataset by using [MaskRCNN-utils/images_distributer](MaskRCNN-utils/images_distributer/separate_train_and_val_images.ipynb) notebook. It moves 5% images to `MaskRCNN-utils/images_distributer/output/val` directory and the rest to `MaskRCNN-utils/images_distributer/output/stage1_train` directory.
* Now the `MaskRCNN-utils/images_distributer/output` contains `stage1_train`, `val` and `stage1_test` dataset directories which can used in Mask R-CNN training and inference.
