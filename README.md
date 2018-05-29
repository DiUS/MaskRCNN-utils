Utilities for working with Mask R-CNN, a neural network for object instance segmentation.


Steps to train a model to segment vein instances using Mask R-CNN:
* Use [Labelbox](labelbox.io) to label rock veins on the image
* Export the Labelbox project as JSON.
* Generate single instance mask using the coordinates in the json using [MaskRCNN-utils/labelbox_parser](labelbox_parser/Labelbox%20JSON%20to%20Instance%20Mask.ipynb) notebook. It generates the results in `MaskRCNN-utils/labelbox_parser/output/`. Copy both images and masks from that output directory into `MaskRCNN-utils/augmentor/input/` directory.
* Apply augmentation to generate more samples using [MaskRCNN-utils/augmentor](augmentor/augmentor.ipynb) notebook. It generates the augmentations in `MaskRCNN-utils/augmentor/output/stage1_train` direcetory. Copy the directory into `MaskRCNN-utils/Mask_RCNN/dataset/vein/`.
* Run [MaskRCNN-utils/Mask_RCNN/vein.sh](Mask_RCNN/vein.sh) to train the model
