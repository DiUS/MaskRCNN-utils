# Vein Segmentation

Copied and modified from samples/nucleus/*

## Command line Usage
Train a new model starting from COCO weights using `train` dataset (which is `stage1_train` minus validation set defined in vein.py)
```
python3 vein.py train --dataset=/path/to/dataset --subset=train --weights=coco
```

Train a new model starting from specific weights file using the full `stage1_train` dataset
```
python3 vein.py train --dataset=/path/to/dataset --subset=stage1_train --weights=/path/to/weights.h5
```

Resume training a model that you had trained earlier
```
python3 vein.py train --dataset=/path/to/dataset --subset=train --weights=last
```

Detect veins on validation images defined in vein.py
```
python3 vein.py detect --dataset=/path/to/dataset --subset=stage1_test --weights=<last or /path/to/weights.h5>
```


## Jupyter notebooks
Two Jupyter notebooks are provided as well: `inspect_vein.ipynb` and `inspect_vein.ipynb`.
They explore the dataset, run stats on it, and go through the detection process step by step.
