# kaggle-kuzushiji-recognition

## Project structure

```
├── code
│   ├── configs
│   ├── crop_det.py
│   ├── make_submission.py
│   ├── mmdetection
│   ├── prepare_det.py
│   └── work_dirs
├── data
│   ├── dtrainval.pkl
│   ├── train_images
│   ├── test_images
│   ├── train_crops
│   └── ...
├── download
│   ├── test_images.zip
│   ├── train_images.zip
│   └── ...
└── submits
    ├── submit001.csv
    └── ...
```

## Requirements

* 2x 1080 Ti
* [mmdetection](https://github.com/open-mmlab/mmdetection/) with a few modifications (e.g. mmdetection/mmdet/datasets/pipelines/test_aug.py)
* requirements.txt


## Overview

The solution is straightfoward.

Cascade R-CNN with:

* Strong backbones
* Multi-scale train&test

Due to limited GPU memory, models were trained on 1024x1024 crops and tested on full images(with a max size limit).

LB score 0.935 with:
* HRNet w32
* train scales 512~768
* test scales [0.5, 0.625, 0.75]

LB score 0.946 with:
* HRNet w32
* train scales 768~1280
* test scales [0.75, 0.875, 1.0, 1.125, 1.25]

Ensembling HRNet_w32 and HRNet_w48 results -> 0.950.


## Run

Install mmdetection following ./mmdetection/docs/INSTALL.md

Install packages following requirements.txt

### Preprocess

```
python prepare_det.py
python crop_det.py
```

### Train
```
./mmdetection/tools/dist_train.sh configs/hr32.py 2 --seed 0
```

### Test
```
./mmdetection/tools/dist_test.sh configs/hr32.py work_dirs/hr32/epoch_12.pth 2 --out work_dirs/hr32/test_result.pkl
```

### Postprocess and make submission
```
python make_submission.py ../submits/submit001.csv --input work_dirs/hr32/test_result.pkl
# python make_submission.py ../submits/submit001.csv --input work_dirs/hr32/test_result.pkl work_dirs/hr48/test_result.pkl
```

### Inference
```
python inference.py configs/hr32.py ../data/kuzushiji-hrnetv2p_w32-f391e720.pth ../data/test_images/test_012f99f8.jpg ../data/out.jpg
```
