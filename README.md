# mobilenet-yolo-syg model

[![license1](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE)](LICENSE)
[![license2](https://raw.githubusercontent.com/david8862/keras-YOLOv3-model-set/master/LICENSE)](LICENSE)

## Introduction
A YOLOv4-MobileNet object detection pipeline inherited from [keras-YOLOv3-model-set](https://github.com/david8862/keras-YOLOv3-model-set) and [keras-yolo3-Mobilenet](https://github.com/Adamdad/keras-YOLOv3-mobilenet). 
Implement with keras, including model training/tuning, model evaluation and on device deployment. The model supports [dataset called SYGData0829](https://github.com/ermubuzhiming/OMZ-files-download/releases/download/v1-ly) collected by our team and including 4 classes which are 'motor,bike,truck,bus'.

## Setup
### Prerequisites
* Python 3.6.5
* Tensorflow 1.15.0
* Keras 2.2.5

## Training your own model
### Dataset preparation
To download SYGData0829 dataset, you need to follow the steps below:
1. Go to the [github repo](https://github.com/ermubuzhiming/OMZ-files-download/releases/tag/v1-ly)
2. Select [`SYGData0829.zip.001`](https://github.com/ermubuzhiming/OMZ-files-download/releases/download/v1-ly/SYGData0829.zip.001) 、
[`SYGData0829.zip.002`](https://github.com/ermubuzhiming/OMZ-files-download/releases/download/v1-ly/SYGData0829.zip.002) 、
[`SYGData0829.zip.003`](https://github.com/ermubuzhiming/OMZ-files-download/releases/download/v1-ly/SYGData0829.zip.003) 、
[`SYGData0829.zip.004`](https://github.com/ermubuzhiming/OMZ-files-download/releases/download/v1-ly/SYGData0829.zip.004) and download archive
3. Unpack archive
```
unzip -zvf SYGData0829.zip -d <path_same_with_train.py>
cd <path_same_with_train.py>
python voc_annotation.py  
```
You will get 3 files which the model will load the related dataset according to.
### Start to train
You can directly run the file with the default parameters.
```
python train.py
```
or you can modify [train.py](https://github.com/ermubuzhiming/OMZ-model-download/blob/main/train.py) to set 
`annotation_path`,`classes_path`,`anchors_path`,`weights_path`,`log_dir`, `Init_epoch`,`Freeze_epoch`,`batch_size`,`learning_rate_base`,
`Freeze_epoch`,`Epoch` and other train parameters, then run the files.

## Inference
If you want to test image,modify `FLAG` as `True` in the file [yolo_image.py](https://github.com/ermubuzhiming/OMZ-model-download/blob/main/yolo_image.py),
make dictionary named img and put test images, then
```
python yolo_image.py
```
If you want to test video,make dictionary named video and put test videos, modify `video_path` in the file 
[yolo_image.py](https://github.com/ermubuzhiming/OMZ-model-download/blob/main/yolo_image.py), then
```
python yolo_image.py
```
## Evalution
### Preparation
* modify `all_path` and `save_path` in the file [get_gt_txt1.py](https://github.com/ermubuzhiming/OMZ-model-download/get_gt_txt1.py) 
to get the ground-truth in the test dataset.
* modify `all_path` and `save_path` in the file [get_dr_txt2.py](https://github.com/ermubuzhiming/OMZ-model-download/get_dr_txt2.py)
to get the detection-result and images-optional in the test dataset.
* modify `MINOVERLAP` in the file [get_map3.py](https://github.com/ermubuzhiming/OMZ-model-download/get_map3.py)
to calculate mAP.
### Start to evaluate
```
python get_gt_txt1.py
python get_dr_txt2.py
python get_map3.py
```
### evalution metric and result
| Metric            | Value                |
|-------------------|----------------------|
| AP@motor          |  94.5%               |
| AP@truck          |  57%                 |
| AP@bus            |  86%                 |
| AP@car            |  77%                 | 
| mAP               |  67.84%              | 
