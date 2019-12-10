# Introduction

This directory contains PyTorch YOLOv3 implementation towards ECE-285 Final Project on Multi-Object Detection. 

# Description
The repo contains inference and training code for YOLOv3 in PyTorch. Training is done on the VOCPascal dataset: http://host.robots.ox.ac.uk/pascal/VOC/

# Requirements
Python 3.7 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `torch >= 1.1.0`
- `opencv-python`
- `tqdm`

## Demo
- `python detect.py --source <file.jpg> --cfg cfg/yolov3-voc.cfg --weights <>` //Download weights using the google drive links below. 

<img src= "aeroplane.jpg" width=400>    <img src= "car.jpg" width=400>

## mAP calculation/Inference
- `python test.py --data data_voc/2012_train.data --cfg cfg/yolov3-voc.cfg --img-size <> --epochs <>` 
### For anti-aliased model
- Download anti-aliased model from the google drive link in the following section.
- To test for a diagonal translation up to 10 pixels (translation amount is user adjustable).
 `python AAtest.py --data data_voc/2012_train.data --cfg cfg/yolov3-voc.cfg --weights AAyolov3.pt --batch-size 16 --img-size 320 --test_AA_shift 10 --AAmode 1`
- To test other models with diagonal translation
`python AAtest.py --data data_voc/2012_train.data --cfg cfg/yolov3-voc.cfg --weights <weights.pt> --batch-size 16 --img-size 320 --test_AA_shift 10 --AAmode 0`

## Dataset
Download VOCPascal2012/2007 dataset from http://host.robots.ox.ac.uk/pascal/VOC/. data_voc folder needs to have VOCdevkit folder in which you would need to put VOCPascal2012 and VOCPascal2007 folder containing images. To get the format which is read by the model, you would need run voc_label.py in the data_voc folder without any arguments.

## Training
After organizing the dataset folder as mentioned above, we can start to train the model using the following command. Please choose the configuration file accordingly for the model you wish to train. We have setup the model to train on following configurations:
- Default: yolov3-voc.cfg
- With Group Normalization: yolov3-voc-grpnorm.cfg
- Focal Loss: yolov3-voc-anchors_plus_focal_loss.cfg
- With deformable convolution: 	yolov3-voc_deconv.cfg

**Start Training:** 
- `python train.py --data data_voc/2012_train.data --cfg cfg/yolov3-voc.cfg --img-size <> --epochs <>`

**Resume Training:**
- `python3 train.py --resume` to resume training from `weights/last.pt`.


**Note** 
To train the model with Deformable Convolution you would need to setup the environment. This is because the DNCv2 repository requires us to run a shell file tries to write something to site-packages directory for which we do not have access. Using conda environment, it make changes in the local site-packages which we resorted to. Here, to make it easy to check other model implementations, we have separated the DCNv2 part. You can run all the models as it is without requiring any environment initialization. Please follow the below instructions to train/test deformable convolution model. Also, you can use the same environment to run CenterNet model.


~~~
conda create --name yolov3 python=3.6
~~~

~~~
conda activate yolov3
~~~

~~~
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
~~~

~~~
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
make
python setup.py install --user
~~~

~~~
Go to cloned repo in DCNv2 directory and run:
./make.sh
~~~

Now you can copy the `models_dc.py` to `models.py` and can start training the YOLOv3 with deformable convolution using the same command as show above and selecting appropriate cfg file.

## Google Drive Links to pre-trained models
`YOLOv3 weights`

Default: https://drive.google.com/open?id=1lU7vbVZewn6Up9O2NUYyYV1gQWXEMMtl

Our Anchor Estimation and Focal Loss: https://drive.google.com/open?id=1LMZeOA9onrqVLWXFYngx19tSxb4WvpAK

With Deformable Convolution: https://drive.google.com/open?id=14QVmdsUUK4SjH3859Rf41odAwVd5Czha

With Anti-aliasing: https://drive.google.com/file/d/13Ue6g9QfLJ2sbQ28ah92cSCNiJd6GkxN/view?usp=sharing

`CenterNet wieghts`

DLA-34 (With Deformable Convolution): https://drive.google.com/open?id=1lAdwRHdHxRjFGl8Zz16BSGG1r8lQK2v0

ResNet-101 (Without Deformable Convolution): https://drive.google.com/open?id=1suaTpGWqjucCDr9H2J7KtVIqxHnPCT95

ResNet-101 (With Deformable Convolution): https://drive.google.com/open?id=1d9hX7TlNUtewjwvtSwwsSMZf5urfU1Y2



## Image Augmentation

`datasets.py` applies random OpenCV-powered (https://opencv.org/) augmentation to the input images in accordance with the following specifications. Augmentation is applied **only** during training, not during inference. Bounding boxes are automatically tracked and updated with the images. 416 x 416 examples pictured below.

Augmentation | Description
--- | ---
Translation | +/- 10% (vertical and horizontal)
Rotation | +/- 5 degrees
Reflection | 50% probability (horizontal-only)
H**S**V Saturation | +/- 50%
HS**V** Intensity | +/- 50%


# Credits
This repository is forked from https://github.com/ultralytics/yolov3.

The Deformable Convolution respository is forked from https://github.com/CharlesShang/DCNv2.

Anti-aliasing models repo: https://github.com/adobe/antialiased-cnns

**Credit to Joseph Redmon for YOLO:** https://pjreddie.com/darknet/yolo/.




