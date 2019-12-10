Multi-Object detection using CenterNet model from the paper Object as Points towards ECE-285 final project in Fall 2019.

## Model Setup
It is advised to create a virutal conda environment to make the setup process simple. Please follow below steps:

~~~
conda create --name CenterNet python=3.6
~~~

~~~
conda activate CenterNet
~~~

1. Install PyTorch.
~~~
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
~~~

2. Install COCOAPI
~~~
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
make
python setup.py install --user
~~~

3. Clone this repo
~~~
CenterNet_ROOT=/path/to/clone/CenterNet
git clone https://github.com/xingyizhou/CenterNet $CenterNet_ROOT
~~~

4. Install the requirements:
~~~
pip install -r requirements.txt
~~~

5. Compile the deformable convolution module:
~~~
cd $CenterNet_ROOT/src/lib/models/networks/DCNv2
./make.sh
~~~

6. Compile NMS
~~~
cd $CenterNet_ROOT/src/lib/external
make
~~~

## Datset Setup
 Run

    cd $CenterNet_ROOT/tools/
    bash get_pascal_voc.sh

- The above script includes:
    - Download, unzip, and move Pascal VOC images from the [VOC website](http://host.robots.ox.ac.uk/pascal/VOC/). 
    - [Download](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip) Pascal VOC annotation in COCO format (from [Detectron](https://github.com/facebookresearch/Detectron/tree/master/detectron/datasets/data)). 
    - Combine train/val 2007/2012 annotation files into a single json.


- Move the created `voc` folder to `data` (or create symlinks) to make the data folder like:

  ~~~
  ${CenterNet_ROOT}
  |-- data
  `-- |-- voc
      `-- |-- annotations
          |   |-- pascal_trainval0712.json
          |   |-- pascal_test2017.json
          |-- images
          |   |-- 000001.jpg
          |   ......
          `-- VOCdevkit
  
  ~~~
  The `VOCdevkit` folder is needed to run the evaluation script from [faster rcnn](https://github.com/rbgirshick/py-faster-rcnn/blob/master/tools/reval.py).

## Testing

~~~
python test.py ctdet --exp_id <> --arch <> --dataset pascal --input_res <>  <weights>
# flip test
python test.py ctdet --exp_id <> --arch <> --dataset pascal --input_res <> --load_model <weights> --flip_test 
~~~
--arch could be: hourglass, resdcn_101, resdcn_18

## Training 

~~~
python main.py ctdet --exp_id <folder_id> --batch_size <> --arch <> --dataset pascal --num_epochs 100 --lr_step 45,60
~~~


## CenterNet wieghts

DLA-34 (With Deformable Convolution): https://drive.google.com/open?id=1lAdwRHdHxRjFGl8Zz16BSGG1r8lQK2v0

ResNet-101 (Without Deformable Convolution): https://drive.google.com/open?id=1suaTpGWqjucCDr9H2J7KtVIqxHnPCT95

ResNet-101 (With Deformable Convolution): https://drive.google.com/open?id=1d9hX7TlNUtewjwvtSwwsSMZf5urfU1Y2

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{zhou2019objects,
      title={Objects as Points},
      author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
      booktitle={arXiv preprint arXiv:1904.07850},
      year={2019}
    }
