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


## Testing
~~~
python demo.py ctdet --demo /path/to/image/or/folder/or/video --load_model ../models/ctdet_coco_dla_2x.pth
~~~

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{zhou2019objects,
      title={Objects as Points},
      author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
      booktitle={arXiv preprint arXiv:1904.07850},
      year={2019}
    }
