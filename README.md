# TensorSeg
TensorSeg is a toolkit for building Segmentation approachs in Tensorflow.

This code was used to archive [first place](http://www.cvlibs.net/datasets/kitti/eval_road_detail.php?result=ca96b8137feb7a636f3d774c408b1243d8a6e0df) in the Kitti Road Segmentation challenge. If you use this code, please cite our paper: [MultiNet](https://arxiv.org/abs/1612.07695)

## Train on Kitti Segmentation Data

1. Clone the repository: `git clone https://github.com/MarvinTeichmann/TensorSeg`
2. Initialize all submodules: `git submodule update --init --recursive`
3. Install numpy, scipy, pillow and matplotlib 
(e.g. `pip install numpy scipy pillow matplotlib`)
4. Retrieve kitti data url here: `http://www.cvlibs.net/download.php?file=data_road.zip`
3. Download and prepared data by running: `python download_data.py --kitti_url [url]`  
4. Run `python train.py` to start training

## Tensorflow Version Note

This code works with tf 0.12. Deprecation warnings will be tackled soon.


# Configuration 

http://tensorvision.readthedocs.io/en/master/user/configuration.html