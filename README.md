# KittiSeg

KittiSeg trains an FCN based model for Segmentation of Roads on the Kitti road detection dataset. The model achieved [first place](http://www.cvlibs.net/datasets/kitti/eval_road_detail.php?result=ca96b8137feb7a636f3d774c408b1243d8a6e0df) on the Benchmark at submission time and is descripted in our paper: [MultiNet](https://arxiv.org/abs/1612.07695).

The model is optimized to perform well even with very few data samples avaible. The implementation is trained on just *250* images and receives a state-of-the art MaxF1 score of over *96%*. It's inference time is less then *95ms*.



![](https://i.imgur.com/Usi1XhE.png)

The code contains for `train`, `evaluate` and `visualize` semantic segmentation in tensorflow. It is build to be compatible with the [TensorVision](http://tensorvision.readthedocs.io/en/master/user/tutorial.html#workflow) backend which allows to organize experiments in a very clean way.

## Requirements

The code requires Tensorflow 1.0 as well as the following python libraries: 

* matplotlib
* numpy
* Pillow
* scipy

Those modules can be installed using: `pip install numpy scipy pillow matplotlib` or `pip install -r requirements.txt`.

## Setup

1. Clone this repository: `git clone git@github.com:MarvinTeichmann/KittiSeg.git`
2. Initialize all submodules: `git submodule update --init --recursive`
3. Retrieve kitti data url here: [http://www.cvlibs.net/download.php?file=data_road.zip](http://www.cvlibs.net/download.php?file=data_road.zip)
4. Call `python download_data.py --kitti_url URL_YOU_RETRIEVED`

I strongly recommand to exececute step 4. instead of downloading the data yourself. The script will also prepare the data to be in the right input format and check whether the right data is downloaded.




## Tutorial

### Getting started

Run: `python evaluate.py` to evaluate a trained model. 

Run: `python train.py` to train a new model on the Kitti Data.

### Modifying Model & Train on your own data

The model is controlled by the file `hypes/KittiSeg.json`. Modifying this file should be enough to train the model on your own data and adjust the architecture according to your needs. You can create a new file `hypes/my_hype.json` and train that architecture using:

`python train.py --hypes hypes/my_hype.json`



For advanced modifications, the code is controlled by 5 different modules, which are specified in `hypes/KittiSeg.json`.

```
"model": {
   "input_file": "../inputs/kitti_seg_input.py",
   "architecture_file" : "../encoder/fcn8_vgg.py",
   "objective_file" : "../decoder/kitti_multiloss.py",
   "optimizer_file" : "../optimizer/generic_optimizer.py",
   "evaluator_file" : "../evals/kitti_eval.py"
},
```

Those modules operate independently. This allows easy experiments with different datasets (`input_file`), encoder networks (`architecture_file`), etc. Also see [TensorVision](http://tensorvision.readthedocs.io/en/master/user/tutorial.html#workflow) for a specification of each of those files.


## Managing Folders

By default, the data is stored in the folder `KittiSeg/DATA` and the output of runs in `KittiSeg/RUNS`. This behaviour can be changed by adjusting the environoment Variabels: `$TV_DIR_DATA` and `$TV_DIR_RUNS`.

For organizing your experiments you can use:
`python train.py --project batch_size_bench --name size_5`. This will store the run in the subfolder:  `$TV_DIR_RUNS/batch_size_bench/size_5_%DATE`

This is useful if you want to run different series of experiments.


## Utilize TensorVision backend

KittiSeg is build on top of the TensorVision [TensorVision](https://github.com/TensorVision/TensorVision) backend. TensorVision modulizes computer vision training and helps organizing experiments. 


To utilize the entire TensorVision functionality install it using 

`$ cd KittiSeg/submodules/TensorVision` <br>
`$ python setup install`

Now you can use the TensorVision command line tools, which includes:

`tv-train --hypes hypes/KittiSeg.json` trains a json model. <br>
`tv-continue --logdir PATH/TO/RUNDIR` continues interrupted training <br>
`tv-analyze --logdir PATH/TO/RUNDIR` evaluated trained model <br>


## Useful Flags & Variabels

Here are some Flags which will be useful when working with KittiSeg and TensorVision. All flags are avaible across all scripts. 

`--hypes` : specify which hype-file to use <br>
`--logdir` : specify which logdir to use <br>
`--gpus` : specify on which GPUs to run the code <br>
`--name` : assign a name to the run <br>
`--project` : assign a project to the run <br>
`--nosave` : debug run, logdir will be set to `debug` <br>

In addition the following TensorVision environoment Variables will be useful:

`$TV_DIR_DATA`: specify meta directory for data <br>
`$TV_DIR_RUNS`: specify meta directiry for output <br>
`$TV_USE_GPUS`: specify default GPU behavour. <br>

On a cluster it is useful to set `$TV_USE_GPUS=force`. This will make the flag `--gpus` manditory and ensure, that run will be executed on the right gpu.
