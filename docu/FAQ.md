# FAQ

### I have issues loading my own data.

Have a look at [inputs.md](inputs.md) and this [issue](https://github.com/MarvinTeichmann/KittiSeg/issues/8). Feel free to comment on the issue, if you have any question. Also feel free to add some lines in the docu in [inputs.md](inputs.md) and this [issue](https://github.com/MarvinTeichmann/KittiSeg/issues/8) if you think it needs more explanations. As owner of this code, it is not easy to understand what is obvious and which parts might need additional documentation.

### I would like to train on RGB data, is this possible?

Yes, since commit f7fdb24, all images will be converted to RGB upon load. Greyscale is those supported out-of-the box. This means, that even for greyscale data each pixel will be represented by a tripel. This is important when specifing the input format in [your hype file](hypes/KittiSeg.json). Black will be stored as [0,0,0], white [255,255,255] and some light grey can be [200, 200, 200].

### Can I use your code to train segmentation with more then two classes

Yes, I had an earlier version run on Cityscapes data. This code is not compatible with the current TensorFlow and TensorVision version and I did not find the time to port it, yet.

However making this run is not to much of an afford. You will need to adapt `_make_data_gen` in the [input_producer](../inputs/kitti_seg_input.py) to produce an `gt_image` tensor with more then two channels. 

In addition, you will need to write new evaluation code. The current [evaluator file](../evals/kitti_evals.py) computes kitti scores which are only defined on binary segmentation problems. 

Feel free to open a pull request if you find the time to implement those changes. I am also happy to help with any issues you might encounter.







