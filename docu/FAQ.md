# FAQ

### How can I use KittiSeg on my own data?

Have a look at [inputs.md](inputs.md) and this [issue](https://github.com/MarvinTeichmann/KittiSeg/issues/8). Feel free to open a further issue or comment on [issue 8](https://github.com/MarvinTeichmann/KittiSeg/issues/8) if your question is not covered so far. 

Also, once you figured out how to make it work, feel free to add some lines of explanation to [inputs.md](inputs.md). As owner of this code, it is not easy get an idea of which conceptual aspects needs more explanation. 

### I would like to train on greyscale images, is this possible?

Yes, since commit f7fdb24, all images will be converted to RGB upon load. Greyscale is those supported out-of-the box. This means, that even for greyscale data each pixel will be represented by a tripel. This is important when specifying the input format in [your hype file](../hypes/KittiSeg.json). Black will be stored as [0,0,0], white [255,255,255] and some light grey can be [200, 200, 200].

### Can I use your code to train segmentation with more then two classes

Yes, I had an earlier version run on Cityscapes data. This code is not compatible with the current TensorFlow and TensorVision version and I did not find the time to port it, yet.

However making this run is not to much of an afford. You will need to adapt `_make_data_gen` in the [input_producer](../inputs/kitti_seg_input.py) to produce an `gt_image` tensor with more then two channels. 

In addition, you will need to write new evaluation code. The current [evaluator file](../evals/kitti_evals.py) computes kitti scores which are only defined on binary segmentation problems. 

Feel free to open a pull request if you find the time to implement those changes. I am also happy to help with any issues you might encounter.







