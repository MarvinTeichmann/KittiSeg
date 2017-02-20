# FAQ

### How can I use KittiSeg on my own data?

Have a look at [inputs.md](inputs.md) and this [issue](https://github.com/MarvinTeichmann/KittiSeg/issues/8). Feel free to open a further issue or comment on [issue 8](https://github.com/MarvinTeichmann/KittiSeg/issues/8) if your question is not covered so far. 

Also, once you figured out how to make it work, feel free to add some lines of explanation to [inputs.md](inputs.md). As owner of this code, it is not easy get an idea of which conceptual aspects needs more explanation. 

### I would like to train on greyscale images, is this possible?

Yes, since commit f7fdb24, all images will be converted to RGB upon load. Greyscale is those supported out-of-the box. This means, that even for greyscale data each pixel will be represented by a tripel. This is important when specifying the input format in [your hype file](../hypes/KittiSeg.json). Black will be stored as [0,0,0], white [255,255,255] and some light grey can be [200, 200, 200].

### Can I use KittiSeg for multi-class segmentation?

Yes, I had an earlier version run on Cityscapes data. Unfortunatly, my Cityscapes code is not compatible with the current TensorFlow and TensorVision version anymore and I did not find the time to port it, yet.

However making this run is not to much of an afford. You will need to adapt `_make_data_gen` in the [input_producer](../inputs/kitti_seg_input.py) to produce an `gt_image` tensor with more then two channels. In addition, you will need to write new evaluation code. The current [evaluator file](../evals/kitti_evals.py) computes kitti scores which are only defined on binary segmentation problems. 

Feel free to open a pull request if you find the time to implement those changes. I am also happy to help with any issues you might encounter.

### How can I make a model trained on Kitti data perform better on non-kitti street images? ([Issue #14](https://github.com/MarvinTeichmann/KittiSeg/issues/14))

Turn data augmentation on. The current version has all data augmentation turned of on default to perform well on the benchmark. This makes the trained model very sensitive to various aspects including lighting conditions and sharpness. Distortions, like random brightness, random resizing (including the change of aspect ratio) and even fancier thinks will force the ignore camera depended hints. Many common distortions are already in the [input-producer](https://github.com/MarvinTeichmann/KittiSeg/blob/master/inputs/kitti_seg_input.py), but turned of on default. 

Alternative, consider training on your data (if possible) or apply fine-tuning using view labeled images of your data.






