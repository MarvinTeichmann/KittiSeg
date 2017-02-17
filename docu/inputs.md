## How to train on your own data

### Easy way

The easiest way is to provide data in a similar way to the kitti data. To do that create files `train` and `val` similar to [train3.txt](../data/train3.txt). Each line of this file is supposed to contain a path to an image and a path to the corresponding ground truth. 

The ground truth file is assumed to be an image. By default `red` is considered as `background` and `purple` as foreground. All other colours are considered as 'unknown', the loss from those pixels are ignored during training. You can configure those colours in the `hype` file by changing

```
  "data": {
    "road_color" : [255,0,255],
    "background_color" : [255,0,0]
  },
```


### Hard way

The disadvantage of the easy way is, that it only works for binary segmentation problems (i.e. two classes). The alternative is to write you own input producer and evaluation file. All other files are independent of the data. 

In (kitti_seg_input.py)[kitti_seg_input.py] the actual data is loaded in the functions *_make_data_gen* and *_load_gt_file*. If you modify those you should be able to load any kind of dataset. 

The eval file 'kitti_eval.py' is designed to utilize the original evaluation code provided by the kitti road detection benchmark. If you train on your own data with different evaluation metrics I recommend using your own evaluation code. 

