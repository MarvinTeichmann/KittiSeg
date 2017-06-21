"""
Utilize vgg_fcn8 as encoder.
------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_fcn import fcn8_vgg

import tensorflow as tf

import os


def inference(hypes, images, train=True):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    vgg16_npy_path = os.path.join(hypes['dirs']['data_dir'], 'weights',
                                  "vgg16.npy")
    vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path=vgg16_npy_path)

    vgg_fcn.wd = hypes['wd']

    vgg_fcn.build(images, train=train, num_classes=2, random_init_fc8=True)

    logits = {}

    logits['images'] = images

    if hypes['arch']['fcn_in'] == 'pool5':
        logits['fcn_in'] = vgg_fcn.pool5
    elif hypes['arch']['fcn_in'] == 'fc7':
        logits['fcn_in'] = vgg_fcn.fc7
    else:
        raise NotImplementedError

    logits['feed2'] = vgg_fcn.pool4
    logits['feed4'] = vgg_fcn.pool3

    logits['fcn_logits'] = vgg_fcn.upscore32

    logits['deep_feat'] = vgg_fcn.pool5
    logits['early_feat'] = vgg_fcn.conv4_3

    return logits
