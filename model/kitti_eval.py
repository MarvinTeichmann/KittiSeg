#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
import random
from seg_utils import seg_utils as seg

import tensorflow as tf


def eval_image(hypes, gt_image, cnn_image):
    """."""
    thresh = np.array(range(0, 256))/255.0
    road_gt = gt_image[:, :, 2] > 0
    valid_gt = gt_image[:, :, 0] > 0

    FN, FP, posNum, negNum = seg.evalExp(road_gt, cnn_image,
                                         thresh, validMap=None,
                                         validArea=valid_gt)

    return FN, FP, posNum, negNum


def evaluate(hypes, sess, image_pl, softmax):
    data_dir = hypes['dirs']['data_dir']
    data_file = hypes['data']['val_file']
    data_file = os.path.join(data_dir, data_file)
    image_dir = os.path.dirname(data_file)

    thresh = np.array(range(0, 256))/255.0
    total_fp = np.zeros(thresh.shape)
    total_fn = np.zeros(thresh.shape)
    total_posnum = 0
    total_negnum = 0

    image_list = []

    with open(data_file) as file:
        for i, datum in enumerate(file):
                datum = datum.rstrip()
                image_file, gt_file = datum.split(" ")
                image_file = os.path.join(image_dir, image_file)
                gt_file = os.path.join(image_dir, gt_file)

                image = scp.misc.imread(image_file)

                if hypes['jitter']['reseize_input']:
                    image_height = hypes['jitter']['image_height']
                    image_width = hypes['jitter']['image_width']
                    input_image = scp.misc.imresize(
                        image, size=(image_height, image_width),
                        interp='bilinear')
                else:
                    input_image = image

                shape = input_image.shape

                gt_image = scp.misc.imread(gt_file)
                feed_dict = {image_pl: input_image}

                output = sess.run([softmax], feed_dict=feed_dict)
                output_im = output[0][:, 1].reshape(shape[0], shape[1])

                if hypes['jitter']['reseize_input']:
                    gt_shape = gt_image.shape
                    output_im = scp.misc.imresize(output_im,
                                                  size=(gt_shape[0],
                                                        gt_shape[1]),
                                                  interp='bilinear')

                if i % 5 == 0:
                    ov_image = seg.make_overlay(image, output_im)
                    name = os.path.basename(image_file)
                    image_list.append((name, ov_image))

                FN, FP, posNum, negNum = eval_image(hypes, gt_image, output_im)

                total_fp += FP
                total_fn += FN
                total_posnum += posNum
                total_negnum += negNum

    eval_dict = seg.pxEval_maximizeFMeasure(total_posnum, total_negnum,
                                            total_fn, total_fp,
                                            thresh=thresh)

    eval_list = []

    eval_list.append(('MaxF1', 100*eval_dict['MaxF']))
    eval_list.append(('BestThresh', 100*eval_dict['BestThresh']))
    eval_list.append(('Average Precision', 100*eval_dict['AvgPrec']))

    return eval_list, image_list
