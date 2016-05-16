# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:50:47 2015

@author: teichman
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import logging
import os
import sys
# import random
from random import shuffle

import ipdb

import numpy as np

import scipy as scp
import scipy.misc

import tensorflow as tf
from tensorflow.python.ops import math_ops


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def _load_gt_file(hypes, data_file=None):
    """Take the data_file and hypes and create a generator
    that outputs the image and the gt_image. """

    base_path = os.path.realpath(os.path.dirname(data_file))
    files = [line.rstrip() for line in open(data_file)]

    for epoche in itertools.count():
        shuffle(files)
        for file in files:
            image_file, gt_image_file = file.split(" ")
            image_file = os.path.join(base_path, image_file)
            assert os.path.exists(image_file), \
                "File does not exist: %s" % image_file
            gt_image_file = os.path.join(base_path, gt_image_file)
            assert os.path.exists(gt_image_file), \
                "File does not exist: %s" % gt_image_file
            image = scipy.misc.imread(image_file)
            # Please update Scipy, if mode='RGB' is not avaible
            gt_image = scp.misc.imread(gt_image_file, mode='RGB')

            yield image, gt_image


def _make_data_gen(hypes, phase, data_dir):
    """Returns a data generator that outputs image samples."""

    """Returns a data generator that outputs image samples."""

    if phase == 'train':
        data_file = hypes['data']["train_file"]
    elif phase == 'val':
        data_file = hypes['data']["val_file"]
    else:
        assert False, "Unknown Phase %s" % phase

    data_file = os.path.join(data_dir, data_file)

    road_color = np.array(hypes['data']['road_color'])
    background_color = np.array(hypes['data']['background_color'])

    data = _load_gt_file(hypes, data_file)

    for image, gt_image in data:

        gt_bg = np.all(gt_image == background_color, axis=2)
        gt_road = np.all(gt_image == road_color, axis=2)

        assert(gt_road.shape == gt_bg.shape)
        shape = gt_bg.shape
        gt_bg = gt_bg.reshape(shape[0], shape[1], 1)
        gt_road = gt_road.reshape(shape[0], shape[1], 1)

        gt_image = np.concatenate((gt_bg, gt_road), axis=2)

        yield image, gt_image

        if phase == 'train':

            yield np.fliplr(image), np.fliplr(gt_image)

            yield np.flipud(image), np.flipud(gt_image)

            yield np.flipud(np.fliplr(image)), np.flipud(np.fliplr(gt_image))




def create_queues(hypes, phase):
    arch = hypes['arch']
    dtypes = [tf.float32, tf.int32]
    shapes = (
        [arch['image_height'], arch['image_width'], arch['num_channels']],
        [arch['image_height'], arch['image_width'], arch['num_classes']],)
    capacity = 100
    q = tf.FIFOQueue(capacity=100, dtypes=dtypes, shapes=shapes)
    tf.scalar_summary("queue/%s/fraction_of_%d_full" %
                      (q.name + phase, capacity),
                      math_ops.cast(q.size(), tf.float32) * (1. / capacity))

    return q


def start_enqueuing_threads(hypes, q, sess, data_dir):

    shape = [hypes['arch']['image_height'], hypes['arch']['image_width'],
             hypes['arch']['num_channels']]
    image_pl = tf.placeholder(tf.float32,
                              shape=shape)

    # Labels
    shape = [hypes['arch']['image_height'], hypes['arch']['image_width'],
             hypes['arch']['num_classes']]
    label_pl = tf.placeholder(tf.int32,
                              shape=shape)

    def make_feed(data):
        image, label = data
        return {image_pl: image, label_pl: label}

    def enqueue_loop(sess, enqueue_op, phase, gen):
        # infinity loop enqueueing data
        for d in gen:
            sess.run(enqueue_op[phase], feed_dict=make_feed(d))

    threads = []
    enqueue_op = {}
    for phase in ['train', 'val']:
        # enqueue once manually to avoid thread start delay
        enqueue_op[phase] = q[phase].enqueue((image_pl, label_pl))
        gen = _make_data_gen(hypes, phase, data_dir)
        data = gen.next()
        sess.run(enqueue_op[phase], feed_dict=make_feed(data))
        num_threads = 4
        for i in range(num_threads):
            threads.append(tf.train.threading.Thread(target=enqueue_loop,
                                                     args=(sess, enqueue_op,
                                                           phase, gen)))
        threads[-1].start()


def _read_processed_image(q, phase):
    image, label = q[phase].dequeue()
    if phase == 'train':

        # Because these operations are not commutative, consider randomizing
        # randomize the order their operation.
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

    image = tf.image.per_image_whitening(image)

    return image, label


def inputs(hypes, q, phase, data_dir):
    num_threads = 4
    example_list = [_read_processed_image(q, phase)
                    for i in range(num_threads)]

    batch_size = hypes['solver']['batch_size']
    minad = 32
    capacity = minad + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch_join(
        example_list,
        batch_size=batch_size,
        min_after_dequeue=minad,
        capacity=capacity)

    # Display the training images in the visualizer.
    tensor_name = image_batch.op.name
    tf.image_summary(tensor_name + 'images', image_batch)

    return image_batch, label_batch


def main():

    with open('../hypes/kitti_seg.json', 'r') as f:
        hypes = json.load(f)

    q = {}
    q['train'] = create_queues(hypes, 'train')
    q['val'] = create_queues(hypes, 'val')
    data_dir = "../DATA"

    _make_data_gen(hypes, 'train', data_dir)

    image_batch, label_batch = inputs(hypes, q, 'train', data_dir)

    logging.info("Start running")

    with tf.Session() as sess:
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        coord = tf.train.Coordinator()
        start_enqueuing_threads(hypes, q, sess, data_dir)

        logging.info("Start running")
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in itertools.count():
            image = image_batch.eval()
            gt = label_batch.eval()
            scp.misc.imshow(image[0])
            gt_bg = gt[0, :, :, 0]
            gt_road = gt[0, :, :, 1]
            scp.misc.imshow(gt_bg)
            scp.misc.imshow(gt_road)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
