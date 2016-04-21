# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:50:47 2015

@author: teichman
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import ipdb
import json
import logging
import os
import sys
import random
from random import shuffle

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
            gt_image = scp.misc.imread(gt_image_file)

            yield image, gt_image


def _make_data_gen(hypes, phase, data_dir):
    """Returns a data generator that outputs image samples."""

    if phase == 'train':
        data_file = hypes['data']["train_file"]
    elif phase == 'val':
        data_file = hypes['data']["val_file"]
    else:
        assert False, "Unknown Phase %s" % phase

    data_file = os.path.join(data_dir, data_file)

    image_size = hypes['arch']['image_size']
    num_pixels = image_size * image_size
    background_color = np.array(hypes['data']['background_color'])

    data = _load_gt_file(hypes, data_file)

    for image, gt_image in data:
        shape = image.shape
        assert shape[0] - image_size > 0, \
            "Invalid image_size"
        assert shape[1] - image_size > 0, \
            "Invalid image_size"
        
        # Select Background Image
        for i in range(100):
            x = np.random.randint(shape[0] - image_size)
            y = np.random.randint(shape[1] - image_size)

            gt_patch = gt_image[x:(x + image_size), y:(y + image_size)]
            mygt = (gt_patch != background_color)
            if np.sum(mygt) == 0:
                if random.random() > 0.66:
                    image_patch = image[x:(x + image_size), y:(y + image_size)]
                    yield image_patch, 0
            elif np.sum(mygt) > 0.1 * num_pixels:
                image_patch = image[x:(x + image_size), y:(y + image_size)]
                yield image_patch, 1
                


def placeholders(hypes):
    """ Placeholders are not used in cifar10"""

    return None


def create_queues(hypes, phase):
    arch = hypes['arch']
    dtypes = [tf.float32, tf.int32]
    shapes = (
        [arch['image_size'], arch['image_size'], arch['num_channels']],
        [],)
    capacity = 100
    q = tf.FIFOQueue(capacity=100, dtypes=dtypes, shapes=shapes)
    tf.scalar_summary("queue/%s/fraction_of_%d_full" % (q.name + phase, capacity),
                      math_ops.cast(q.size(), tf.float32) * (1. / capacity))

    return q


def start_enqueuing_threads(hypes, q, sess, data_dir):
    image_size = hypes['arch']['image_size']
    num_channels = hypes['arch']['num_channels']
    image_pl = tf.placeholder(tf.float32,
                              shape=[image_size, image_size, num_channels])

    # Labels
    label_pl = tf.placeholder(tf.int32, shape=[])

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
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Because these operations are not commutative, consider randomizing
        # randomize the order their operation.
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

    image = tf.image.per_image_whitening(image)

    return image, label


def inputs(hypes, q, phase, data_dir):
    num_threads = 4
    example_list = [_read_processed_image(q, phase) for i in range(num_threads)]


    batch_size = hypes['solver']['batch_size']
    minad = 10000
    capacity = minad + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch_join(example_list,
                                                           batch_size=batch_size,
                                                           min_after_dequeue=minad,
                                                           capacity=capacity)
    return image_batch, label_batch


def main():

    with open('../hypes/medseg.json', 'r') as f:
        hypes = json.load(f)

    q = {}
    q['train'] = create_queues(hypes, 'train')
    q['val'] = create_queues(hypes, 'val')
    data_dir = "../DATA"
    image_batch, label_batch = inputs(hypes, q, 'train', data_dir)

    with tf.Session() as sess:
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        start_enqueuing_threads(hypes, q, sess, data_dir)

        for i in itertools.count():
            image = image_batch.eval()
            logging.info("Class is: %s", label_batch.eval())
            # scp.misc.imshow(image[0])

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
