# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:50:47 2015.

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
import random
from random import shuffle

import numpy as np

import scipy as scp
import scipy.misc

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.training import queue_runner
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import dtypes


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def _load_gt_file(hypes, data_file=None):
    """Take the data_file and hypes and create a generator.

    The generator outputs the image and the gt_image.
    """
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
    """Return a data generator that outputs image samples."""
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

        if phase == 'val':
            yield image, gt_image
        elif phase == 'train':

            yield jitter_input(hypes, image, gt_image)

            yield jitter_input(hypes, np.fliplr(image), np.fliplr(gt_image))


def jitter_input(hypes, image, gt_image):

    jitter = hypes['jitter']
    res_chance = jitter['res_chance']
    crop_chance = jitter['crop_chance']

    if jitter['random_resize'] and res_chance > random.random():
        lower_size = jitter['lower_size']
        upper_size = jitter['upper_size']
        sig = jitter['sig']
        image, gt_image = random_resize(image, gt_image, lower_size,
                                        upper_size, sig)
        image, gt_image = crop_to_size(hypes, image, gt_image)

    if jitter['random_crop'] and crop_chance > random.random():
        max_crop = jitter['max_crop']
        crop_chance = jitter['crop_chance']
        image, gt_image = random_crop(image, gt_image, max_crop)

    if jitter['fix_shape']:
        image_height = jitter['image_height']
        image_width = jitter['image_width']
        assert not jitter['reseize_image']
        image, gt_image = resize_label_image_with_pad(image, gt_image,
                                                      image_height,
                                                      image_width)
    elif jitter['reseize_image']:
        image_height = jitter['image_height']
        image_width = jitter['image_width']
        image, gt_image = resize_label_image(image, gt_image,
                                             image_height,
                                             image_width)

    assert(image.shape[:-1] == gt_image.shape[:-1])
    return image, gt_image


def random_crop(image, gt_image, max_crop):
    offset_x = random.randint(1, max_crop)
    offset_y = random.randint(1, max_crop)

    if random.random() > 0.5:
        image = image[offset_x:, offset_y:, :]
        gt_image = gt_image[offset_x:, offset_y:, :]
    else:
        image = image[:-offset_x, :-offset_y, :]
        gt_image = gt_image[:-offset_x, :-offset_y, :]

    return image, gt_image


def resize_label_image_with_pad(image, label, image_height, image_width):
    shape = image.shape
    assert(image_height >= shape[0])
    assert(image_width >= shape[1])

    pad_height = image_height - shape[0]
    pad_width = image_width - shape[1]
    offset_x = random.randint(0, pad_height)
    offset_y = random.randint(0, pad_width)

    new_image = np.zeros([image_height, image_width, 3])
    new_image[offset_x:offset_x+shape[0], offset_y:offset_y+shape[1]] = image

    new_label = np.zeros([image_height, image_width, 2])
    new_label[offset_x:offset_x+shape[0], offset_y:offset_y+shape[1]] = label

    return new_image, new_label


def resize_label_image(image, gt_image, image_height, image_width):
    image = scipy.misc.imresize(image, size=(image_height, image_width),
                                interp='bilinear')
    shape = gt_image.shape
    gt_zero = np.zeros([shape[0], shape[1], 1])
    gt_image = np.concatenate((gt_image, gt_zero), axis=2)
    gt_image = scipy.misc.imresize(gt_image, size=(image_height, image_width),
                                   interp='nearest')
    gt_image = gt_image[:, :, 0:2]/255

    return image, gt_image


def random_resize(image, gt_image, lower_size, upper_size, sig):
    factor = random.normalvariate(1, sig)
    if factor < lower_size:
        factor = lower_size
    if factor > upper_size:
        factor = upper_size
    image = scipy.misc.imresize(image, factor)
    shape = gt_image.shape
    gt_zero = np.zeros([shape[0], shape[1], 1])
    gt_image = np.concatenate((gt_image, gt_zero), axis=2)
    gt_image = scipy.misc.imresize(gt_image, factor, interp='nearest')
    gt_image = gt_image[:, :, 0:2]/255
    return image, gt_image


def crop_to_size(hypes, image, gt_image):
    new_width = image.shape[1]
    new_height = image.shape[0]
    width = hypes['arch']['image_width']
    height = hypes['arch']['image_height']
    if new_width > width:
        max_x = max(new_height-height, 0)
        max_y = new_width-width
        offset_x = random.randint(0, max_x)
        offset_y = random.randint(0, max_y)
        image = image[offset_x:offset_x+height, offset_y:offset_y+width]
        gt_image = gt_image[offset_x:offset_x+height, offset_y:offset_y+width]

    return image, gt_image


def create_queues(hypes, phase):
    """Create Queues."""
    arch = hypes['arch']
    dtypes = [tf.float32, tf.int32]

    if hypes['jitter']['fix_shape']:
        height = hypes['jitter']['image_height']
        width = hypes['jitter']['image_width']
        channel = hypes['arch']['num_channels']
        num_classes = hypes['arch']['num_classes']
        shapes = [[height, width, channel],
                  [height, width, num_classes]]
    else:
        shapes = None

    capacity = 50
    q = tf.FIFOQueue(capacity=50, dtypes=dtypes, shapes=shapes)
    tf.scalar_summary("queue/%s/fraction_of_%d_full" %
                      (q.name + "_" + phase, capacity),
                      math_ops.cast(q.size(), tf.float32) * (1. / capacity))

    return q


def start_enqueuing_threads(hypes, q, phase, sess):
    """Start enqueuing threads."""
    image_pl = tf.placeholder(tf.float32)
    label_pl = tf.placeholder(tf.int32)
    data_dir = hypes['dirs']['data_dir']

    def make_feed(data):
        image, label = data
        return {image_pl: image, label_pl: label}

    def enqueue_loop(sess, enqueue_op, phase, gen):
        # infinity loop enqueueing data
        for d in gen:
            sess.run(enqueue_op, feed_dict=make_feed(d))

    enqueue_op = q.enqueue((image_pl, label_pl))
    gen = _make_data_gen(hypes, phase, data_dir)
    gen.next()
    # sess.run(enqueue_op, feed_dict=make_feed(data))
    if phase == 'val':
        num_threads = 1
    else:
        num_threads = 1
    for i in range(num_threads):
        t = tf.train.threading.Thread(target=enqueue_loop,
                                      args=(sess, enqueue_op,
                                            phase, gen))
        t.daemon = True
        t.start()


def _read_processed_image(hypes, q, phase):
    image, label = q.dequeue()
    jitter = hypes['jitter']
    if phase == 'train':
        # Because these operations are not commutative, consider randomizing
        # randomize the order their operation.
        augment_level = jitter['augment_level']
        if augment_level > 0:
            image = tf.image.random_brightness(image, max_delta=30)
            image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
        if augment_level > 1:
            image = tf.image.random_hue(image, max_delta=0.15)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.6)

    if 'whitening' not in hypes['arch'] or \
            hypes['arch']['whitening']:
        image = tf.image.per_image_whitening(image)
        logging.info('Whitening is enabled.')
    else:
        logging.info('Whitening is disabled.')

    image = tf.expand_dims(image, 0)
    label = tf.expand_dims(label, 0)

    return image, label


def _dtypes(tensor_list_list):
    all_types = [[t.dtype for t in tl] for tl in tensor_list_list]
    types = all_types[0]
    for other_types in all_types[1:]:
        if other_types != types:
            raise TypeError("Expected types to be consistent: %s vs. %s." %
                            (", ".join(x.name for x in types),
                             ", ".join(x.name for x in other_types)))
    return types


def _enqueue_join(queue, tensor_list_list):
    enqueue_ops = [queue.enqueue(tl) for tl in tensor_list_list]
    queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))


def shuffle_join(tensor_list_list, capacity,
                 min_ad, phase):
    name = 'shuffel_input'
    types = _dtypes(tensor_list_list)
    queue = data_flow_ops.RandomShuffleQueue(
        capacity=capacity, min_after_dequeue=min_ad,
        dtypes=types)

    # Build enque Operations
    _enqueue_join(queue, tensor_list_list)

    full = (math_ops.cast(math_ops.maximum(0, queue.size() - min_ad),
                          dtypes.float32) * (1. / (capacity - min_ad)))
    # Note that name contains a '/' at the end so we intentionally do not place
    # a '/' after %s below.
    summary_name = (
        "queue/%s/fraction_over_%d_of_%d_full" %
        (name + '_' + phase, min_ad, capacity - min_ad))
    tf.scalar_summary(summary_name, full)

    dequeued = queue.dequeue(name='shuffel_deqeue')
    # dequeued = _deserialize_sparse_tensors(dequeued, sparse_info)
    return dequeued


def _processe_image(hypes, image):
    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    augment_level = hypes['jitter']['augment_level']
    if augment_level > 0:
        image = tf.image.random_brightness(image, max_delta=30)
        image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    if augment_level > 1:
        image = tf.image.random_hue(image, max_delta=0.15)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.6)

    return image


def inputs(hypes, q, phase):
    """Generate Inputs images."""
    if phase == 'val':
        image, label = q.dequeue()
        image = tf.expand_dims(image, 0)
        label = tf.expand_dims(label, 0)
        return image, label

    if not hypes['jitter']['fix_shape']:
        image, label = q.dequeue()
        nc = hypes["arch"]["num_classes"]
        label.set_shape([None, None, nc])
        image.set_shape([None, None, 3])
        image = tf.expand_dims(image, 0)
        label = tf.expand_dims(label, 0)
    else:
        image, label = q.dequeue_many(hypes['solver']['batch_size'])

    image = _processe_image(hypes, image)

    # Display the training images in the visualizer.
    tensor_name = image.op.name
    tf.image_summary(tensor_name + '/image', image)

    road = tf.expand_dims(tf.to_float(label[:, :, :, 0]), 3)
    tf.image_summary(tensor_name + '/gt_image', road)

    return image, label


def main():
    """main."""
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
