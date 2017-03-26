"""
The MIT License (MIT)

Original Work: Copyright (c) 2016 Ryan Dahl
(See: https://github.com/ry/tensorflow-resnet)

Modified Work: Copyright (c) 2017 Marvin Teichmann

For details see 'licenses/RESNET_LICENSE.txt'
"""
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

import datetime
import numpy as np
import os
import time

import logging

MOVING_AVERAGE_DECAY = 0.998
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
MOMENTUM = 0.9
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = tf.GraphKeys.UPDATE_OPS
# must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]


network_url = "Not yet uploaded."


def checkpoint_fn(layers):
    return 'ResNet-L%d.ckpt' % layers


def inference(hypes, images, train=True,
              num_classes=1000,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              preprocess=True,
              bottleneck=True):
    # if preprocess is True, input should be RGB [0,1], otherwise BGR with mean
    # subtracted

    layers = hypes['arch']['layers']

    if layers == 50:
        num_blocks = [3, 4, 6, 3]
    elif layers == 101:
        num_blocks = [3, 4, 23, 3]
    elif layers == 152:
        num_blocks = [3, 8, 36, 3]
    else:
        assert()

    if preprocess:
        x = _imagenet_preprocess(images)

    is_training = tf.convert_to_tensor(train,
                                       dtype='bool',
                                       name='is_training')

    logits = {}

    with tf.variable_scope('scale1'):
        x = _conv(x, 64, ksize=7, stride=2)
        x = _bn(x, is_training, hypes)
        x = _relu(x)
        scale1 = x

    with tf.variable_scope('scale2'):
        x = _max_pool(x, ksize=3, stride=2)
        x = stack(x, num_blocks[0], 64, bottleneck, is_training, stride=1,
                  hypes=hypes)
        scale2 = x

    with tf.variable_scope('scale3'):
        x = stack(x, num_blocks[1], 128, bottleneck, is_training, stride=2,
                  hypes=hypes)
        scale3 = x

    with tf.variable_scope('scale4'):
        x = stack(x, num_blocks[2], 256, bottleneck, is_training, stride=2,
                  hypes=hypes)
        scale4 = x

    with tf.variable_scope('scale5'):
        x = stack(x, num_blocks[3], 512, bottleneck, is_training, stride=2,
                  hypes=hypes)
        scale5 = x

    logits['images'] = images

    logits['fcn_in'] = scale5
    logits['feed2'] = scale4
    logits['feed4'] = scale3

    logits['early_feat'] = scale3
    logits['deep_feat'] = scale5

    if train:
        restore = tf.global_variables()
        hypes['init_function'] = _initalize_variables
        hypes['restore'] = restore

    return logits


def _initalize_variables(hypes):
    if hypes['load_pretrained']:
        logging.info("Pretrained weights are loaded.")
        logging.info("The model is fine-tuned from previous training.")
        restore = hypes['restore']
        init = tf.global_variables_initializer()
        sess = tf.get_default_session()
        sess.run(init)

        saver = tf.train.Saver(var_list=restore)

        layers = hypes['arch']['layers']

        assert layers in [50, 101, 152]

        filename = checkpoint_fn(layers)

        if 'TV_DIR_DATA' in os.environ:
            filename = os.path.join(os.environ['TV_DIR_DATA'], 'weights',
                                    "tensorflow_resnet", filename)
        else:
            filename = os.path.join('DATA', 'weights', "tensorflow_resnet",
                                    filename)

        if not os.path.exists(filename):
            logging.error("File not found: {}".format(filename))
            logging.error("Please download weights from here: {}"
                          .format('network_url'))
            exit(1)

        logging.info("Loading weights from disk.")
        saver.restore(sess, filename)
    else:
        logging.info("Random initialization performed.")
        sess = tf.get_default_session()
        init = tf.global_variables_initializer()
        sess.run(init)


def _imagenet_preprocess(rgb):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    red, green, blue = tf.split(
        axis=3, num_or_size_splits=3, value=rgb * 255.0)
    bgr = tf.concat(axis=3, values=[blue, green, red])
    bgr -= IMAGENET_MEAN_BGR
    return bgr


def stack(x, num_blocks, filters_internal, bottleneck, is_training, stride,
          hypes):
    for n in range(num_blocks):
        s = stride if n == 0 else 1
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x,
                      filters_internal,
                      bottleneck=bottleneck,
                      is_training=is_training,
                      stride=s,
                      hypes=hypes)
    return x


def block(x, filters_internal, is_training, stride, bottleneck, hypes):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed.
    # That is the case when bottleneck=False but when bottleneck is
    # True, filters_internal*4 filters are outputted. filters_internal
    # is how many filters
    # the 3x3 convs output internally.
    if bottleneck:
        filters_out = 4 * filters_internal
    else:
        filters_out = filters_internal

    shortcut = x  # branch 1

    if bottleneck:
        with tf.variable_scope('a'):
            x = _conv(x, filters_internal, ksize=1, stride=stride)
            x = _bn(x, is_training, hypes)
            x = _relu(x)

        with tf.variable_scope('b'):
            x = _conv(x, filters_internal, ksize=3, stride=1)
            x = _bn(x, is_training, hypes)
            x = _relu(x)

        with tf.variable_scope('c'):
            x = _conv(x, filters_out, ksize=1, stride=1)
            x = _bn(x, is_training, hypes)
    else:
        with tf.variable_scope('A'):
            x = _conv(x, filters_internal, ksize=3, stride=stride)
            x = _bn(x, is_training, hypes)
            x = _relu(x)

        with tf.variable_scope('B'):
            x = _conv(x, filters_out, ksize=3, stride=1)
            x = _bn(x, is_training, hypes)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or stride != 1:
            shortcut = _conv(shortcut, filters_out, ksize=1, stride=stride)
            shortcut = _bn(shortcut, is_training, hypes)

    return _relu(x + shortcut)


def _relu(x):
    return tf.nn.relu(x)


def _bn(x, is_training, hypes):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer(),
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer(),
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)

    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean,
                                                               BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    if hypes['use_moving_average_bn']:
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

        mean, variance = control_flow_ops.cond(
            is_training, lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))
    else:
        mean, variance = mean, variance

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    # x.set_shape(inputs.get_shape()) ??

    return x


def _fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def _conv(x, filters_out, ksize=3, stride=1):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')
