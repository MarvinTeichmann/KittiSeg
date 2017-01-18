"""
VGG network architecture.

See http://arxiv.org/abs/1409.1556
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf


def _activation_summary(x):
    """
    Create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Parameters
    ----------
    x : Tensor
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _weight_variable(shape, stddev=0.01):
    """
    Create initialized weight variables.

    Parameters
    ----------
    name : str
        Name of the variable.
    shape : list of ints
    stddev : float
        Standard deviation of a truncated Gaussian.
    """
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(name='weights', shape=shape,
                           initializer=initializer)


def _bias_variable(shape, constant=0.0):
    initializer = tf.constant_initializer(constant)
    return tf.get_variable(name='biases', shape=shape, initializer=initializer)


def _variable_with_weight_decay(shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Parameters
    ----------
    name: name of the variable
    shape: list of ints
    stddev : float
        Standard deviation of a truncated Gaussian.
    wd: add L2Loss weight decay multiplied by this float. If None, weight
      decay is not added for this variable.

    Returns
    -------
    Variable Tensor
    """
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable('weights', shape=shape,
                          initializer=initializer)

    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _conv_layer(name, bottom, num_filter,
                ksize=[3, 3], strides=[1, 1, 1, 1], padding='SAME'):
    with tf.variable_scope(name) as scope:
        # get number of input channels
        n = bottom.get_shape()[3].value
        if n is None:
            # if placeholder are used, n might be undefined
            # this should only happen in the first layer.
            logging.warning("None type in Layer: %s", name)
            # Assume RGB image in that case.
            n = 3
        logging.debug("Layer: %s, Fan-in: %d" % (name, n))
        shape = [ksize[0], ksize[1], n, num_filter]
        num_input = ksize[0] * ksize[1] * n
        stddev = (2 / num_input)**0.5
        logging.debug("Layer: %s, stddev: %f" % (name, stddev))
        weights = _weight_variable(shape, stddev)
        bias = _bias_variable([num_filter], constant=0.0)
        conv = tf.nn.conv2d(bottom, weights,
                            strides=strides, padding=padding)
        bias_layer = tf.nn.bias_add(conv, bias, name=scope.name)
        relu = tf.nn.relu(bias_layer, name=scope.name)
        _activation_summary(relu)
    return relu


def _max_pool(name, bottom, ksize=[1, 2, 2, 1],
              strides=None, padding='SAME'):
    if strides is None:
        strides = ksize

    return tf.nn.max_pool(bottom, ksize=ksize, strides=strides,
                          padding=padding, name=name)


def _reshape(bottom, name="reshape"):
    logging.debug("Size of Reshape %s " % bottom.get_shape())
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        return tf.reshape(bottom, [-1, dim])


def _fc_layer_with_dropout(bottom, name, size,
                           train, wd=0.005, keep_prob=0.5):

    with tf.variable_scope(name) as scope:
        n1 = bottom.get_shape()[1].value
        stddev = (2 / n1)**0.5

        logging.debug("Layer: %s, Size: %d", name, n1)
        logging.debug("Layer: %s, stddev: %f", name, stddev)
        weights = _variable_with_weight_decay(shape=[n1, size],
                                              stddev=stddev, wd=wd)
        biases = _bias_variable([size])

        fullc = tf.nn.relu_layer(bottom, weights, biases, name=scope.name)
        _activation_summary(fullc)

        # Adding Dropout
        if train:
            fullc = tf.nn.dropout(fullc, keep_prob, name='dropout')

        return fullc


def _logits(bottom, num_classes):
    # Computing Softmax
    with tf.variable_scope('logits') as scope:
        n1 = bottom.get_shape()[1].value
        stddev = (1 / n1)**0.5
        weights = _variable_with_weight_decay(shape=[n1, num_classes],
                                              stddev=stddev, wd=0.0)

        biases = _bias_variable([num_classes])
        logits = tf.add(tf.matmul(bottom, weights), biases, name=scope.name)
        _activation_summary(logits)

    return logits


def inference(hypes, images, train=True):
    """Build the MNIST model up to where it may be used for inference.

    Parameters
    ----------
    images : Images placeholder, from inputs().
    train : whether the network is used for train of inference

    Returns
    -------
    softmax_linear : Output tensor with the computed logits.
    """
    # First Block of Convolutional Layers
    # with tf.name_scope('Pool1') as scope:
    conv1_1 = _conv_layer(name="conv1_1", bottom=images, num_filter=32)
    conv1_2 = _conv_layer(name="conv1_2", bottom=conv1_1, num_filter=32)
    pool1 = _max_pool(name="pool1", bottom=conv1_2)

    # Second Block of Convolutional Layers
    # with tf.name_scope('Pool2') as scope:
    conv2_1 = _conv_layer(name="conv2_1", bottom=pool1, num_filter=64)
    conv2_2 = _conv_layer(name="conv2_2", bottom=conv2_1, num_filter=64)
    pool2 = _max_pool(name="pool2", bottom=conv2_2)

    # Third Block of Convolutional Layers
    # with tf.name_scope('Pool3') as scope:
    conv3_1 = _conv_layer(name="conv3_1", bottom=pool2, num_filter=128)
    conv3_2 = _conv_layer(name="conv3_2", bottom=conv3_1, num_filter=128)
    pool3 = _max_pool(name="pool3", bottom=conv3_2)

    # Reshape for fully convolutional Layer
    reshape = _reshape(pool3)

    # First Fully Convolutional Layer
    fc4 = _fc_layer_with_dropout(name='fc4', bottom=reshape,
                                 train=train, size=384)

    # Second Fully Convolutional Layer
    fc5 = _fc_layer_with_dropout(name='fc5', bottom=fc4,
                                 train=train, size=192)
    # Adding Softmax
    logits = _logits(fc5, hypes['arch']['num_classes'])

    return logits
