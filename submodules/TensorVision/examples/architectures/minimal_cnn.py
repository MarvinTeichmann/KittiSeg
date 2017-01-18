"""Minimal CNN for testing purposes."""

import tensorflow as tf
import re

import logging


def weight_variable(name, shape, stddev=0.1):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(name, shape=shape, initializer=initializer)


def bias_variable(name, shape, constant=0.1):
    initializer = tf.constant_initializer(constant)
    return tf.get_variable(name, shape=shape, initializer=initializer)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name=name
                          )


def _activation_summary(x):
    """Helper to create summaries for activations.

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


def inference(H, images, train=True):
    """Build the MNIST model up to where it may be used for inference.

    Parameters
    ----------
    images: Images placeholder, from inputs().
    train: whether the network is used for train of inference

    Returns
    -------
    softmax_linear: Output tensor with the computed logits.
    """
    num_filter_1 = 32
    num_filter_2 = 64

    # First Convolutional Layer
    with tf.variable_scope('Conv1') as scope:
        # Adding Convolutional Layers
        W_conv1 = weight_variable(
            'weights', [5, 5, H['arch']['num_channels'], num_filter_1])
        b_conv1 = bias_variable('biases', [num_filter_1])

        h_conv1 = tf.nn.relu(
            conv2d(images, W_conv1) + b_conv1, name=scope.name)
        _activation_summary(h_conv1)

    # First Pooling Layer
    h_pool1 = max_pool_2x2(h_conv1, name='pool1')

    # Second Convolutional Layer
    with tf.variable_scope('Conv2') as scope:
        W_conv2 = weight_variable(
            'weights', [5, 5, num_filter_1, num_filter_2])
        b_conv2 = bias_variable('biases', [num_filter_2])

        h_conv2 = tf.nn.relu(
            conv2d(h_pool1, W_conv2) + b_conv2, name=scope.name)
        _activation_summary(h_conv2)

    # Second Pooling Layer
    h_pool2 = max_pool_2x2(h_conv2, name='pool2')

    # Find correct dimension
    dim = 1
    for d in h_pool2.get_shape()[1:].as_list():
        dim *= d

    # Adding Fully Connected Layers
    with tf.variable_scope('fc1') as scope:
        W_fc1 = weight_variable('weights', [dim, 1024])
        b_fc1 = bias_variable('biases', [1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, dim])
        h_fc1 = tf.nn.relu(
            tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name=scope.name)
        _activation_summary(h_fc1)

    # Adding Dropout
    if train:
        h_fc1 = tf.nn.dropout(h_fc1, 0.5, name='dropout')

    with tf.variable_scope('logits') as scope:
        W_fc2 = weight_variable('weights', [1024, H['arch']['num_classes']])
        b_fc2 = bias_variable('biases', [H['arch']['num_classes']])
        logits = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2, name=scope.name)
        _activation_summary(logits)

    return logits
