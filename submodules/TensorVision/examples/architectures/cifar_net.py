import tensorflow as tf
import re

import sys
import logging


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Parameters
    ----------
    name : str
        Name of the variable
    shape : list of ints
    stddev : float
        standard deviation of a truncated Gaussian
    wd : add L2Loss weight decay multiplied by this float. If None, weight
      decay is not added for this Variable.

    Returns
    -------
    Variable Tensor
    """

    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable(name, shape=shape,
                          initializer=initializer)

    if wd and (not tf.get_variable_scope().reuse):
        var_name = tf.get_variable_scope().name + name
        logging.debug("Adding weight decay for variable %s", var_name)
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def weight_variable(name, shape, stddev=0.1):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(name, shape=shape, initializer=initializer)


def bias_variable(name, shape, constant=0.1):
    initializer = tf.constant_initializer(constant)
    return tf.get_variable(name, shape=shape, initializer=initializer)


def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


def max_pool_3x3(x, name):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name=name
                          )


def normalization(x, name):
    return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                     name=name)


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def inference(H, images, train=True):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """

    num_filter_1 = 64
    num_filter_2 = 64

    # First Convolutional Layer
    with tf.variable_scope('Conv1') as scope:
        W_conv1 = _variable_with_weight_decay('weights',
                                              shape=[5, 5,
                                                     H['arch']['num_channels'],
                                                     num_filter_1],
                                              stddev=1e-4, wd=0.0)
        b_conv1 = bias_variable('biases', [num_filter_1], constant=0.0)

        x_image = tf.reshape(images, [-1,
                                      H['arch']['image_size'],
                                      H['arch']['image_size'],
                                      H['arch']['num_channels']])

        h_conv1 = tf.nn.relu(
            conv2d(x_image, W_conv1) + b_conv1, name=scope.name)
        _activation_summary(h_conv1)

    # First Pooling Layer
    h_pool1 = max_pool_3x3(h_conv1, name='pool1')
    # First Normalization
    norm1 = normalization(h_pool1, name='norm1')

    # Second Convolutional Layer
    with tf.variable_scope('Conv2') as scope:
        W_conv2 = _variable_with_weight_decay('weights',
                                              [5, 5, num_filter_1,
                                               num_filter_2],
                                              stddev=1e-4, wd=0.0)
        b_conv2 = bias_variable('biases', [num_filter_2])

        h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2, name=scope.name)
        _activation_summary(h_conv2)

    # Second Pooling Layer
    h_pool2 = max_pool_3x3(h_conv2, name='pool2')

    # Second Normalization
    norm2 = normalization(h_pool2, name='norm1')

    # Fully Connected 1
    with tf.variable_scope('FullC1') as scope:
        # Move everything into depth so we can perform a single matrix
        # multiply.
        dim = 1
        for d in norm2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(norm2, [-1, dim])

        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = bias_variable('biases', [384])
        fullc1 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
        _activation_summary(fullc1)

    # local4
    with tf.variable_scope('FullC2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = bias_variable('biases', [192])
        fullc2 = tf.nn.relu_layer(fullc1, weights, biases, name=scope.name)
        _activation_summary(fullc2)

    # Computing Softmax
    with tf.variable_scope('logits') as scope:
        W_fc2 = _variable_with_weight_decay('weights',
                                            [192, H['arch']['num_classes']],
                                            stddev=1/192.0, wd=0.0)
        b_fc2 = bias_variable('biases', [H['arch']['num_classes']])
        logits = tf.add(tf.matmul(fullc2, W_fc2), b_fc2, name=scope.name)
        _activation_summary(logits)

    return logits
