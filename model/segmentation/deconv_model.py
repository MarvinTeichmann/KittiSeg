from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import ipdb

import tensorflow as tf


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


def _weight_variable(shape, stddev=0.01):
    """Helper to create initialized weight Variables.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
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

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
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
                ksize=[3, 3], strides=[1, 1, 1, 1], wd=5e-3, padding='SAME'):
    with tf.variable_scope(name) as scope:
        n = bottom.get_shape()[3].value
        logging.debug("Layer: %s, Fan-in: %d" % (name, n))
        shape = [ksize[0], ksize[1], n, num_filter]
        num_input = ksize[0] * ksize[1] * n
        stddev = (2 / num_input)**0.5
        logging.debug("Layer: %s, stddev: %f" % (name, stddev))
        weights = _variable_with_weight_decay(shape, stddev, wd)
        bias = _bias_variable([num_filter], constant=0.0)
        conv = tf.nn.conv2d(bottom, weights,
                            strides=strides, padding=padding)
        relu = tf.nn.relu(conv + bias, name=scope.name)
        _activation_summary(relu)
    return relu


def _upsample_layer(name, bottom1, bottom2, num_filter,
                    ksize=[3, 3], strides=[1, 2, 2, 1],
                    wd=5e-3, padding='SAME'):
    with tf.variable_scope(name) as scope:
        in_features = bottom1.get_shape()[3].value
        shape = tf.shape(bottom2)
        new_shape = [shape[0], shape[1], shape[2], num_filter]
        output_shape = tf.pack(new_shape)

        logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
        shape = [ksize[0], ksize[1], num_filter, in_features]
        num_input = ksize[0] * ksize[1] * in_features / strides[1]
        stddev = (2 / num_input)**0.5
        logging.debug("Layer: %s, stddev: %f" % (name, stddev))
        weights = _variable_with_weight_decay(shape, stddev, wd)
        bias = _bias_variable([num_filter], constant=0.0)
        deconv = tf.nn.conv2d_transpose(bottom1, weights, output_shape,
                                        strides=strides, padding=padding)
        relu = tf.nn.relu(deconv + bias, name=scope.name)
        _activation_summary(relu)

        concate = tf.concat(3, [relu, bottom2])

    return concate


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


def _logits(bottom, num_classes, wd=5e-3):
    # Computing Softmax
    name = "logits"
    ksize = [1, 1]
    with tf.variable_scope("logits") as scope:
        n = bottom.get_shape()[3].value
        logging.debug("Layer: %s, Fan-in: %d" % (name, n))
        shape = [ksize[0], ksize[1], n, num_classes]
        num_input = ksize[0] * ksize[1] * n
        stddev = (2 / num_input)**0.5
        logging.debug("Layer: %s, stddev: %f" % (name, stddev))
        weights = _variable_with_weight_decay(shape, stddev, wd)
        bias = _bias_variable([num_classes], constant=0.0)
        conv = tf.nn.conv2d(bottom, weights, strides=[1, 1, 1, 1],
                            padding='SAME')
        logits = tf.nn.bias_add(conv, bias, name=scope.name)
    return logits


def inference(hypes, images, train=True):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """

    keep_prob = 0.5

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
    conv4_1 = _conv_layer(name="conv4_1", bottom=pool3, num_filter=256)
    conv4_2 = _conv_layer(name="conv4_2", bottom=conv4_1, num_filter=256)
    conv4_3 = _conv_layer(name="conv4_3", bottom=conv4_2, num_filter=256)

    if train:
        conv4_3 = tf.nn.dropout(conv4_3, keep_prob, name='dropout_conv4_3')

    conv4_4 = _conv_layer(name="conv4_4", bottom=conv4_3, num_filter=256)
    conv4_5 = _conv_layer(name="conv4_5", bottom=conv4_4, num_filter=256)
    conv4_6 = _conv_layer(name="conv4_6", bottom=conv4_5, num_filter=256)

    if train:
        conv4_6 = tf.nn.dropout(conv4_6, keep_prob, name='dropout_conv4_6')

    # Upsample layer 3
    up3 = _upsample_layer(name="up4", bottom1=conv4_6, bottom2=conv3_2,
                          num_filter=128)
    conv3_up = _conv_layer(name="conv3_up", bottom=up3, num_filter=128,
                           ksize=[1, 1])

    # Upsample layer 2
    up2 = _upsample_layer(name="up2", bottom1=conv3_up, bottom2=conv2_2,
                          num_filter=96)
    conv2_up = _conv_layer(name="conv2_up", bottom=up2, num_filter=96,
                           ksize=[1, 1])

    # Upsample layer 1
    up1 = _upsample_layer(name="up1", bottom1=conv2_up, bottom2=conv1_2,
                          num_filter=96)
    conv1_up = _conv_layer(name="conv1_up", bottom=up1, num_filter=96,
                           ksize=[1, 1])

    if train:
        conv1_up = tf.nn.dropout(conv1_up, keep_prob, name='dropout_up')

    # Adding Softmax
    logits = _logits(bottom=conv1_up, num_classes=2)

    return logits
