from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import re

import sys
import logging


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
    return tf.get_variable(name='weights', shape=shape, initializer=initializer)


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

    if wd and (tf.get_variable_scope().reuse == False):
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _conv_layer(name, bottom, num_filter,
                ksize=[3, 3], strides=[1, 1, 1, 1], padding='SAME'):
    with tf.variable_scope(name) as scope:
        n = bottom.get_shape()[3].value
        logging.debug("Layer: %s, Fan-in: %d" % (name, n))
        shape = [ksize[0], ksize[1], n, num_filter]
        num_input = ksize[0]*ksize[1]*n
        stddev = (2/num_input)**0.5
        logging.debug("Layer: %s, stddev: %f" % (name, stddev))
        # stddev = 1e-4
        weights = _weight_variable(shape, stddev)
        bias = _bias_variable([num_filter], constant=0.0)
        conv = tf.nn.conv2d(bottom, weights,
                            strides=strides, padding=padding)
        relu = tf.nn.relu(conv + bias, name=scope.name)
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
    with tf.variable_scope(name) as scope:
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        return tf.reshape(bottom, [-1, dim])


def _fc_layer_with_dropout(bottom, name, size,
                           train, wd=0.005, keep_prob=0.5):

    with tf.variable_scope(name) as scope:
        n1 = bottom.get_shape()[1].value
        stddev = (2/n1)**0.5
        #stddev = 0.04
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


def _softmax(bottom, num_classes):
    # Computing Softmax
    with tf.variable_scope('logits') as scope:
        n1 = bottom.get_shape()[1].value
        stddev = (1/n1)**0.5
        weights = _variable_with_weight_decay(shape=[n1, num_classes],
                                              stddev=stddev, wd=0.0)

        biases = _bias_variable([num_classes])
        logits = tf.add(tf.matmul(bottom, weights), biases, name=scope.name)
        _activation_summary(logits)

    return logits


def inference(H, images, train=True):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
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
    reshape = _reshape(pool2)

    # First Fully Convolutional Layer
    fc4 = _fc_layer_with_dropout(name='fc4', bottom=reshape,
                                 train=train, size=384)

    # Second Fully Convolutional Layer
    fc5 = _fc_layer_with_dropout(name='fc5', bottom=fc4,
                                 train=train, size=192)
    # Adding Softmax
    logits = _softmax(fc5, H['arch']['num_classes'])

    return logits


def loss(H, logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    # Convert from sparse integer labels in the range [0, NUM_CLASSSES)
    # to 1-hot dense float vectors (that is we will have batch_size vectors,
    # each with NUM_CLASSES values, all of which are 0.0 except there will
    # be a 1.0 in the entry corresponding to the label).
    with tf.name_scope('loss'):
        batch_size = tf.size(labels)
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size), 1)
        concated = tf.concat(1, [indices, labels])
        onehot_labels = tf.sparse_to_dense(
            concated, tf.pack([batch_size, H['arch']['num_classes']]), 1.0, 0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                                onehot_labels,
                                                                name='xentropy')
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss


def evaluation(H, logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, labels, 1)
        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))
