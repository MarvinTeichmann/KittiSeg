from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np

import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, "tensorflow_fcn/vgg16.npy")
data_dict = np.load(path, encoding='latin1').item()
print("npy file loaded")


def inference(hypes, images, train=True):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """

    up = build(images, train=True, num_classes=2, random_init_fc8=True)

    return up


def build(rgb, train=False, num_classes=20, random_init_fc8=False,
          debug=False):
    """
    Build the VGG model using loaded weights
    Parameters
    ----------
    rgb: image batch tensor
        Image in rgb shap. Scaled to Intervall [0, 255]
    train: bool
        Whether to build train or inference graph
    num_classes: int
        How many classes should be predicted (by fc8)
    random_init_fc8 : bool
        Whether to initialize fc8 layer randomly.
        Finetuning is required in this case.
    """
    # Convert RGB to BGR

    with tf.name_scope('Processing'):

        red, green, blue = tf.split(3, 3, rgb)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        if debug:
            bgr = tf.Print(bgr, [tf.shape(bgr)],
                           message='Shape of input image: ',
                           summarize=4, first_n=1)

    conv1_1 = _conv_layer(bgr, "conv1_1")
    conv1_2 = _conv_layer(conv1_1, "conv1_2")
    pool1 = _max_pool(conv1_2, 'pool1')

    if debug:
        pool1 = tf.Print(pool1, [tf.shape(pool1)],
                         message='Shape of pool1: ',
                         summarize=4, first_n=1)

    conv2_1 = _conv_layer(pool1, "conv2_1")
    conv2_2 = _conv_layer(conv2_1, "conv2_2")
    pool2 = _max_pool(conv2_2, 'pool2')

    if debug:
        pool2 = tf.Print(pool2, [tf.shape(pool2)],
                         message='Shape of pool2: ',
                         summarize=4, first_n=1)

    conv3_1 = _conv_layer(pool2, "conv3_1")
    conv3_2 = _conv_layer(conv3_1, "conv3_2")
    conv3_2 = _conv_layer(conv3_2, "conv3_3")
    pool3 = _max_pool(conv3_2, 'pool3')

    if debug:
        pool3 = tf.Print(pool3, [tf.shape(pool3)],
                         message='Shape of pool3: ',
                         summarize=4, first_n=1)

    conv4_1 = _conv_layer(pool3, "conv4_1")
    conv4_2 = _conv_layer(conv4_1, "conv4_2")
    conv4_3 = _conv_layer(conv4_2, "conv4_3")
    pool4 = _max_pool(conv4_3, 'pool4')

    if debug:
        pool4 = tf.Print(pool4, [tf.shape(pool4)],
                         message='Shape of pool4: ',
                         summarize=4, first_n=1)

    conv5_1 = _conv_layer(pool4, "conv5_1")
    conv5_2 = _conv_layer(conv5_1, "conv5_2")
    conv5_3 = _conv_layer(conv5_2, "conv5_3")
    pool5 = _max_pool(conv5_3, 'pool5')

    if debug:
        pool5 = tf.Print(pool5, [tf.shape(pool5)],
                         message='Shape of pool5: ',
                         summarize=4, first_n=1)

    fc6 = _fc_layer(pool5, "fc6")

    if train:
        fc6 = tf.nn.dropout(fc6, 0.5)

    if debug:
        fc6 = tf.Print(fc6, [tf.shape(fc6)],
                       message='Shape of fc6: ',
                       summarize=4, first_n=1)

    fc7 = _fc_layer(fc6, "fc7")
    if train:
        fc7 = tf.nn.dropout(fc7, 0.5)

    if debug:
        fc7 = tf.Print(fc7, [tf.shape(fc7)],
                       message='Shape of fc7: ',
                       summarize=4, first_n=1)

    if random_init_fc8:
        score_fr = _score_layer(fc7, "score_fr",
                                num_classes)
    else:
        score_fr = _fc_layer(fc7, "score_fr",
                             num_classes=num_classes,
                             relu=False)
    if debug:
        score_fr = tf.Print(score_fr, [tf.shape(score_fr)],
                            message='Shape of score_fr: ',
                            summarize=4, first_n=1)

    pred = tf.argmax(score_fr, dimension=3)

    up = _upscore_layer(score_fr, shape=tf.shape(bgr),
                        num_classes=num_classes,
                        name='up', ksize=64, stride=32)

    if debug:
        up = tf.Print(up, [tf.shape(up)],
                      message='Shape of score_fr: ',
                      summarize=4, first_n=1)

    pred_up = tf.argmax(up, dimension=3)

    return up


def _max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)


def _conv_layer(bottom, name):
    with tf.variable_scope(name) as scope:
        filt = get_conv_filter(name)
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        # Add summary to Tensorboard
        _activation_summary(relu)
        return relu


def _fc_layer(bottom, name, num_classes=None,
              relu=True):
    with tf.variable_scope(name) as scope:
        shape = bottom.get_shape().as_list()

        if name == 'fc6':
            filt = get_fc_weight_reshape(name, [7, 7, 512, 4096])
        elif name == 'score_fr':
            name = 'fc8'  # Name of score_fr layer in VGG Model
            filt = get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                         num_classes=num_classes)
        else:
            filt = get_fc_weight_reshape(name, [1, 1, 4096, 4096])
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        conv_biases = get_bias(name, num_classes=num_classes)
        bias = tf.nn.bias_add(conv, conv_biases)

        if relu:
            bias = tf.nn.relu(bias)

        _activation_summary(bias)
        return bias


def _score_layer(bottom, name, num_classes):
    with tf.variable_scope(name) as scope:
        # get number of input channels
        in_features = bottom.get_shape()[3].value
        shape = [1, 1, in_features, num_classes]
        # He initialization Sheme
        num_input = in_features
        stddev = (2 / num_input)**0.5
        # Apply convolution
        w_decay = 1e-4
        weights = _variable_with_weight_decay(shape, stddev, w_decay)
        conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
        # Apply bias
        conv_biases = _bias_variable([num_classes], constant=0.0)
        bias = tf.nn.bias_add(conv, conv_biases)

        _activation_summary(bias)

        return bias


def _upscore_layer(bottom, shape,
                   num_classes, name,
                   ksize=4, stride=2,
                   wd=5e-4):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value

        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.pack(new_shape)

        logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
        f_shape = [ksize, ksize, num_classes, in_features]

        # create
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input)**0.5

        weights = get_deconv_filter(f_shape, wd)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

    _activation_summary(deconv)
    return deconv


def get_deconv_filter(f_shape, wd):
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)


def get_conv_filter(name):
    init = tf.constant_initializer(value=data_dict[name][0],
                                   dtype=tf.float32)
    shape = data_dict[name][0].shape
    print('Layer name: %s' % name)
    print('Layer shape: %s' % str(shape))
    return tf.get_variable(name="filter", initializer=init, shape=shape)


def get_bias(name, num_classes=None):
    bias_wights = data_dict[name][1]
    shape = data_dict[name][1].shape
    if name == 'fc8':
        bias_wights = _bias_reshape(bias_wights, shape[0],
                                    num_classes)
        shape = [num_classes]
    init = tf.constant_initializer(value=bias_wights,
                                   dtype=tf.float32)
    return tf.get_variable(name="biases", initializer=init, shape=shape)


def get_fc_weight(name):
    init = tf.constant_initializer(value=data_dict[name][0],
                                   dtype=tf.float32)
    shape = data_dict[name][0].shape
    return tf.get_variable(name="weights", initializer=init, shape=shape)


def _bias_reshape(bweight, num_orig, num_new):
    n_averaged_elements = num_orig//num_new
    avg_bweight = np.zeros(num_new)
    for i in range(0, num_orig, n_averaged_elements):
        start_idx = i
        end_idx = start_idx + n_averaged_elements
        avg_idx = start_idx//n_averaged_elements
        avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
    return avg_bweight


def _summary_reshape(fweight, shape, num_new):
    num_orig = shape[3]
    shape[3] = num_new
    n_averaged_elements = num_orig//num_new
    avg_fweight = np.zeros(shape)
    for i in range(0, num_orig, n_averaged_elements):
        start_idx = i
        end_idx = start_idx + n_averaged_elements
        avg_idx = start_idx//n_averaged_elements
        avg_fweight[:, :, :, avg_idx] = np.mean(
            fweight[:, :, :, start_idx:end_idx], axis=3)
    return avg_fweight


def _variable_with_weight_decay(shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal
    distribution.
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


def _bias_variable(shape, constant=0.0):
    initializer = tf.constant_initializer(constant)
    return tf.get_variable(name='biases', shape=shape,
                           initializer=initializer)


def get_fc_weight_reshape(name, shape, num_classes=None):
    print('Layer name: %s' % name)
    print('Layer shape: %s' % shape)
    weights = data_dict[name][0]
    weights = weights.reshape(shape)
    if num_classes is not None:
        weights = _summary_reshape(weights, shape,
                                   num_new=num_classes)
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="weights", initializer=init, shape=shape)


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
