#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
import random
from seg_utils import seg_utils as seg


import tensorflow as tf


def _add_softmax(hypes, logits):
    num_classes = hypes['arch']['num_classes']
    with tf.name_scope('decoder'):
        logits = tf.reshape(logits, (-1, num_classes))
        # logits = logits + epsilon

        softmax = tf.nn.softmax(logits)

    return softmax


def decoder(hypes, logits, train):
    """Apply decoder to the logits.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].

    Return:
      logits: the logits are already decoded.
    """
    decoded_logits = {}
    decoded_logits['logits'] = logits
    decoded_logits['softmax'] = _add_softmax(hypes, logits)
    return decoded_logits


def loss(hypes, decoded_logits, labels):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    logits = decoded_logits['logits']
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, 2))
        shape = [logits.get_shape()[0], 2]
        # logits = logits + epsilon
        labels = tf.to_float(tf.reshape(labels, (-1, 2)))

        softmax = tf.nn.softmax(logits)
        head = hypes['arch']['weight']
        cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax), head),
                                       reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')

        tf.add_to_collection('total_losses', cross_entropy_mean)

        enc_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        dec_loss = tf.add_n(tf.get_collection('dec_losses'), name='total_loss')
        weight_loss = enc_loss + dec_loss

        total_loss = tf.add_n(tf.get_collection('total_losses'),
                              name='total_loss') + weight_loss

        losses = {}
        losses['total_loss'] = total_loss
        losses['xentropy'] = cross_entropy_mean
        losses['weight_loss'] = weight_loss

    return losses


def evaluation(hyp, images, labels, decoded_logits, losses, global_step):
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
    eval_list = []
    logits = tf.reshape(decoded_logits['logits'], (-1, 2))
    labels = tf.reshape(labels, (-1, 2))

    pred = tf.argmax(logits, dimension=1)

    negativ = tf.to_int32(tf.equal(pred, 0))
    tn = tf.reduce_sum(negativ*labels[:, 0])
    fn = tf.reduce_sum(negativ*labels[:, 1])

    positive = tf.to_int32(tf.equal(pred, 1))
    tp = tf.reduce_sum(positive*labels[:, 1])
    fp = tf.reduce_sum(positive*labels[:, 0])

    eval_list.append(('Acc. ', (tn+tp)/(tn + fn + tp + fp)))
    eval_list.append(('xentropy', losses['xentropy']))
    eval_list.append(('weight_loss', losses['weight_loss']))

    # eval_list.append(('Precision', tp/(tp + fp)))
    # eval_list.append(('True BG', tn/(tn + fp)))
    # eval_list.append(('True Street [Recall]', tp/(tp + fn)))

    return eval_list
