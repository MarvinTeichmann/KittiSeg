from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import tensorflow as tf


def get_learning_rate(hype, global_step):
    if "learning_rates" not in hype['solver']:
        return hype['solver']['learning_rates']

    for i, num in enumerate(hype['solver']['steps']):
        if global_step < num:
            return hype['solver']['learning_rates'][i-1]


def training(hype, loss, global_step):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    with tf.name_scope('train'):
        tf.scalar_summary(loss.op.name, loss)
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(hype['solver']['learning_rate'])
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Create a variable to track the global step.
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a
        # single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
