from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import tensorflow as tf


def get_learning_rate(hypes, step):
    lr = hypes['solver']['learning_rate']
    lr_step = hypes['solver']['learning_rate_step']
    if lr_step is not None:
        adjusted_lr = (lr * 0.5 ** max(0, (step / lr_step) - 2))
        return adjusted_lr
    else:
        return lr


def training(hypes, loss, global_step, learning_rate):
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

        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, tvars = zip(*grads_and_vars)

        clip_norm = hypes["clip_norm"]
        clipped_grads, norm = tf.clip_by_global_norm(grads, clip_norm)

        clipped_grads_and_vars = zip(clipped_grads, tvars)
        train_op = optimizer.apply_gradients(clipped_grads_and_vars,
                                             global_step=global_step)
    return train_op
