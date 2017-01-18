from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def decoder(hypes, logits):
    """
    Apply decoder to the logits.

    Parameters
    ----------
    logits : Logits tensor, float - [batch_size, NUM_CLASSES].

    Returns
    -------
    logits : the logits are already decoded.
    """
    return logits


def loss(hypes, logits, labels):
    """
    Calculate the loss from the logits and the labels.

    Parameters
    ----------
    logits : Logits tensor, float - [batch_size, NUM_CLASSES].
    labels : Labels tensor, int32 - [batch_size].

    Returns
    -------
    loss : Loss tensor of type float.
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
            concated,
            tf.pack([batch_size, hypes['arch']['num_classes']]),
            1.0, 0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits, onehot_labels, name='xentropy')

        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss


def evaluation(hypes, logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Parameters
    ----------
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns
    -------
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
