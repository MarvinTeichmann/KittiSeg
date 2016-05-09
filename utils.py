from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datetime import datetime
import time
import os
import logging

import config as cfg


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('debug', True, 'Soggy Leaves')


# usage: train.py --config=my_model_params.py
flags.DEFINE_string('hypes', cfg.default_config,
                    'File storing model parameters.')


def get_train_dir(hypes_fname):
    if FLAGS.debug:
        train_dir = os.path.join(cfg.model_dir, 'debug')
        logging.info(
            "Saving/Loading Model from debug Folder: %s ", train_dir)
        logging.info("Use --name=MYNAME to use Folder: %s ",
                     os.path.join(cfg.model_dir, "MYNAME"))
    else:
        json_name = hypes_fname.split('/')[-1].replace('.json', '')
        date = datetime.now().strftime('%Y_%m_%d_%H.%M')
        run_name = '%s_%s' % (json_name, date)
        train_dir = os.path.join(cfg.model_dir, run_name)\

    return train_dir


# TODO: right place to store placeholders

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
      batch_size: The batch size will be baked into both placeholders.

    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
      keep_prob: keep_prob placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.

    keep_prob = tf.placeholder("float")
    return keep_prob


def fill_feed_dict(kb, train):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
      kb: The keep prob placeholder.
      train: whether data set is on train.

    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.

    if train:
        feed_dict = {
            kb: 0.5}
    else:
        feed_dict = {
            kb: 1.0}
    return feed_dict


# TODO: right place to store eval?
