"""Evaluation of the model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

import logging
import sys
import imp

import tensorvision.utils as utils


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")


# TODO: Iterate over all possible Values
# Write Values to Tensorboard


def evaluate(train_dir):
    """
    Load the model and run evaluation.

    Current Version runs the evaluation defined in network.evaluation 
    and prints the output to std out.

    Parameters
    ----------
    train_dir : str
      Path to a directory which includes a folder model_files. This folder
      has to include a params.py, input.py and a network.py
    """
    target_dir = os.path.join(train_dir, "model_files")
    params = imp.load_source("params", os.path.join(target_dir, "params.py"))
    data_input = imp.load_source("input", os.path.join(target_dir, "input.py"))
    network = imp.load_source("network",
                              os.path.join(target_dir, "network.py"))

    with tf.Graph().as_default():
        # Retrieve images and labels
        eval_data = FLAGS.eval_data == 'test'
        images, labels = data_input.inputs(eval_data=eval_data,
                                           data_dir=utils.cfg.data_dir,
                                           batch_size=params.batch_size)

        # Generate placeholders for the images and labels.
        keep_prob = utils.placeholder_inputs(params.batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = network.inference(images, keep_prob)

        # Add to the Graph the Ops for loss calculation.
        loss = network.loss(logits, labels)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = network.evaluation(logits, labels)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No checkpoints found! ")
            exit(1)

        print("Doing Evaluation with lots of data")
        utils.do_eval(sess=sess,
                      eval_correct=eval_correct,
                      keep_prob=keep_prob,
                      num_examples=params.num_examples_per_epoch_for_eval,
                      params=params,
                      name="eval")


def main(_):
    """Orchestrate the evaluation of a model in the default training dir."""
    train_dir = utils.get_train_dir()
    evaluate(train_dir)

if __name__ == '__main__':
    tf.app.run()
