#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
import pdb
import json
import logging
import os.path
import sys

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)


import time

from shutil import copyfile

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

import utils as utils

flags = tf.app.flags
FLAGS = flags.FLAGS


def _copy_parameters_to_traindir(hypes, input_file, target_name, target_dir):
    """
    Helper to copy files defining the network to the saving dir.

    Parameters
    ----------
    input_file : str
        name of source file
    target_name : str
        target name
    traindir : str
        directory where training data is saved
    """
    target_file = os.path.join(target_dir, target_name)
    input_file = os.path.join(hypes['dirs']['base_path'], input_file)
    copyfile(input_file, target_file)


def initialize_training_folder(hypes):
    """Creating the training folder and copy all model files into it.

    The model will be executed from the training folder and all
    outputs will be saved there.

    Args:
      hypes: hypes
    """
    target_dir = os.path.join(hypes['dirs']['output_dir'], "model_files")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Creating an additional logging saving the console outputs
    # into the training folder
    logging_file = os.path.join(hypes['dirs']['output_dir'], "output.log")
    filewriter = logging.FileHandler(logging_file, mode='w')
    formatter = logging.Formatter(
        '%(asctime)s %(name)-3s %(levelname)-3s %(message)s')
    filewriter.setLevel(logging.INFO)
    filewriter.setFormatter(formatter)
    logging.getLogger('').addHandler(filewriter)

    # TODO: read more about loggers and make file logging neater.

    hypes_file = os.path.basename(tf.app.flags.FLAGS.hypes)
    _copy_parameters_to_traindir(
        hypes, hypes_file, "hypes.json", target_dir)
    _copy_parameters_to_traindir(
        hypes, hypes['model']['input_file'], "data_input.py", target_dir)
    _copy_parameters_to_traindir(
        hypes, hypes['model']['architecture_file'], "architecture.py",
        target_dir)
    _copy_parameters_to_traindir(
        hypes, hypes['model']['objective_file'], "objective.py", target_dir)
    _copy_parameters_to_traindir(
        hypes, hypes['model']['optimizer_file'], "solver.py", target_dir)


def maybe_download_and_extract(hypes):
    """
    Download the data if it isn't downloaded by now.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    """
    f = os.path.join(hypes['dirs']['base_path'], hypes['model']['input_file'])
    data_input = imp.load_source("input", f)
    if hasattr(data_input, 'maybe_download_and_extract'):
        data_input.maybe_download_and_extract(hypes, hypes['dirs']['data_dir'])


def write_precision_to_summary(precision, summary_writer, name, global_step,
                               sess):
    """
    TODO.

    Parameters
    ----------
    precision : TODO
        TODO
    summary_writer : TODO
        TODO
    name : TODO
        TODO
    global_step : TODO
        TODO
    sess : TODO
        TODO
    """
    # write result to summary
    summary = tf.Summary()
    # summary.ParseFromString(sess.run(summary_op))
    summary.value.add(tag='Evaluation/' + name + ' Precision',
                      simple_value=precision)
    summary_writer.add_summary(summary, global_step)


def do_eval(hypes, eval_correct, phase, sess):
    """Run one evaluation against the full epoch of data.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    eval_correct : TODO
        The Tensor that returns the number of correct predictions.
    sess : TODO
        The session in which the model has been trained.
    name : str
        Describes the data the evaluation is run on

    Returns
    -------
    TODO
    """
    # And run one epoch of eval.

    if phase == 'train':
        num_examples = hypes['data']['num_examples_per_epoch_for_train']
    if phase == 'val':
        num_examples = hypes['data']['num_examples_per_epoch_for_eval']

    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = num_examples // hypes['solver']['batch_size']
    num_examples = steps_per_epoch * hypes['solver']['batch_size']

    # run evaluation on num_examples many images
    for step in xrange(steps_per_epoch):
        true_count += sess.run(eval_correct[phase])

    precision = true_count / num_examples

    logging.info('Data: % s  Num examples: % d  Num correct: % d'
                 'Precision @ 1: % 0.04f ' %
                 (phase, num_examples, true_count, precision))

    return precision


def run_training(hypes):
    """Train model for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.

    # Tell TensorFlow that the model will be built into the default Graph.
    base_path = hypes['dirs']['base_path']
    f = os.path.join(base_path, hypes['model']['input_file'])
    data_input = imp.load_source("input", f)
    f = os.path.join(base_path, hypes['model']['architecture_file'])
    arch = imp.load_source("arch", f)
    f = os.path.join(base_path, hypes['model']['objective_file'])
    objective = imp.load_source("objective", f)
    f = os.path.join(base_path, hypes['model']['optimizer_file'])
    solver = imp.load_source("solver", f)

    with tf.Graph().as_default():

        global_step = tf.Variable(0.0, trainable=False)

        q, logits, decoder, = {}, {}, {}
        image_batch, label_batch = {}, {}
        eval_correct = {}

        # Add Input Producers to the Graph
        with tf.name_scope('Input'):
            q['train'] = data_input.create_queues(hypes, 'train')
            input_batch = data_input.inputs(hypes, q, 'train',
                                            hypes['dirs']['data_dir'])
            image_batch['train'], label_batch['train'] = input_batch

        logits['train'] = arch.inference(hypes, image_batch['train'], 'train')

        decoder['train'] = objective.decoder(hypes, logits['train'])

        # Add to the Graph the Ops for loss calculation.
        loss = objective.loss(hypes, decoder['train'], label_batch['train'])

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = solver.training(hypes, loss, global_step=global_step)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct['train'] = objective.evaluation(hypes, decoder['train'],
                                                     label_batch['train'])

        # Validation Cycle to the Graph
        with tf.name_scope('Validation'):
            q['val'] = data_input.create_queues(hypes, 'val')
            input_batch = data_input.inputs(hypes, q, 'val',
                                            hypes['dirs']['data_dir'])
            image_batch['val'], label_batch['val'] = input_batch

            tf.get_variable_scope().reuse_variables()

            logits['val'] = arch.inference(hypes, image_batch['val'], 'val')

            decoder['val'] = objective.decoder(hypes, logits['val'])

            eval_correct['val'] = objective.evaluation(hypes, decoder['val'],
                                                       label_batch['val'])

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
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        with tf.name_scope('data_load'):
            data_input.start_enqueuing_threads(hypes, q, sess,
                                               hypes['dirs']['data_dir'])

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(hypes['dirs']['output_dir'],
                                                graph=sess.graph)

        # And then after everything is built, start the training loop.
        solver = hypes['solver']
        start_time = time.time()
        for step in xrange(solver['max_steps']):

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.

            _, loss_value = sess.run([train_op, loss])

            # Write the summaries and print an overview fairly often.
            if step % int(utils.cfg.step_show) == 0:
                # Print status to stdout.
                duration = (time.time() - start_time) / 100
                examples_per_sec = solver['batch_size'] / duration
                sec_per_batch = float(duration)
                info_str = utils.cfg.step_str
                logging.info(info_str.format(step=step,
                                             total_steps=solver['max_steps'],
                                             loss_value=loss_value,
                                             sec_per_batch=sec_per_batch,
                                             examples_per_sec=examples_per_sec)
                             )
                # Update the events file.
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                start_time = time.time()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == solver['max_steps']:
                checkpoint_path = os.path.join(hypes['dirs']['output_dir'],
                                               'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                start_time = time.time()
                # Evaluate against the training set.

            if (step + 1) % 1000 == 0 or (step + 1) == solver['max_steps']:

                logging.info('Doing Evaluate with Training Data.')

                precision = do_eval(hypes, eval_correct, phase='train',
                                    sess=sess)
                write_precision_to_summary(precision, summary_writer,
                                           "Train", step, sess)

                logging.info('Doing Evaluation with Testing Data.')
                precision = do_eval(hypes, eval_correct, phase='val',
                                    sess=sess)
                write_precision_to_summary(precision, summary_writer,
                                           'val', step, sess)

                start_time = time.time()

        # stopping input Threads
        coord.request_stop()
        coord.join(threads)


def main(_):
    """Run main function."""
    if FLAGS.hypes is None:
        logging.error("No hypes are given.")
        logging.error("Usage: tv-train --hypes hypes.json")
        exit(1)

    if FLAGS.gpus is None:
        if 'TV_USE_GPUS' in os.environ:
            if os.environ['TV_USE_GPUS'] == 'force':
                logging.error('Please specify a GPU.')
                logging.error('Usage tv-train --gpus <ids>')
                exit(1)
            else:
                gpus = os.environ['TV_USE_GPUS']
                logging.info("GPUs are set to: %s", gpus)
                os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    else:
        logging.info("GPUs are set to: %s", FLAGS.gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

    with open(tf.app.flags.FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)

    utils.load_plugins()
    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)

    logging.info("Initialize Training Folder")
    initialize_training_folder(hypes)
    maybe_download_and_extract(hypes)
    logging.info("Start Training")
    run_training(hypes)


if __name__ == '__main__':
    tf.app.run()
