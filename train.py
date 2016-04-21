"""Trains, Evaluates and Saves the model network using a Queue."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path
import time
import logging
import sys
import imp
from shutil import copyfile

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import utils as utils

flags = tf.app.flags
FLAGS = flags.FLAGS


# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def _copy_parameters_to_traindir(input_file, target_name, target_dir):
    """Helper to copy files defining the network to the saving dir.

    Args:
      input_file: name of source file
      target_name: target name
      traindir: directory where training data is saved
    """
    target_file = os.path.join(target_dir, target_name)
    copyfile(input_file, target_file)


def initialize_training_folder(hypes, train_dir):
    """Creating the training folder and copy all model files into it.

    The model will be executed from the training folder and all
    outputs will be saved there.

    Args:
      hypes: hypes
      train_dir: The training folder
    """
    target_dir = os.path.join(train_dir, "model_files")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Creating an additional logging saving the console outputs
    # into the training folder
    logging_file = os.path.join(train_dir, "output.log")
    filewriter = logging.FileHandler(logging_file, mode='w')
    formatter = logging.Formatter(
        '%(asctime)s %(name)-3s %(levelname)-3s %(message)s')
    filewriter.setLevel(logging.INFO)
    filewriter.setFormatter(formatter)
    logging.getLogger('').addHandler(filewriter)

    # TODO: read more about loggers and make file logging neater.

    config_file = tf.app.flags.FLAGS.config
    _copy_parameters_to_traindir(config_file, "hypes.json", target_dir)
    _copy_parameters_to_traindir(
        hypes['model']['input_file'], "data_input.py", target_dir)
    _copy_parameters_to_traindir(
        hypes['model']['arch_file'], "architecture.py", target_dir)
    _copy_parameters_to_traindir(
        hypes['model']['solver_file'], "solver.py", target_dir)


def maybe_download_and_extract(hypes, train_dir):
    data_input = imp.load_source("input", hypes['model']['input_file'])
    if hasattr(data_input, 'maybe_download_and_extract'):
        target_dir = os.path.join(train_dir, "model_files")
        data_input.maybe_download_and_extract(hypes, utils.cfg.data_dir)


def write_precision_to_summary(precision, summary_writer, name, global_step,
                               sess):
    # write result to summary
    summary = tf.Summary()
    # summary.ParseFromString(sess.run(summary_op))
    summary.value.add(tag='Evaluation/' + name + ' Precision',
                      simple_value=precision)
    summary_writer.add_summary(summary, global_step)


def do_eval(hypes, eval_correct, phase, sess):
    """Runs one evaluation against the full epoch of data.

    Args:
      hypes: hypes
      eval_correct: The Tensor that returns the number of correct predictions.
      sess: The session in which the model has been trained.
      name: string descriping the data the evaluation is run on
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
        start_time = time.time()
        true_count += sess.run(eval_correct[phase])
        duration = time.time() - start_time

    precision = true_count / num_examples

    logging.info('Data: %s  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                 (phase, num_examples, true_count, precision))

    return precision


def run_training(hypes, train_dir):
    """Train model for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.

    # Tell TensorFlow that the model will be built into the default Graph.
    target_dir = os.path.join(train_dir, "model_files")
    data_input = imp.load_source("input", hypes['model']['input_file'])
    arch = imp.load_source("arch", hypes['model']['arch_file'])
    solver = imp.load_source("solver", hypes['model']['solver_file'])

    with tf.Graph().as_default():

        global_step = tf.Variable(0.0, trainable=False)

        q, logits = {}, {}
        image_batch, label_batch = {}, {}
        eval_correct = {}

        # Add Input Producers to the Graph
        with tf.name_scope('Input'):
            q['train'] = data_input.create_queues(hypes, 'train')
            input_batch = data_input.inputs(hypes, q, 'train',
                                            utils.cfg.data_dir)
            image_batch['train'], label_batch['train'] = input_batch

        logits['train'] = arch.inference(hypes, image_batch['train'], 'train')

        # Add to the Graph the Ops for loss calculation.
        loss = arch.loss(hypes, logits['train'], label_batch['train'])

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = solver.training(hypes, loss, global_step=global_step)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct['train'] = arch.evaluation(hypes, logits['train'],
                                                label_batch['train'])

        # Validation Cycle to the Graph
        with tf.name_scope('Validation') as scope:
            q['val'] = data_input.create_queues(hypes, 'val')
            input_batch = data_input.inputs(hypes, q, 'val',
                                            utils.cfg.data_dir)
            image_batch['val'], label_batch['val'] = input_batch

            tf.get_variable_scope().reuse_variables()

            logits['val'] = arch.inference(hypes, image_batch['val'], 'val')

            eval_correct['val'] = arch.evaluation(hypes, logits['val'],
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
                                               utils.cfg.data_dir)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(train_dir,
                                                graph_def=sess.graph_def)

        # And then after everything is built, start the training loop.
        solver = hypes['solver']
        start_time = time.time()
        for step in xrange(solver['max_steps']):

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.

            _, loss_value = sess.run([train_op, loss])

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                duration = (time.time() - start_time)/100
                examples_per_sec = solver['batch_size'] / duration
                sec_per_batch = float(duration)
                logging.info(
                 'Step %d: loss = %.2f ( %.3f sec (per Batch); %.1f examples/sec;)'%
                    (step, loss_value, sec_per_batch, examples_per_sec))
                # Update the events file.
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                start_time = time.time()

            # Save a checkpoint and evaluate the model periodically.
            if (step+1) % 1000 == 0 or (step + 1) == solver['max_steps']:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                start_time = time.time()
                # Evaluate against the training set.

            if (step+1) % 1000 == 0 or (step + 1) == solver['max_steps']:

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
    if FLAGS.config == "example_params.py":
        logging.info("Training on default config.")
        logging.info(
            "Use train.py --config=your_config.py to train different models")

    with open(tf.app.flags.FLAGS.config, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)

    train_dir = utils.get_train_dir()
    initialize_training_folder(hypes, train_dir)
    maybe_download_and_extract(hypes, train_dir)
    run_training(hypes, train_dir)


if __name__ == '__main__':
    tf.app.run()
