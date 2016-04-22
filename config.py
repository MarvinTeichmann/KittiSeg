# Dictionaries
import tensorflow as tf
import os
flags = tf.app.flags
FLAGS = flags.FLAGS

import logging

flags.DEFINE_string('gpus', None,
                    'Folder where Data will be stored.')

if FLAGS.gpus == None:
    logging.error("Please spezify a GPU to run on.")
    logging.error("Usage: train.py --gpus <IDS>")
    exit(1)

print("GPUs are set to: %s", FLAGS.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

default_name = "trash"
data_dir = "DATA"
model_dir = "output"
default_config = "hypes/kitti.json"