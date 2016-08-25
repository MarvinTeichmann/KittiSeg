"""Utils for TensorVision."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time

from datetime import datetime

import tensorflow as tf

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

if 'TV_SAVE' in os.environ and os.environ['TV_SAVE']:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug, '
                       'hence it will get overwritten by further runs.'))
else:
    tf.app.flags.DEFINE_boolean(
        'save', False, ('Whether to save the run. In case --nosave (default) '
                        'output will be saved to the folder TV_DIR_RUNS/debug '
                        'hence it will get overwritten by further runs.'))


flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

# usage: train.py --config=my_model_params.py
flags.DEFINE_string('hypes', None,
                    'File storing model parameters.')

flags.DEFINE_string('gpus', None,
                    ('Which gpus to use. For multiple GPUs use comma seperated'
                     'ids. [e.g. --gpus 0,3]'))


def set_dirs(hypes, hypes_fname):
    """Add directories into hypes."""
    if 'dirs' not in hypes:
        hypes['dirs'] = {}

    # Set base_path
    if 'base_path' not in hypes['dirs']:
        base_path = os.path.dirname(os.path.realpath(hypes_fname))
        hypes['dirs']['base_path'] = base_path
    else:
        base_path = hypes['dirs']['base_path']

    # Set output dir
    if 'output_dir' not in hypes['dirs']:
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(base_path, os.environ['TV_DIR_RUNS'])
        else:
            runs_dir = os.path.join(base_path, 'RUNS')

        if not FLAGS.save and FLAGS.name is None:
            output_dir = os.path.join(runs_dir, 'debug')
        else:
            json_name = hypes_fname.split('/')[-1].replace('.json', '')
            date = datetime.now().strftime('%Y_%m_%d_%H.%M')
            run_name = '%s_%s' % (json_name, date)
            if FLAGS.name is not None:
                run_name = run_name + "_" + FLAGS.name
            output_dir = os.path.join(runs_dir, run_name)

        hypes['dirs']['output_dir'] = output_dir

    # Set data dir
    if 'data_dir' not in hypes['dirs']:
        if 'TV_DIR_DATA' in os.environ:
            data_dir = os.path.join(base_path, os.environ['TV_DIR_DATA'])
        else:
            data_dir = os.path.join(base_path, 'DATA')

        hypes['dirs']['data_dir'] = data_dir

    return


# Add basic configuration
def cfg():
    """General configuration values."""
    return None


def _set_cfg_value(cfg_name, env_name, default, cfg):
    """Set a value for the configuration.

    Parameters
    ----------
    cfg_name : str
    env_name : str
    default : str
    cfg : function
    """
    if env_name in os.environ:
        setattr(cfg, cfg_name, os.environ[env_name])
    else:
        logging.info("No environment variable '%s' found. Set to '%s'.",
                     env_name,
                     default)
        setattr(cfg, cfg_name, default)


_set_cfg_value('plugin_dir',
               'TV_PLUGIN_DIR',
               os.path.expanduser("~/tv-plugins"),
               cfg)
_set_cfg_value('step_show', 'TV_STEP_SHOW', 100, cfg)
_set_cfg_value('step_eval', 'TV_STEP_EVAL', 100, cfg)
_set_cfg_value('step_str',
               'TV_STEP_STR',
               ('Step {step}/{total_steps}: loss = {loss_value:.2f}'
                'lr = {lr_value:.6f}'
                '( {sec_per_batch:.3f} sec; '
                '{examples_per_sec:.1f} examples/sec)'),
               cfg)


def load_plugins():
    """Load all TV plugins."""
    if os.path.isdir(cfg.plugin_dir):
        onlyfiles = [f for f in os.listdir(cfg.plugin_dir)
                     if os.path.isfile(os.path.join(cfg.plugin_dir, f))]
        pyfiles = [f for f in onlyfiles if f.endswith('.py')]
        import imp
        for pyfile in pyfiles:
            logging.info('Loaded plugin "%s".', pyfile)
            imp.load_source(os.path.splitext(os.path.basename(pyfile))[0],
                            pyfile)
