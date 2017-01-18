"""Utility functions for TensorVision."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
import json
import logging
import os

from datetime import datetime
import matplotlib.cm as cm

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy.misc
import sys
import struct

import tensorflow as tf

from six.moves import urllib

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('gpus', None,
                    ('Which gpus to use. For multiple GPUs use comma seperated'
                     'ids. [e.g. --gpus 0,3]'))


def print_eval_dict(eval_dict, prefix=''):
    for name, value in eval_dict:
            logging.info('    %s %s : % 0.04f ' % (name, prefix, value))
    return


def download(url, dest_directory):
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    logging.info("Download URL: {}".format(url))
    logging.info("Download DIR: {}".format(dest_directory))

    def _progress(count, block_size, total_size):
                prog = float(count * block_size) / float(total_size) * 100.0
                sys.stdout.write('\r>> Downloading %s %.1f%%' %
                                 (filename, prog))
                sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath,
                                             reporthook=_progress)
    print()
    return


def set_dirs(hypes, hypes_fname):
    """
    Add directories to hypes.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    hypes_fname : str
        Path to hypes_file
    """
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
            runs_dir = os.path.join(base_path, '../RUNS')

        # test for project dir
        if hasattr(FLAGS, 'project') and FLAGS.project is not None:
            runs_dir = os.path.join(runs_dir, FLAGS.project)

        if not FLAGS.save and FLAGS.name is None:
            output_dir = os.path.join(runs_dir, 'debug')
        else:
            json_name = hypes_fname.split('/')[-1].replace('.json', '')
            date = datetime.now().strftime('%Y_%m_%d_%H.%M')
            if FLAGS.name is not None:
                json_name = FLAGS.name + "_" + json_name
            run_name = '%s_%s' % (json_name, date)
            output_dir = os.path.join(runs_dir, run_name)

        hypes['dirs']['output_dir'] = output_dir

    # Set data dir
    if 'data_dir' not in hypes['dirs']:
        if 'TV_DIR_DATA' in os.environ:
            data_dir = os.path.join(base_path, os.environ['TV_DIR_DATA'])
        else:
            data_dir = os.path.join(base_path, '../DATA')

        hypes['dirs']['data_dir'] = data_dir

    _add_paths_to_sys(hypes)

    return


def set_gpus_to_use():
    """Set the gpus to use."""
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


def load_modules_from_hypes(hypes, postfix=""):
    """Load all modules from the files specified in hypes.

    Namely the modules loaded are:
    input_file, architecture_file, objective_file, optimizer_file

    Parameters
    ----------
    hypes : dict
        Hyperparameters

    Returns
    -------
    hypes, data_input, arch, objective, solver
    """
    modules = {}
    base_path = hypes['dirs']['base_path']

    # _add_paths_to_sys(hypes)
    f = os.path.join(base_path, hypes['model']['input_file'])
    data_input = imp.load_source("input_%s" % postfix, f)
    modules['input'] = data_input

    f = os.path.join(base_path, hypes['model']['architecture_file'])
    arch = imp.load_source("arch_%s" % postfix, f)
    modules['arch'] = arch

    f = os.path.join(base_path, hypes['model']['objective_file'])
    objective = imp.load_source("objective_%s" % postfix, f)
    modules['objective'] = objective

    f = os.path.join(base_path, hypes['model']['optimizer_file'])
    solver = imp.load_source("solver_%s" % postfix, f)
    modules['solver'] = solver

    f = os.path.join(base_path, hypes['model']['evaluator_file'])
    eva = imp.load_source("evaluator_%s" % postfix, f)
    modules['eval'] = eva

    return modules


def _add_paths_to_sys(hypes):
    """
    Add all module dirs to syspath.

    This adds the dirname of all modules to path.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    """
    base_path = hypes['dirs']['base_path']
    if 'path' in hypes:
            for path in hypes['path']:
                path = os.path.realpath(os.path.join(base_path, path))
                sys.path.insert(1, path)
    return


def load_modules_from_logdir(logdir, dirname="model_files", postfix=""):
    """Load hypes from the logdir.

    Namely the modules loaded are:
    input_file, architecture_file, objective_file, optimizer_file

    Parameters
    ----------
    logdir : string
        Path to logdir

    Returns
    -------
    data_input, arch, objective, solver
    """
    model_dir = os.path.join(logdir, dirname)
    f = os.path.join(model_dir, "data_input.py")
    # TODO: create warning if file f does not exists
    data_input = imp.load_source("input_%s" % postfix, f)
    f = os.path.join(model_dir, "architecture.py")
    arch = imp.load_source("arch_%s" % postfix, f)
    f = os.path.join(model_dir, "objective.py")
    objective = imp.load_source("objective_%s" % postfix, f)
    f = os.path.join(model_dir, "solver.py")
    solver = imp.load_source("solver_%s" % postfix, f)

    f = os.path.join(model_dir, "eval.py")
    eva = imp.load_source("evaluator_%s" % postfix, f)
    modules = {}
    modules['input'] = data_input
    modules['arch'] = arch
    modules['objective'] = objective
    modules['solver'] = solver
    modules['eval'] = eva

    return modules


def load_hypes_from_logdir(logdir, subdir="model_files"):
    """Load hypes from the logdir.

    Namely the modules loaded are:
    input_file, architecture_file, objective_file, optimizer_file

    Parameters
    ----------
    logdir : string
        Path to logdir

    Returns
    -------
    hypes
    """
    model_dir = os.path.join(logdir, subdir)
    hypes_fname = os.path.join(model_dir, "hypes.json")
    with open(hypes_fname, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)
    _add_paths_to_sys(hypes)
    hypes['dirs']['base_path'] = os.path.realpath(logdir)
    hypes['dirs']['output_dir'] = os.path.realpath(logdir)
    hypes['dirs']['image_dir'] = os.path.join(hypes['dirs']['output_dir'],
                                              'images')

    return hypes


def create_filewrite_handler(logging_file, mode='w'):
    """
    Create a filewriter handler.

    A copy of the output will be written to logging_file.

    Parameters
    ----------
    logging_file : string
        File to log output

    Returns
    -------
    The filewriter handler
    """
    target_dir = os.path.dirname(logging_file)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    filewriter = logging.FileHandler(logging_file, mode=mode)
    formatter = logging.Formatter(
        '%(asctime)s %(name)-3s %(levelname)-3s %(message)s')
    filewriter.setLevel(logging.INFO)
    filewriter.setFormatter(formatter)
    logging.getLogger('').addHandler(filewriter)
    return filewriter


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
_set_cfg_value('step_show', 'TV_STEP_SHOW', 50, cfg)
_set_cfg_value('step_eval', 'TV_STEP_EVAL', 250, cfg)
_set_cfg_value('step_write', 'TV_STEP_WRITE', 1000, cfg)
_set_cfg_value('max_to_keep', 'TV_MAX_KEEP', 10, cfg)
_set_cfg_value('step_str',
               'TV_STEP_STR',
               ('Step {step}/{total_steps}: loss = {loss_value:.2f}; '
                'lr = {lr_value:.2e}; '
                '{sec_per_batch:.3f} sec (per Batch); '
                '{examples_per_sec:.1f} imgs/sec'),
               cfg)


def load_plugins():
    """Load all TensorVision plugins."""
    if os.path.isdir(cfg.plugin_dir):
        onlyfiles = [f for f in os.listdir(cfg.plugin_dir)
                     if os.path.isfile(os.path.join(cfg.plugin_dir, f))]
        pyfiles = [f for f in onlyfiles if f.endswith('.py')]
        import imp
        for pyfile in pyfiles:
            logging.info('Loaded plugin "%s".', pyfile)
            imp.load_source(os.path.splitext(os.path.basename(pyfile))[0],
                            pyfile)


def load_labeled_files_json(json_datafile_path):
    """
    Load a JSON file which contains a list of {'raw': 'xy', 'mask': 'z'}.

    Parameters
    ----------
    json_datafile_path : str
        Path to a JSON file which contains a list of labeled images.

    Returns
    -------
    list of dictionaries
    """
    with open(json_datafile_path) as data_file:
        data = json.load(data_file)
    base_path = os.path.dirname(os.path.realpath(json_datafile_path))
    for i in range(len(data)):
        if not os.path.isabs(data[i]['raw']):
            data[i]['raw'] = os.path.realpath(os.path.join(base_path,
                                                           data[i]['raw']))
            if not os.path.isfile(data[i]['raw']):
                logging.warning("'%s' does not exist.", data[i]['raw'])
        if not os.path.isabs(data[i]['mask']):
            data[i]['mask'] = os.path.realpath(os.path.join(base_path,
                                                            data[i]['mask']))
            if not os.path.isfile(data[i]['mask']):
                logging.warning("'%s' does not exist.", data[i]['mask'])
    return data


def overlay_segmentation(input_image, segmentation, color_dict):
    """
    Overlay input_image with a hard segmentation result.

    Store the result with the same name as segmentation_image, but with
    `-overlay`.

    Parameters
    ----------
    input_image : numpy.array
        An image of shape [width, height, 3].
    segmentation : numpy.array
        Segmentation of shape [width, height].
    color_changes : dict
        The key is the class and the value is the color which will be used in
        the overlay. Each color has to be a tuple (r, g, b, a) with
        r, g, b, a in {0, 1, ..., 255}.
        It is recommended to choose a = 0 for (invisible) background and
        a = 127 for all other classes.

    Returns
    -------
    numpy.array
        The image overlayed with the segmenation
    """
    width, height = segmentation.shape
    output = scipy.misc.toimage(segmentation)
    output = output.convert('RGBA')
    for x in range(0, width):
        for y in range(0, height):
            if segmentation[x, y] in color_dict:
                output.putpixel((y, x), color_dict[segmentation[x, y]])
            elif 'default' in color_dict:
                output.putpixel((y, x), color_dict['default'])

    background = scipy.misc.toimage(input_image)
    background.paste(output, box=None, mask=output)

    return np.array(background)


def fast_overlay(input_image, segmentation, color=[0, 255, 0, 127]):
    """
    Overlay input_image with a hard segmentation result for two classes.

    Store the result with the same name as segmentation_image, but with
    `-overlay`.

    Parameters
    ----------
    input_image : numpy.array
        An image of shape [width, height, 3].
    segmentation : numpy.array
        Segmentation of shape [width, height].
    color: color for forground class

    Returns
    -------
    numpy.array
        The image overlayed with the segmenation
    """
    color = np.array(color).reshape(1, 4)
    shape = input_image.shape
    segmentation = segmentation.reshape(shape[0], shape[1], 1)

    output = np.dot(segmentation, color)
    output = scipy.misc.toimage(output, mode="RGBA")

    background = scipy.misc.toimage(input_image)
    background.paste(output, box=None, mask=output)

    return np.array(background)


def soft_overlay_segmentation(input_image,
                              seg_probability,
                              colormap=None,
                              alpha=0.4):
    """
    Overlay image with propability map.

    Overlays the image with a colormap ranging
    from blue to red according to the probability map
    given in gt_prob. This is good to analyse the segmentation
    result of a single class.

    Parameters
    ----------
    input_image : numpy.array
        Image of shape [width, height, 3]
    seg_probability : numpy.array
        Propability map for one class with shape [width, height]
    colormap : matplotlib colormap object
        Defines which floats get which color
    alpha : float
        How strong is the overlay compared to the input image


    Returns
    -------
    numpy.array
        Soft overlay of the input image with a propability map of shape
        [width, height, 3]

    Notes
    -----
    See `Matplotlib reference
    <http://matplotlib.org/examples/color/colormaps_reference.html>`_
    for more colormaps.
    """
    assert alpha >= 0.0
    assert alpha <= 1.0
    if colormap is None:
        colormap = cm.get_cmap('bwr')

    overimage = colormap(seg_probability, bytes=True)
    output = alpha * overimage[:, :, 0:3] + (1.0 - alpha) * input_image

    return output


def get_color2class(hypes):
    """
    Load dictionary which maps colors to classes as well as the default class.

    The classes are integers with values which range from 0 to N-1, where N is
    the total number of classes.

    This requires hypes to have an entry "classes". This entry has to be a list
    of dictionaries with key `colors`. This key is a list of HTML color strings
    in RGB format.

    ```
    "classes": [
      {"name": "road",
       "colors": ["#ff0000", "#ff1000"],
       "output": "#00ff007f"},
      {"name": "background",
       "colors": ["default", "#ff0000"],
       "output": "#ff00007f"},
      {"name": "ignore",
       "colors": ["#000000"]}
    ],
    ```

    The string `default` in the color list may only be in one class. If there
    are colors which are not mapped to any other class, the class with
    "default" gets assigned.

    The index of the dictionary in the list is the value of the integer matrix
    which is returned.

    Parameters
    ----------
    hypes : dict
        Hyperparameters

    Returns
    -------
    tuple
        (color2class_dict, default_class) where default_class can be None.
        The dictionary `color2class_dict` maps (R, G, B) tuples to class labels
        (ints).
    """
    color2class_dict = {}
    default_class = None
    for i, cl in enumerate(hypes['classes']):
        for color in cl['colors']:
            if color == 'default':
                if default_class is not None:
                    raise Exception(("The 'default' color was assigned to "
                                     "class '%s' and to class '%s'.") %
                                    (hypes['classes'][default_class]['name'],
                                     hypes['classes'][i]['name']))
                default_class = i
            else:
                if isinstance(color, basestring):
                    if not color.startswith('#'):
                        logging.error("Colors have to start with '#'. "
                                      "It was '%s'." % color)
                        raise Exception("Wrong color code.")
                    else:
                        color = color[1:]
                        color = struct.unpack('BBB', color.decode('hex'))
                if isinstance(color, list):
                    color = tuple(color)
                if color in color2class_dict:
                    raise Exception(("The color '%s' was assigned multiple "
                                     "times.") % str(color))
                color2class_dict[color] = i
    return color2class_dict, default_class


def load_segmentation_mask(hypes, gt_image_path):
    """
    Load a segmentation mask from an image.

    The mask is an integer array with shape (height, width). The integer values
    range from 0 to N-1, where N is the total number of classes.

    This requires hypes to have an entry 'classes'. This entry has to be a list
    of dictionaries with key `colors`. This key is a list of HTML color strings
    in RGB format.

    ```
    "classes": [
      {"name": "road",
       "colors": ["#ff0000", "#ff1000"],
       "output": "#00ff007f"},
      {"name": "background",
       "colors": ["default", "#ff0000"],
       "output": "#ff00007f"},
      {"name": "ignore",
       "colors": ["#000000"]}
    ],
    ```

    The string `default` in the color list may only be in one class. If there
    are colors which are not mapped to any other class, the class with
    "default" gets assigned.

    The index of the dictionary in the list is the value of the integer matrix
    which is returned.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    gt_image_path : str
        Path to an image file.

    Returns
    -------
    numpy array
        The ground truth mask.

    Note
    ----
    This function always loads the ground truth image in RGB mode. If the image
    is not in RGB mode it will get converted to RGB. This is important to know
    for the colors in hypes['classes'].
    """
    img = scipy.misc.imread(gt_image_path, mode='RGB')

    # map colors to classes
    color2class_dict, default_class = get_color2class(hypes)

    # Create gt image which is a matrix of classes
    gt = np.zeros((img.shape[0], img.shape[1]), dtype=int)  # only one channel
    assigned = np.zeros((img.shape[0], img.shape[1]), dtype=int)
    for color, class_label in color2class_dict.items():
        affected_pixels = np.all(img == color, axis=2)
        gt += affected_pixels * class_label
        assigned += affected_pixels
    remaining_pixels = np.logical_not(assigned)
    if np.any(remaining_pixels):
        if default_class is None:
            print("[ERROR] Some pixels did not get assigned a class. ")
            print("No 'default' class was assigned either.")
            print("The pixel colors are:")
            for i, row in enumerate(img):
                for j, pixel in enumerate(row):
                    pixel_color = tuple(pixel)
                    if remaining_pixels[i][j]:
                        print("  %s" % str(pixel_color))
            sys.exit(-1)
        else:
            gt += remaining_pixels * default_class
    return gt
