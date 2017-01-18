.. configuration:

=============
Configuration
=============

TensorVision comes with reasonable defaults. You only need to read this if you
want tweak it to your needs.

TensorVision is configured with environment variables. It is quite easy to
set them yourself (see `multiple ways`_).
The supported variables are:

* ``TV_DIR_DATA``: The default directory where to look for data.
* ``TV_DIR_RUNS``: The default directory where to look for the model.
* ``TV_IS_DEV``: Either 0 or 1 - set if you want to see debug messages.
* ``TV_PLUGIN_DIR``: Directory with Python scripts which will be loaded by utils.py
* ``TV_SAVE``: Whether to keep all runs on default. By default runs will be written to ``TV_DIR_RUNS/debug`` and overwritten by newer runs, unless ``tv-train --save`` is called.
* ``TV_USE_GPUS``: Controll which gpus to use. Default all GPUs are used, GPUs
  can be specified using ``--gpus``. Setting ``TV_USE_GPUS='force'`` makes the
  flag ``--gpus`` compulsory, this is useful in cluster environoments. Use
  ``TV_USE_GPUS='0,3'`` to run Tensorflow an the GPUs with ids 0 and 3.
* ``TV_STEP_SHOW``: After how many epochs of training should the ``TV_STEP_STR`` be printed?
* ``TV_STEP_EVAL``: After how many epochs of training evaluation is done.
* ``TV_STEP_WRITE``: After how many epochs of training checkpoints are written to disk.
* ``TV_MAX_KEEP``: How many checkpoints to keep.
* ``TV_STEP_STR``: Set what you want to see each 100th step of the training.
  The default is


.. code-block:: python

   Step {step}/{total_steps}: loss = {loss_value:.2f}
   ( {sec_per_batch:.3f} sec (per Batch);
   {examples_per_sec:.1f} examples/sec;)


.. _multiple ways: http://unix.stackexchange.com/a/117470/4784