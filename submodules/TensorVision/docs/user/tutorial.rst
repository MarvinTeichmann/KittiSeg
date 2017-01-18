.. _tutorial:

========
Tutorial
========

This tutorial introduces the general workflow when using TensorVision.
Examples can be found in the `Modell Zoo`_ repository.

Basics
======

Train a model:


.. code-block:: bash

   tv-train --hypes config.json


Evaluate a model:


.. code-block:: bash

   python eval.py


Flags:

* ``--hypes=myconfig.json``
* ``--name=myname``


Workflow
========

Each time you get a new task


Create JSON file
----------------

Create a json file (e.g. `cifar10_cnn.json`). It has at least the following
content:

.. code-block:: json

   {
     "model": {
       "input_file": "examples/inputs/cifar10_input.py",
       "architecture_file" : "examples/networks/cifar_net.py",
       "objective_file" : "examples/objectives/softmax_classifier.py",
       "optimizer_file" : "examples/optimizer/exp_decay.py"
     }
   }


Adjust input file
-----------------

The ``input_file`` contains the path to a Python file. This Python file has to
have a function ``inputs(hypes, q, phase, data_dir)``.

The parameters of `inputs` are:

* ``hypes``: A dictionary which contains everything your ``model.json`` file
  had.
* ``q``: A queue (e.g. `FIFOQueue`_)
* ``phase``: Either ``train`` or ``val``
* ``data_dir``: Path to the data. This can be set with ``TV_DIR_DATA``.

The expected return value is a tuple (xs, ys), where x is a list of features
and y is a list of labels.


Adjust architecture file
------------------------

The ``architecture_file`` contains the architecture of the network. It has to
have the function ``inference(hypes, images, train=True)``, which takes image
tensors creates a computation graph to produce logits


Adjust objective file
---------------------

The ``objective_file`` contains task spezific code od the model. It
has to implement the following functions:

* ``decoder(hypes, images, train=True)``
* ``loss(hypes, decoder, labels)``
* ``evaluation(hypes, decoder, labels)``


Adjust the solver file
----------------------

The ``optimizer_file`` contains the path to a Python file. This Python file has
to have a function ``training(H, loss, global_step, learning_rate)``. It defines how one tries to find a minimum of the loss function. Additionally it should provide a function ``get_learning_rate(hype, global_step)``, which returns the current learning rate at each step.



.. _Modell Zoo: https://github.com/TensorVision/modell_zoo
.. _FIFOQueue : https://www.tensorflow.org/versions/r0.8/how_tos/threading_and_queues/index.html


Scripts
=======

TensorVision brings some scripts which you can use:

* ``tv-train``: Trains, evaluates and saves the model network using a queue.
* ``tv-continue``: Continues training of a model from logdir.
* ``tv-analyze``: Evaluates the model.
* ``tv-maskstats``: Get statistics about the distribution of classes in the masks.


Hypes file
==========

TensorVision makes use of a configuration file for each project. As it was
originally intended to have hyperparameters of models, it is commonly called
"hypes file" or ``hypes.json`` throughout this project.

This configuration file allows you to adjust given networks easily to new
problem domains.


data
----

It is recommended to create one json file for the training data sources as
well as one for the testing data sources. Each file is a list of dictionaries,
where each dictionary has the keys ``raw`` and ``mask``. For example, your
``trainfiles.json`` could look like this:

.. code-block:: json

    [
        {
            "raw": "/home/moose/GitHub/MediSeg/DATA/OP1/img_00.png",
            "mask": "/home/moose/GitHub/MediSeg/DATA/OP1/img_00_GT.png"
        },
        {
            "raw": "/home/moose/GitHub/MediSeg/DATA/OP1/img_01.png",
            "mask": "/home/moose/GitHub/MediSeg/DATA/OP1/img_01_GT.png"
        },
        {
            "raw": "/home/moose/GitHub/MediSeg/DATA/OP1/img_02.png",
            "mask": "/home/moose/GitHub/MediSeg/DATA/OP1/img_02_GT.png"
        }
    ]

You should add the path of those files to your ``hypes.json``:


.. code-block:: json

    "data": {
      "train": "../../DATA/trainfiles.json",
      "test": "../../DATA/testfiles.json"
    },

While this is not required, it will allow you to use ``tv-maskstats`` and make
your code more readable and easier to adjust.


classes
-------

It is recommended to add a description of your labeled data to your
hyperparameters file. This makes your code more readable and gives the
possibility to use ``tv-maskstats`` as well as :func:`tensorvision.utils.load_segmentation_mask`.
The ``classes`` block looks like this:

.. code-block:: json

    "classes": [
        {"name": "background",
         "colors": ["#000000"],
         "output": "#ff000000"},
        {"name": "instrument",
         "colors": ["#464646", "#a0a0a0", "#ffffff", "default"],
         "output": "#00ff007f"}
     ]