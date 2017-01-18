.. _installation:

============
Installation
============

TensorVision has a couple of prerequisites that need to be installed first, but
it is not very picky about versions.

Most of the instructions below assume you are running a Linux or Mac system,
but are otherwise very generic.

If you run into any trouble, please check the `TensorFlow installation instructions
<https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html>`_ which cover installing
the prerequisites for a range of operating systems, or ask for help as a GitHub
issue (https://github.com/TensorVision/TensorVision/issues).


Prerequisites
=============

Python + pip
------------

TensorVision currently requires Python 2.7 or 3.4 to run. Please install Python
via the package manager of your operating system if it is not included already.

Python includes ``pip`` for installing additional modules that are not shipped
with your operating system, or shipped in an old version, and we will make use
of it below. We recommend installing these modules into your home directory
via ``--user``, or into a `virtual environment
<http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/>`_
via ``virtualenv``.

C compiler
----------

Numpy/scipy require a compiler if you install them via ``pip``. On Linux, the
default compiler is usually ``gcc``, and on Mac OS, it's ``clang``. Again,
please install them via the package manager of your operating system.

numpy/scipy + BLAS
------------------

TensorVision requires numpy. Numpy/scipy rely on a BLAS library to provide fast
linear algebra routines. They will work fine without one, but a lot slower, so
it is worth getting this right (but this is less important if you plan to use a
GPU).

If you install numpy and scipy via your operating system's package manager,
they should link to the BLAS library installed in your system. If you install
numpy and scipy via ``pip install numpy`` and ``pip install scipy``, make sure
to have development headers for your BLAS library installed (e.g., the
``libopenblas-dev`` package on Debian/Ubuntu) while running the installation
command. Please refer to the `numpy/scipy build instructions
<http://www.scipy.org/scipylib/building/index.html>`_ if in doubt.


Stable TensorVision release
===========================

Currently, no stable version is available.


Bleeding-edge version
=====================

To install the latest version of TensorVision, run the following commands:

.. code-block:: bash

  pip install --upgrade https://github.com/TensorVision/TensorVision/archive/master.zip

Again, add ``--user`` if you want to install to your home directory instead.


.. _tensorvision-development-install:


Development installation
========================

Alternatively, you can install TensorVision from source, in a way that any
changes to your local copy of the source tree take effect without requiring a
reinstall. This is often referred to as *editable* or *development* mode.
Firstly, you will need to obtain a copy of the source tree:

.. code-block:: bash

  git clone https://github.com/TensorVision/TensorVision.git

It will be cloned to a subdirectory called ``TensorVision``. Make sure to place
it in some permanent location, as for an *editable* installation, Python will
import the module directly from this directory and not copy over the files.
Enter the directory and install the requirements:

.. code-block:: bash

  cd TensorVision
  pip install -r requirements.txt

You should also install the additional development requirements which can be
found in ``requirements-dev.txt``.

To install the TensorVision package itself, in editable mode, run:

.. code-block:: bash

  pip install --editable .

As always, add ``--user`` to install it to your home directory instead.

**Optional**: If you plan to contribute to TensorVision, you will need to fork
the TensorVision repository on GitHub. This will create a repository under your
user account. Update your local clone to refer to the official repository as
``upstream``, and your personal fork as ``origin``:

.. code-block:: bash

  git remote rename origin upstream
  git remote add origin https://github.com/<your-github-name>/TensorVision.git

If you set up an `SSH key <https://help.github.com/categories/ssh/>`_, use the
SSH clone URL instead: ``git@github.com:<your-github-name>/TensorVision.git``.

You can now use this installation to develop features and send us pull requests
on GitHub, see :doc:`development`!


You can run the tests by

.. code-block:: bash

  python setup.py test


GPU support
===========

Thanks to TensorFlow, TensorVision transparently supports training your
networks on a GPU, which may be 10 to 50 times faster than training them on a
CPU. Currently, this requires an NVIDIA GPU with CUDA support, and some
additional software for TensorFlow to use it.

CUDA
----

Install the latest CUDA Toolkit and possibly the corresponding driver available
from NVIDIA: https://developer.nvidia.com/cuda-downloads

Closely follow the *Getting Started Guide* linked underneath the download table
to be sure you don't mess up your system by installing conflicting drivers.

After installation, make sure ``/usr/local/cuda/bin`` is in your ``PATH``, so
``nvcc --version`` works. Also make sure ``/usr/local/cuda/lib64`` is in your
``LD_LIBRARY_PATH``, so the toolkit libraries can be found.

