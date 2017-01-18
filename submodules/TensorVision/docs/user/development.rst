Development
===========

The TensorVision project was started by Marvin Teichmann and Martin Thoma in
February 2016.

As an open-source project by researchers for researchers, we highly welcome
contributions! Every bit helps and will be credited.



What to contribute
------------------

Give feedback
~~~~~~~~~~~~~

To send us general feedback, questions or ideas for improvement.

If you have a very concrete feature proposal, add it to the `issue tracker on
GitHub`_:

* Explain how it would work, and link to a scientific paper if applicable.
* Keep the scope as narrow as possible, to make it easier to implement.


Report bugs
~~~~~~~~~~~

Report bugs at the `issue tracker on GitHub`_.
If you are reporting a bug, please include:

* your TensorVision and TensorFlow version.
* steps to reproduce the bug, ideally reduced to a few Python commands.
* the results you obtain, and the results you expected instead.


Fix bugs
~~~~~~~~

Look through the GitHub issues for bug reports. Anything tagged with "bug" is
open to whoever wants to implement it. If you discover a bug in TensorVision
you can fix yourself, by all means feel free to just implement a fix and not
report it first.


Implement features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for feature proposals. Anything tagged with
"feature" or "enhancement" is open to whoever wants to implement it. If you
have a feature in mind you want to implement yourself, please note that we
cannot guarantee upfront that your code will be included. Please do not
hesitate to just propose your idea in a GitHub issue first, so we can discuss
it and/or guide you through the implementation.


Write documentation
~~~~~~~~~~~~~~~~~~~

Whenever you find something not explained well, misleading, glossed over or
just wrong, please update it! The *Edit on GitHub* link on the top right of
every documentation page and the *[source]* link for every documented entity
in the API reference will help you to quickly locate the origin of any text.



How to contribute
-----------------

Edit on GitHub
~~~~~~~~~~~~~~

As a very easy way of just fixing issues in the documentation, use the *Edit
on GitHub* link on the top right of a documentation page or the *[source]* link
of an entity in the API reference to open the corresponding source file in
GitHub, then click the *Edit this file* link to edit the file in your browser
and send us a Pull Request. All you need for this is a free GitHub account.

For any more substantial changes, please follow the steps below to setup
TensorVision for development.


Development setup
~~~~~~~~~~~~~~~~~

First, follow the instructions for performing a development installation of
TensorVision (including forking on GitHub):
:ref:`tensorvision-development-install`

To be able to run the tests and build the documentation locally, install
additional requirements with: ``pip install -r requirements-dev.txt`` (adding
``--user`` if you want to install to your home directory instead).

If you use the bleeding-edge version of TensorFlow, then instead of running that
command, just use ``pip install`` to manually install all dependencies listed
in ``requirements-dev.txt`` with their correct versions; otherwise it will
attempt to downgrade TensorFlow to the known good version in ``requirements.txt``.


Documentation
~~~~~~~~~~~~~

The documentation is generated with `Sphinx
<http://sphinx-doc.org/latest/index.html>`_. To build it locally, run the
following commands:

.. code:: bash

    cd docs
    make html

Afterwards, open ``docs/_build/html/index.html`` to view the documentation as
it would appear on `readthedocs <http://tensorvision.readthedocs.org/>`_. If you
changed a lot and seem to get misleading error messages or warnings, run
``make clean html`` to force Sphinx to recreate all files from scratch.

When writing docstrings, follow existing documentation as much as possible to
ensure consistency throughout the library. For additional information on the
syntax and conventions used, please refer to the following documents:

* `reStructuredText Primer <http://sphinx-doc.org/rest.html>`_
* `Sphinx reST markup constructs <http://sphinx-doc.org/markup/index.html>`_
* `A Guide to NumPy/SciPy Documentation <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_


Testing
~~~~~~~

Tensorvision wants to achieve a code coverage of 100%, which creates some duties:

* Whenever you change any code, you should test whether it breaks existing
  features by just running the test suite. The test suite will also be run by
  `Travis <https://travis-ci.org/>`_ for any Pull Request to TensorVision.
* Any code you add needs to be accompanied by tests ensuring that nobody else
  breaks it in future. `Coveralls <https://coveralls.io/>`_ will check whether
  the code coverage stays at 100% for any Pull Request to TensorVision.
* Every bug you fix indicates a missing test case, so a proposed bug fix should
  come with a new test that fails without your fix.

To run the full test suite, just do

.. code:: bash

    py.test

Testing will end with a code
coverage report specifying which code lines are not covered by tests, if any.
Furthermore, it will list any failed tests, and failed `PEP8
<https://www.python.org/dev/peps/pep-0008/>`_ checks.

Finally, for a loop-on-failing mode, do ``pip install pytest-xdist`` and run
``py.test -f``. This will pause after the run, wait for any source file to
change and run all previously failing tests again.

Before commiting any change, you should run

.. code:: bash

    tv-train --hypes examples/cifar10_minimal.json
    tv-analyze --hypes examples/cifar10_minimal.json --logdir examples/RUNS/debug/

to see if everything still works as expected.


Sending Pull Requests
~~~~~~~~~~~~~~~~~~~~~

When you're satisfied with your addition, the tests pass and the documentation
looks good without any markup errors, commit your changes to a new branch, push
that branch to your fork and send us a Pull Request via GitHub's web interface.

All these steps are nicely explained on GitHub:
https://guides.github.com/introduction/flow/

When filing your Pull Request, please include a description of what it does, to
help us reviewing it. If it is fixing an open issue, say, issue #123, add
*Fixes #123*, *Resolves #123* or *Closes #123* to the description text, so
GitHub will close it when your request is merged.



.. _issue tracker on GitHub: https://github.com/TensorVision/TensorVision/issues