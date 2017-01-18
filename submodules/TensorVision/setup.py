import os
import re
from setuptools import find_packages
from setuptools import setup
# We need io.open() (Python 3's default open) to specify file encodings
import io

here = os.path.abspath(os.path.dirname(__file__))
try:
    # obtain version string from __init__.py
    # Read ASCII file with builtin open() so __version__ is str in
    # Python 2 and 3
    with open(os.path.join(here, 'tensorvision', '__init__.py'), 'r') as f:
        init_py = f.read()
    version = re.search('__version__ = "(.*)"', init_py).groups()[0]
except Exception:
    version = ''
try:
    # obtain long description from README and CHANGES
    # Specify encoding to get a unicode type in Python 2 and a str in Python 3
    with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
        README = f.read()
    changes_file = os.path.join(here, 'CHANGES.md')
    with io.open(changes_file, 'r', encoding='utf-8') as f:
        CHANGES = f.read()
except IOError:
    README = CHANGES = ''

install_requires = ['numpy',
                    'scipy',
                    'tensorflow']

tests_require = ['mock',
                 'pytest',
                 'pytest-cov',
                 'pytest-pep8',
                 ]

scripts = ['bin/tv-train',
           'bin/tv-analyze',
           'bin/tv-continue']

setup(
    name="tensorvision",
    version=version,
    description="A library to build and train neural networks "
                "in with TensorFlow for Computer Vision",
    long_description="\n\n".join([README, CHANGES]),
    classifiers=["Development Status :: 1 - Planning",
                 "Intended Audience :: Developers",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: MIT License",
                 "Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3",
                 "Programming Language :: Python :: 3.4",
                 "Topic :: Scientific/Engineering :: Artificial Intelligence",
                 ],
    keywords="",
    scripts=scripts,
    author="Marvin Teichman, Martin Thoma",
    author_email="marvin.teichmann@googlemail.com, info@martin-thoma.de",
    url="https://github.com/MarvinTeichmann/TensorVision",
    license="MIT",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={'testing': tests_require, },
)
