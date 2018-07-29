#!/usr/bin/env python

import os

from setuptools import setup, find_packages

this_file = os.path.dirname(__file__)

setup(
    name='prob-phoc',
    description='Compute probabilistic PHOC relevance scores efficiently',
    version='0.1',
    url='https://github.com/jpuigcerver/prob_phoc',
    author='Joan Puigcerver',
    author_email='joapuipe@gmail.com',
    license='MIT',
    # Requirements
    install_requires=[
        'cffi>=1.0.0',
        'numpy',
        'scipy',
        'torch==0.3.1',
    ],
    setup_requires=[
        'cffi>=1.0.0',
        'torch==0.3.1',
    ],
    # Exclude the build files.
    packages=find_packages(exclude=['build']),
    # Extensions to compile.
    ext_package='',
    cffi_modules=[
        os.path.join(this_file, 'build.py:ffi')
    ]
)
