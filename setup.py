#!/usr/bin/env python

import os
import re
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

extra_compile_args = {
    "cxx": ["-std=c++11", "-O3", "-fopenmp"],
    "nvcc": ["-std=c++11", "-O3"],
}

CC = os.getenv("CC", None)
if CC is not None:
    extra_compile_args["nvcc"].append("-ccbin=" + CC)

include_dirs = [os.path.dirname(os.path.realpath(__file__)) + "/cpp"]

headers = ["cpp/common.h", "cpp/cpu.h", "cpp/factory.h", "cpp/generic.h"]
sources = ["cpp/binding.cc"]

if torch.cuda.is_available():
    Extension = CppExtension
else:
    Extension = CppExtension


setup(
    name="prob_phoc",
    description="Probabilistic PHOC relevance scores",
    version="0.2.0",
    url="https://github.com/jpuigcerver/prob_phoc",
    author="Joan Puigcerver",
    author_email="joapuipe@gmail.com",
    license="MIT",
    # Requirements
    setup_requires=["pybind11", "torch>=1.0.0"],
    install_requires=["numpy", "pybind11", "torch>=1.0.0"],
    packages=find_packages(),
    ext_modules=[
        Extension(
            name="prob_phoc._C",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
