#!/usr/bin/env python

import io
import os
import re
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def get_cuda_compile_archs(nvcc_flags=None):
    """Get the target CUDA architectures from CUDA_ARCH_LIST env variable"""
    if nvcc_flags is None:
        nvcc_flags = []

    CUDA_ARCH_LIST = os.getenv("CUDA_ARCH_LIST", None)
    if CUDA_ARCH_LIST is not None:
        for arch in CUDA_ARCH_LIST.split(";"):
            m = re.match(r"^([0-9.]+)(?:\(([0-9.]+)\))?(\+PTX)?$", arch)
            assert m, "Wrong architecture list: %s" % CUDA_ARCH_LIST
            cod_arch = m.group(1).replace(".", "")
            com_arch = m.group(2).replace(".", "") if m.group(2) else cod_arch
            ptx = True if m.group(3) else False
            nvcc_flags.extend(
                ["-gencode", "arch=compute_{},code=sm_{}".format(com_arch, cod_arch)]
            )
            if ptx:
                nvcc_flags.extend(
                    [
                        "-gencode",
                        "arch=compute_{},code=compute_{}".format(com_arch, cod_arch),
                    ]
                )

    return nvcc_flags


def get_long_description():
    fname = os.path.join(os.path.dirname(__file__), "README.md")
    with io.open(fname, "r") as f:
        return f.read()

def get_requirements():
    fname = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with io.open(fname, "r") as f:
        return f.readlines()


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
    headers += ["cpp/gpu.h", "cpp/factory.h", "cpp/generic.h"]
    sources += ["cpp/gpu/impl.cu"]
    Extension = CUDAExtension

    extra_compile_args["cxx"].append("-DWITH_CUDA")
    extra_compile_args["nvcc"].append("-DWITH_CUDA")
    extra_compile_args["nvcc"].extend(get_cuda_compile_archs())
else:
    Extension = CppExtension


description = "Functions to compute probabilistic relevance scores from PHOC embeddings"
long_description = get_long_description()
requirements = get_requirements()

setup(
    name="prob_phoc",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.2.0",
    url="https://github.com/jpuigcerver/prob_phoc",
    author="Joan Puigcerver",
    author_email="joapuipe@gmail.com",
    license="MIT",
    # Requirements
    setup_requires=requirements,
    install_requires=requirements,
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
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
