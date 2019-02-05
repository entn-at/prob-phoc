#!/usr/bin/env python

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
