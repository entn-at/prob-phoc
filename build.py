import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['cpp/phoc.cc']
headers = ['cpp/phoc.h']
include_dirs = ['cpp']
defines = []
with_cuda = False

"""
if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['cpp/cuda/phoc.h']
    headers += ['cpp/cuda/phoc.cc']
    defines += [('WITH_CUDA', None)]
    with_cuda = True
"""

ffi = create_extension(
    'prob_phoc._ext',
    language='cc',
    package=True,
    headers=headers,
    sources=sources,
    include_dirs=include_dirs,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_compile_args=[
        '-O3',
        '-march=native',
        '-fopenmp',
        '-std=c++11'
    ])


if __name__ == '__main__':
    ffi.build()
