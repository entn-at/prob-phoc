from setuptools import setup, find_packages
from torch.utils.ffi import create_extension

phoc_ffi = create_extension(
    name='prob_phoc._phoc',
    language='cc',
    headers=['cpp/phoc.h'],
    sources=['cpp/phoc.cc'],
    include_dirs=['cpp'],
    package=True,
    relative_to='prob_phoc',
    extra_compile_args=['-O3', '-march=native', '-fopenmp',
                        '-std=c++11']).distutils_extension()

setup(name='prob_phoc', version='0.1', packages=find_packages(),
      scripts=[],
      install_requires=['numpy', 'scipy', 'torch'],
      author='Joan Puigcerver',
      author_email='joapuipe@gmail.com',
      license='MIT',
      url='https://github.com/jpuigcerver/PyLaia',
      ext_modules=[phoc_ffi])
