# prob-phoc

[![Build Status](https://travis-ci.com/jpuigcerver/prob-phoc.svg?branch=master)](https://travis-ci.com/jpuigcerver/prob-phoc)

PyTorch functions to compute meaningful probabilistic relevance scores from
PHOC (Pyramid of Histograms of Characters) embeddings.
Although they are called Pyramid of Histograms of Characters, in practice
they are a Pyramid of Bag of Characters. At the end, each word is
represented by a high-dimensional binary vector.

See the [wiki](https://github.com/jpuigcerver/prob-phoc/wiki)
for additional details.

## Usage

The library provides two functions: `cphoc` and `pphoc`, which are
similar to SciPy's `cdist` and `pdist`:

Both functions can operate with PHOC embeddings in the probability space (where
each dimension is a real number in the range [0, 1]), or in the log-probability
space (where each dimension is the logarithm of a probability). These are also
sometimes refered to as the Real and Log semirings.

```python
import torch
from prob_phoc import cphoc, pphoc

x = torch.Tensor(...)
y = torch.Tensor(...)

# Compute the log-relevance scores between all pairs of rows in x, y.
# Note: x and y must have the PHOC log-probabilities.
logprob = cphoc(x, y)

# This is equivalent to:
logprob = cphoc(x, y, method="sum_prod_log")

# If your matrices have probabilities instead of log-probabilities, use:
prob = cphoc(x, y, method="sum_prob_real")

# Compute the log-relevance scores between all pairs of distinct rows in x.
# Note: The output is a vector with N * (N - 1) / 2 elements.
logprob = pphoc(x)
```

## Installation

The easiest way is to install the package from PyPI:

```bash
pip install prob-phoc
```

If you want to install the latest version from the repository, clone it
and use the setup.py script to compile and install the library.

```bash
python setup.py install
```

You will need a C++11 compiler (tested with GCC 4.9).
If you want to compile with CUDA support, you will also need to install
the CUDA Toolkit (tested with versions 8.0, 9.0 and 10.0)

## Tests and benchmarks

After the installation, you can run the tests to ensure that everything is
working fine.

```bash
python -m prob_phoc.test
```

I have also some benchmarks to compare CPU vs. CUDA, for different matrix
sizes and float precision. These take quite a long to run, so don't hold
your breath.

```bash
python -m prob_phoc.benchmark
```
