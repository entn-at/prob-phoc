# prob-phoc

[![Build Status](https://travis-ci.com/jpuigcerver/prob-phoc.svg?branch=master)](https://travis-ci.com/jpuigcerver/prob-phoc)

Pytorch functions to compute the relevance scores for KWS from a probabilistic 
interpretation of the PHOC.

Let _h(x)_ and _h(y)_ be the predicted PHOC from two images, _x_ and _y_.
The components of the PHOC vector can be interpreted probabilistically as:
 
_h(x)_ = P(H_1 = 1 | x), P(H_2 = 1 | x), ..., P(H_D = 1 | x) 
 
The probability that two images render the same text (PHOC), assuming that each
dimension of the PHOC is independent of the others, is:

```math
\sum_{h_1, ..., h_D} P(H_1 = h_1, ..., H_D = h_d | x) \cdot P(H_1 = h_1, ..., H_D = h_d | y)
```

Because the dimensions are independent, there is an efficient way of 
computing this sum in _O(D)_.


## Install

Simply use the `setup.py` script to compile and install the library. 
You will need a C++11 compiler.

```bash
python setup.py install
```

If pip or distutils try to use a C compiler you will get errors during the
compilation. Please, set the environment variable `CC` to a C++11 compiler.

```bash
CC=g++-4.9 python setup.py install
```

After the installation, you can run the tests to ensure that everything is
working fine.

```bash
python -m prob_phoc.test
``` 

## Usage

The library provides two functions: `cphoc` and `pphoc`, which are 
similar to SciPy's `cdist` and `pdist`:

```python 
import torch
from prob_phoc import cphoc, pphoc

x = torch.Tensor(...)
y = torch.Tensor(...)

# Compute the log-relevance scores between all pairs of rows in x, y.
# Note: x and y must be matrices with the same number of columns.
# The output is a N x M matrix.
logprob = cphoc(x, y)

# Compute the log-relevance scores between all pairs of rows in x.
# Note: The output is a vector with N * (N + 1) / 2 elements.
logprob = pphoc(x)
```
