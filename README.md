# prob-phoc

[![Build Status](https://travis-ci.com/jpuigcerver/prob-phoc.svg?branch=master)](https://travis-ci.com/jpuigcerver/prob-phoc)

PyTorch functions to compute meaningful probabilistic relevance scores from
PHOC (Pyramid of Histograms of Characters) embeddings.

PHOC embeddings, originally proposed by [Almazan](https://github.com/almazan/watts),
have been widely used for isolated word recognition and keyword spotting (KWS).
Several similarity/dissimilarity measures have been proposed to perform recognition
or rank the candidate word images for a given query, to perform KWS.
In addition, the models that extract the PHOC embedding for a given image have also
improved in the last years.

In particular, a VGG-like architecture and the Bray-Curtis measure were proposed
by [Sudholt](https://github.com/ssudholt/phocnet) to extract the PHOC embedding
and rank the candidate images in KWS.

Since the PHOC embeddings can be interpreted probabilistically, in my PhD
thesis I proposed a way of computing the ``relevance probability'' for a
given pair of PHOC embeddings, assuming that a pair is relevant if the two
images render the same word (which is the typical definition of ``relevance''
in KWS settings).

Let _h(x)_ and _h(y)_ be the predicted PHOC from two images, _x_ and _y_.
The components of the PHOC vector can be interpreted probabilistically as:

_h(x)_ = P(H_1 = 1 | x), P(H_2 = 1 | x), ..., P(H_D = 1 | x)

Assuming that the two images are independent, and that each dimension of the PHOC
is independent of the others (which is not true, but it is actually assumed turing
the training of the PHOCNet), it can be proven that the probability that two images
render the same text (PHOC embedding) is:

```math
\sum_{h_1, ..., h_D} P(H_1 = h_1, ..., H_D = h_d | x) \cdot P(H_1 = h_1, ..., H_D = h_d | y)
```

Because the dimensions are independent, there is an efficient way of
computing this sum in _O(D)_.

## Installation

Simply use the `setup.py` script to compile and install the library.
You will need a C++11 compiler.

```bash
python setup.py install
```

After the installation, you can run the tests to ensure that everything is
working fine.

```bash
python -m prob_phoc.test
```

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
