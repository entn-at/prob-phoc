from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch

import prob_phoc._C as _C


def _convert_to_tensor_if_needed(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif torch.is_tensor(x):
        return x
    else:
        raise ValueError("The argument cannot be converted to a PyTorch tensor")


def cphoc(xa, xb, y=None, method="sum_prod_log"):
    """Computes probabilistic PHOC relevance scores between each pair of inputs.
    """
    xa = _convert_to_tensor_if_needed(xa)
    xb = _convert_to_tensor_if_needed(xb)
    if y is None:
        y = xa.new(xa.size(0), xb.size(0))
    else:
        y = _convert_to_tensor_if_needed(y)

    _C.cphoc(xa, xb, y, method)
    return y


def pphoc(x, y=None, method="sum_prod_log"):
    """Pairwise probabilistic PHOC relevance scores."""
    x = _convert_to_tensor_if_needed(x)
    if y is None:
        y = x.new(x.size(0) * (x.size(0) - 1) // 2)
    else:
        y = _convert_to_tensor_if_needed(y)

    _C.pphoc(x, y, method)
    return y
