from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import warnings

try:
    from ._ext import cphoc_f32, cphoc_f64, cphoc_max_f32, cphoc_max_f64
    from ._ext import pphoc_f32, pphoc_f64, pphoc_max_f32, pphoc_max_f64
except ImportError as ex:
    warnings.warn(
        "The C++ implementation of prob_phoc could not be imported "
        "(%s). Python implementation will be used. " % str(ex)
    )

    from .default_impl import compute_independent, compute_upper_bound

    def _cphoc(x, y, out):
        for i in range(x.size(0)):
            for j in range(y.size(0)):
                out[i, j] = compute_independent(x[i], y[j])

    def _pphoc(x, out):
        k = 0
        for i in range(x.size(0)):
            for j in range(i + 1, x.size(0)):
                out[k] = compute_independent(x[i], x[j])
                k += 1

    def _cphoc_max(x, y, out):
        for i in range(x.size(0)):
            for j in range(y.size(0)):
                out[i, j] = compute_upper_bound(x[i], y[j])

    def _pphoc_max(x, out):
        k = 0
        for i in range(x.size(0)):
            for j in range(i + 1, x.size(0)):
                out[k] = compute_upper_bound(x[i], x[j])
                k += 1

    cphoc_f32 = cphoc_f64 = _cphoc
    pphoc_f32 = pphoc_f64 = _pphoc
    cphoc_max_f32 = cphoc_max_f64 = _cphoc_max
    pphoc_max_f32 = pphoc_max_f64 = _pphoc_max


def cphoc(x, y, out=None, method="independence"):
    """Computes probabilistic PHOC relevance scores between each pair of inputs.
    """
    assert method in ["independence", "upper_bound"]
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    assert torch.is_tensor(x) and torch.is_tensor(y)
    assert x.type() in ["torch.FloatTensor", "torch.DoubleTensor"]
    assert x.type() == y.type()
    assert x.dim() == 2 and y.dim() == 2
    assert x.size(1) == y.size(1)
    x, y = x.cpu(), y.cpu()

    if out is None:
        out = x.new(x.size(0), y.size(0)).zero_()

    if x.type() == "torch.FloatTensor":
        if method == "independence":
            cphoc_f32(x, y, out)
        else:
            cphoc_max_f32(x, y, out)
    else:
        if method == "independence":
            cphoc_f64(x, y, out)
        else:
            cphoc_max_f64(x, y, out)

    return out


def pphoc(x, out=None, method="independence"):
    """Pairwise probabilistic PHOC relevance scores."""
    assert method in ["independence", "upper_bound"]
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    assert torch.is_tensor(x)
    assert x.type() in ["torch.FloatTensor", "torch.DoubleTensor"]
    assert x.dim() == 2
    x = x.cpu()

    if out is None:
        out = x.new(x.size(0) * (x.size(0) - 1) // 2).zero_()

    if x.type() == "torch.FloatTensor":
        if method == "independence":
            pphoc_f32(x, out)
        else:
            pphoc_max_f32(x, out)
    else:
        if method == "independence":
            pphoc_f64(x, out)
        else:
            pphoc_max_f64(x, out)

    return out
