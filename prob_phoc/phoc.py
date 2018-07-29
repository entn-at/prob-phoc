from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import warnings

try:
    from ._ext import cphoc_f32, cphoc_f64, cphoc_min_f32, cphoc_min_f64
    from ._ext import pphoc_f32, pphoc_f64, pphoc_min_f32, pphoc_min_f64
except ImportError as ex:
    warnings.warn('The C++ implementation of prob_phoc could not be imported '
                  '(%s). Python implementation will be used. ' % str(ex))

    import math
    from scipy.misc import logsumexp

    def _compute(a, b):
        result = 0.0
        for i in range(a.size(0)):
            h1 = a[i] + b[i]
            try:
                h0 = math.log(-math.expm1(a[i])) + math.log(-math.expm1(b[i]))
            except ValueError:
                h0 = -np.inf
            result += logsumexp([h0, h1])

        return result

    def _compute_min(a, b):
        result = float('inf')
        for i in range(a.size(0)):
            h1 = a[i] + b[i]
            try:
                h0 = math.log(-math.expm1(a[i])) + math.log(-math.expm1(b[i]))
            except ValueError:
                h0 = -np.inf
            result = min(result, logsumexp([h0, h1]))

        return result


    def _cphoc(x, y, out):
        for i in range(x.size(0)):
            for j in range(y.size(0)):
                out[i, j] = _compute(x[i], y[j])


    def _pphoc(x, out):
        k = 0
        for i in range(x.size(0)):
            for j in range(i + 1, x.size(0)):
                out[k] = _compute(x[i], x[j])
                k += 1

    def _cphoc_min(x, y, out):
        for i in range(x.size(0)):
            for j in range(y.size(0)):
                out[i, j] = _compute_min(x[i], y[j])


    def _pphoc_min(x, out):
        k = 0
        for i in range(x.size(0)):
            for j in range(i + 1, x.size(0)):
                out[k] = _compute_min(x[i], x[j])
                k += 1

    cphoc_f32 = cphoc_f64 = _cphoc
    pphoc_f32 = pphoc_f64 = _pphoc
    cphoc_min_f32 = cphoc_min_f64 = _cphoc_min
    pphoc_min_f32 = pphoc_min_f64 = _pphoc_min


def cphoc(x, y, out=None, product=True):
    """Computes probabilistic PHOC relevance scores between each pair of inputs.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    assert torch.is_tensor(x) and torch.is_tensor(y)
    assert x.type() in ['torch.FloatTensor', 'torch.DoubleTensor']
    assert x.type() == y.type()
    assert x.dim() == 2 and y.dim() == 2
    assert x.size(1) == y.size(1)
    x, y = x.cpu(), y.cpu()

    if out is None:
        out = x.new(x.size(0), y.size(0)).zero_()

    if x.type() == 'torch.FloatTensor':
        if product:
            cphoc_f32(x, y, out)
        else:
            cphoc_min_f32(x, y, out)
    else:
        if product:
            cphoc_f64(x, y, out)
        else:
            cphoc_min_f64(x, y, out)

    return out


def pphoc(x, out=None, product=True):
    """Pairwise probabilistic PHOC relevance scores."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    assert torch.is_tensor(x)
    assert x.type() in ['torch.FloatTensor', 'torch.DoubleTensor']
    assert x.dim() == 2
    x = x.cpu()

    if out is None:
        out = x.new(x.size(0) * (x.size(0) - 1) // 2,).zero_()

    if x.type() == 'torch.FloatTensor':
        if product:
            pphoc_f32(x, out)
        else:
            pphoc_min_f32(x, out)
    else:
        if product:
            pphoc_f64(x, out)
        else:
            pphoc_min_f64(x, out)

    return out
