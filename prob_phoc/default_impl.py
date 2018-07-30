from __future__ import absolute_import

import math
from scipy.misc import logsumexp
import numpy as np
import torch


def _logmexpm1(x):
    try:
        return math.log(-math.expm1(x))
    except ValueError:
        return -np.inf


def compute_independent(a, b):
    assert torch.is_tensor(a)
    assert torch.is_tensor(b)
    assert len(a) == len(b)
    a, b = a.view(-1), b.view(-1)

    result = 0.0
    for a1, b1 in zip(a, b):
        h1 = a1 + b1
        try:
            h0 = math.log(-math.expm1(a1)) + math.log(-math.expm1(b1))
        except ValueError:
            h0 = -np.inf
        result += logsumexp([h0, h1])

    return result


def compute_upper_bound(a, b):
    assert torch.is_tensor(a)
    assert torch.is_tensor(b)
    assert len(a) == len(b)
    a, b = a.view(-1), b.view(-1)
    if len(a) == 0:
        return 0.0

    ma = max(a[0], -math.expm1(a[0]))
    mb = max(b[0], -math.expm1(b[0]))
    result = max(a[0] + b[0], -(math.expm1(a[0]) + math.expm1(b[0])))
    for a1, b1 in zip(a[1:], b[1:]):
        a0, b0 = _logmexpm1(a1), _logmexpm1(b1)
        aux0 = min(a0 + b0, a0 + mb, b0 + ma, result)
        aux1 = min(a1 + b1, a1 + mb, b1 + ma, result)
        result = max(aux0, aux1)
        ma = min(ma, max(a0, a1))
        mb = min(mb, max(b0, b1))
    return result
