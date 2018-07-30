from __future__ import absolute_import

import unittest

import numpy as np
import torch

from prob_phoc.phoc import cphoc, pphoc
from prob_phoc.default_impl import compute_upper_bound


def cphoc_upper_bound(xx, yy):
    assert xx.size(1) == yy.size(1)
    out = xx.new(xx.size(0), yy.size(0)).zero_()
    for i, x in enumerate(xx):
        for j, y in enumerate(yy):
            out[i, j] = compute_upper_bound(x, y)
    return out


def pphoc_upper_bound(xx):
    N = xx.size(0)
    out = cphoc_upper_bound(xx, xx)
    out2 = []
    for i in range(N):
        for j in range(i + 1, N):
            out2.append(out[i, j])
    if isinstance(xx, torch.FloatTensor):
        return torch.FloatTensor(out2)
    else:
        return torch.DoubleTensor(out2)


class ProbPHOCTest(unittest.TestCase):
    @staticmethod
    def test_cphoc_arbitrary_independence():
        x = torch.DoubleTensor([[0.7, 0.4], [0.9, 0.6]]).log_()
        y = torch.DoubleTensor([[0.5, 0.9], [0.8, 0.9], [0.1, 0.2]]).log_()
        expected_z = np.log(
            np.asarray(
                [
                    [
                        (.7 * .5 + .3 * .5) * (.4 * .9 + .6 * .1),
                        (.7 * .8 + .3 * .2) * (.4 * .9 + .6 * .1),
                        (.7 * .1 + .3 * .9) * (.4 * .2 + .6 * .8),
                    ],
                    [
                        (.9 * .5 + .1 * .5) * (.6 * .9 + .4 * .1),
                        (.9 * .8 + .1 * .2) * (.6 * .9 + .4 * .1),
                        (.9 * .1 + .1 * .9) * (.6 * .2 + .4 * .8),
                    ],
                ],
                dtype=np.float32,
            )
        )
        # Test float64
        z = cphoc(x, y, method="independence").numpy()
        np.testing.assert_almost_equal(z, expected_z)
        # Test float32
        z = cphoc(
            x.type("torch.FloatTensor"),
            y.type("torch.FloatTensor"),
            method="independence",
        )
        z = z.numpy()
        np.testing.assert_almost_equal(z, expected_z.astype(np.float32))

    @staticmethod
    def test_cphoc_arbitrary_upper_bound():
        x = torch.DoubleTensor([[0.7, 0.4], [0.9, 0.6]]).log_()
        y = torch.DoubleTensor([[0.5, 0.9], [0.8, 0.9], [0.1, 0.2]]).log_()
        expected_z = cphoc_upper_bound(x, y).numpy()
        # Test float64
        z = cphoc(x, y, method="upper_bound").numpy()
        np.testing.assert_almost_equal(z, expected_z)
        # Test float32
        z = cphoc(
            x.type("torch.FloatTensor"),
            y.type("torch.FloatTensor"),
            method="upper_bound",
        )
        z = z.numpy()
        np.testing.assert_almost_equal(z, expected_z.astype(np.float32))

    @staticmethod
    def test_pphoc_independence():
        x = torch.DoubleTensor([[.7, .4], [.9, .4], [.5, .2]]).log_()
        expected_y = np.log(
            np.asarray(
                [
                    (.7 * .9 + .3 * .1) * (.4 * .4 + .6 * .6),
                    (.7 * .5 + .3 * .5) * (.4 * .2 + .6 * .8),
                    (.9 * .5 + .1 * .5) * (.4 * .2 + .6 * .8),
                ],
                dtype=np.float64,
            )
        )
        # Test float64 (double)
        y = pphoc(x, method="independence").numpy()
        np.testing.assert_almost_equal(y, expected_y)
        # Test float32 (float)
        y = pphoc(x.type("torch.FloatTensor"), method="independence").numpy()
        np.testing.assert_almost_equal(y, expected_y.astype(np.float32))

    @staticmethod
    def test_pphoc_arbitrary_upper_bound():
        x = torch.DoubleTensor([[.7, .4], [.9, .4], [.5, .2]]).log_()
        expected_z = pphoc_upper_bound(x).numpy()
        # Test float64
        z = pphoc(x, method="upper_bound").numpy()
        np.testing.assert_almost_equal(z, expected_z)
        # Test float32
        z = pphoc(x.type("torch.FloatTensor"), method="upper_bound").numpy()
        np.testing.assert_almost_equal(z, expected_z.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
