from __future__ import absolute_import

import unittest

import numpy as np
import torch

from prob_phoc import cphoc, pphoc


class ProbPHOCTest(unittest.TestCase):
    @staticmethod
    def test_cphoc_sum_prod_log():
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
        z = cphoc(x, y, method="sum_prod_log").numpy()
        np.testing.assert_almost_equal(z, expected_z)
        # Test float32
        z = cphoc(
            x.type("torch.FloatTensor"),
            y.type("torch.FloatTensor"),
            method="sum_prod_log",
        )
        z = z.numpy()
        np.testing.assert_almost_equal(z, expected_z.astype(np.float32))

    @staticmethod
    def test_cphoc_sum_prod_real():
        x = torch.DoubleTensor([[0.7, 0.4], [0.9, 0.6]])
        y = torch.DoubleTensor([[0.5, 0.9], [0.8, 0.9], [0.1, 0.2]])
        expected_z = np.asarray(
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

        # Test float64
        z = cphoc(x, y, method="sum_prod_real").numpy()
        np.testing.assert_almost_equal(z, expected_z)
        # Test float32
        z = cphoc(
            x.type("torch.FloatTensor"),
            y.type("torch.FloatTensor"),
            method="sum_prod_real",
        )
        z = z.numpy()
        np.testing.assert_almost_equal(z, expected_z.astype(np.float32))

    @staticmethod
    def test_pphoc_sum_prod_log():
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
        y = pphoc(x, method="sum_prod_log").numpy()
        np.testing.assert_almost_equal(y, expected_y)
        # Test float32 (float)
        y = pphoc(x.type("torch.FloatTensor"), method="sum_prod_log").numpy()
        np.testing.assert_almost_equal(y, expected_y.astype(np.float32))

    @staticmethod
    def test_pphoc_sum_prod_real():
        x = torch.DoubleTensor([[.7, .4], [.9, .4], [.5, .2]])
        expected_y = np.asarray(
            [
                (.7 * .9 + .3 * .1) * (.4 * .4 + .6 * .6),
                (.7 * .5 + .3 * .5) * (.4 * .2 + .6 * .8),
                (.9 * .5 + .1 * .5) * (.4 * .2 + .6 * .8),
            ],
            dtype=np.float64,
        )
        # Test float64 (double)
        y = pphoc(x, method="sum_prod_real").numpy()
        np.testing.assert_almost_equal(y, expected_y)
        # Test float32 (float)
        y = pphoc(x.type("torch.FloatTensor"), method="sum_prod_real").numpy()
        np.testing.assert_almost_equal(y, expected_y.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
