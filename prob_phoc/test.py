from __future__ import absolute_import

import itertools
import numpy as np
import torch
import unittest

from prob_phoc import cphoc, pphoc

dtype2str = {np.float32: "f32", np.float64: "f64"}


class ProbPHOCTest(unittest.TestCase):
    @staticmethod
    def run_cphoc_sum_prod(device, dtype, use_log):
        x = np.asarray([[.7, .4], [.9, .6]], dtype=dtype)
        y = np.asarray([[.5, .9], [.8, .9], [.1, .2]], dtype=dtype)
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
            dtype=dtype,
        )

        x = torch.tensor(x, device=device)
        y = torch.tensor(y, device=device)
        if use_log:
            x = x.log_()
            y = y.log_()
            expected_z = np.log(expected_z)
            method = "sum_prod_log"
        else:
            method = "sum_prod_real"

        z = cphoc(x, y, method=method).cpu().numpy()
        np.testing.assert_almost_equal(z, expected_z)

    @staticmethod
    def run_pphoc_sum_prod(device, dtype, use_log):
        x = np.asarray([[.7, .4], [.9, .4], [.5, .2]], dtype=dtype)
        expected_y = np.asarray(
            [
                (.7 * .9 + .3 * .1) * (.4 * .4 + .6 * .6),
                (.7 * .5 + .3 * .5) * (.4 * .2 + .6 * .8),
                (.9 * .5 + .1 * .5) * (.4 * .2 + .6 * .8),
            ],
            dtype=dtype,
        )

        x = torch.tensor(x, device=device)
        if use_log:
            x = x.log_()
            expected_y = np.log(expected_y)
            method = "sum_prod_log"
        else:
            method = "sum_prod_real"

        y = pphoc(x, method=method).cpu().numpy()
        np.testing.assert_almost_equal(y, expected_y)


devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
dtypes = [np.float32, np.float64]
for device, dtype, use_log in itertools.product(devices, dtypes, [False, True]):

    def _test_cphoc():
        ProbPHOCTest.run_cphoc_sum_prod(device, dtype, use_log)

    def _test_pphoc():
        ProbPHOCTest.run_pphoc_sum_prod(device, dtype, use_log)

    semiring = "log" if use_log else "real"
    test_name = "test_cphoc_sum_prod_{}_{}_{}".format(
        semiring, device, dtype2str[dtype]
    )
    setattr(ProbPHOCTest, test_name, staticmethod(_test_cphoc))

    test_name = "test_pphoc_sum_prod_{}_{}_{}".format(
        semiring, device, dtype2str[dtype]
    )
    setattr(ProbPHOCTest, test_name, staticmethod(_test_pphoc))

if __name__ == "__main__":
    unittest.main()
