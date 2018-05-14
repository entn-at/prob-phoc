import unittest

import torch

from prob_phoc.phoc import cphoc, pphoc


class ProbPhocTest(unittest.TestCase):
    def cphoc_test(self):
        x = torch.Tensor([[0.7, 0.4], [0.9, 0.4]]).log_()
        y = cphoc(x, x)

    def phoc_test(self):
        x = torch.Tensor([[0.7, 0.4], [0.9, 0.4]]).log_()
        y = pphoc(x)


if __name__ == '__main__':
    unittest.main()
