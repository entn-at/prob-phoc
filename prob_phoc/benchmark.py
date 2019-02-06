from __future__ import absolute_import

import itertools
import numpy as np
import torch
import timeit

from natsort import natsorted
from prob_phoc import cphoc, pphoc


def my_timeit(fn, seconds, args=None, kwargs=None):
    if args is None:
        args = list()
    elif not isinstance(args, (list, tuple)):
        raise ValueError("args must be a list or tuple")

    if kwargs is None:
        kwargs = dict()
    elif not isinstance(kwargs, dict):
        raise ValueError("kwargs must be a dictionary")

    start_time = timeit.default_timer()
    number_of_calls = 0
    while True:
        fn(*args, **kwargs)
        number_of_calls += 1
        end_time = timeit.default_timer()
        if end_time - start_time > seconds:
            return (end_time - start_time) / number_of_calls


class ProbPHOCBenchmark(object):
    @staticmethod
    def _run_cphoc_sum_prod(device, dtype, use_log, n, d):
        xa = torch.randn((n, d), dtype=dtype, device=device)
        xb = torch.randn((n, d), dtype=dtype, device=device)
        if use_log:
            xa = torch.nn.functional.log_softmax(xa, dim=1)
            xb = torch.nn.functional.log_softmax(xb, dim=1)
            method = "sum_prod_log"
        else:
            xa = torch.nn.functional.softmax(xa, dim=1)
            xb = torch.nn.functional.softmax(xb, dim=1)
            method = "sum_prod_real"

        # Warm-up
        _ = cphoc(xa[:2, :], xb[:2, :], method=method)
        # Pre-allocate memory for the output
        y = torch.zeros((n, n), dtype=dtype, device=device)
        # Run cphoc for at least 10 seconds.
        return my_timeit(
            fn=cphoc, seconds=5, args=(xa, xb), kwargs=dict(y=y, method=method)
        )

    @staticmethod
    def _run_pphoc_sum_prod(device, dtype, use_log, n, d):
        x = torch.randn((n, d), dtype=dtype, device=device)
        if use_log:
            x = torch.nn.functional.log_softmax(x, dim=1)
            method = "sum_prod_log"
        else:
            x = torch.nn.functional.softmax(x, dim=1)
            method = "sum_prod_real"

        # Warm-up
        _ = pphoc(x[:2, :], method=method)
        # Pre-allocate memory for the output
        y = torch.zeros((n * (n - 1) // 2,), dtype=dtype, device=device)
        # Run pphoc for at least 10 seconds.
        return my_timeit(
            fn=pphoc, seconds=5, args=(x,), kwargs=dict(y=y, method=method)
        )

    def main(self):
        benchmark_names = [
            name
            for name in dir(self)
            if name.startswith("benchmark_") and callable(getattr(self, name))
        ]
        benchmark_names = natsorted(benchmark_names)
        width = max([len(name) for name in benchmark_names])
        print("# {:{w}} {}".format("benchmark_name", "time_per_call", w=width - 2))
        for name in benchmark_names:
            duration = getattr(self, name)()
            print("{:{w}} {}".format(name, duration, w=width))


def _generate_benchmark(benchmark_name, method, device, dtype, use_log, n, d):
    dtype2str = {torch.float32: "f32", torch.float64: "f64"}
    semiring = "log" if use_log else "real"
    benchmark_name = "benchmark_{}_{}_{}_{}_{}_{}".format(
        benchmark_name, semiring, device, dtype2str[dtype], n, d
    )
    setattr(
        ProbPHOCBenchmark,
        benchmark_name,
        staticmethod(
            lambda: getattr(ProbPHOCBenchmark, method)(device, dtype, use_log, n, d)
        ),
    )


devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
dtypes = [torch.float32, torch.float64]
rows = [100, 500, 1000, 5000]
columns = [10, 50, 100]
for device, dtype, use_log, n, d in itertools.product(
    devices, dtypes, [False, True], rows, columns
):
    _generate_benchmark(
        "cphoc_sum_prod", "_run_cphoc_sum_prod", device, dtype, use_log, n, d
    )
    _generate_benchmark(
        "pphoc_sum_prod", "_run_pphoc_sum_prod", device, dtype, use_log, n, d
    )


if __name__ == "__main__":
    ProbPHOCBenchmark().main()
