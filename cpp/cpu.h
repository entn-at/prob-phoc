#ifndef PROB_PHOC_CPU_H_
#define PROB_PHOC_CPU_H_

#include <memory>

#include <c10/Device.h>
#include <c10/DeviceType.h>

#include "./factory.h"
#include "./generic.h"
#include "./cpu/pairwise_ops.h"


namespace prob_phoc {
namespace cpu {

template <typename T, typename O>
class Impl : public generic::Impl<T> {
 public:
  Impl(const O& pairwise_op) : op_(pairwise_op) {}

  void cphoc(const c10::Device& device, const long int na, const long int nb, const long int d, const T* xa, const T* xb, T* y) const override {
    #pragma omp parallel for collapse(2)
    for (auto i = 0; i < na; ++i) {
      for (auto j = 0; j < nb; ++j) {
        const auto* xa_i = xa + i * d;
        const auto* xb_j = xb + j * d;
        y[i * nb + j] = op_(d, xa_i, xb_j);
      }
    }
  }

  void pphoc(const c10::Device& device, const long int n, const long int d, const T* x, T* y) const override {
    // TODO(jpuigcerver): Get rid of the if to avoid useless operations.
    #pragma omp parallel for schedule(static, 128) collapse(2)
    for (auto i = 0; i < n; ++i) {
      for (auto j = 0; j < n; ++j) {
        if (j > i) {
          const auto x_i = x + i * d;
          const auto x_j = x + j * d;
          const auto k = i * (2 * n - i - 1) / 2 + (j - i - 1);
          y[k] = op_(d, x_i, x_j);
        }
      }
    }
  }

 private:
  const O op_;
};


template <typename T>
class SumProdLogSemiring : public Impl<T, SumProdLogSemiringOp<T>> {
 public:
  SumProdLogSemiring() : Impl<T, SumProdLogSemiringOp<T>>(
      SumProdLogSemiringOp<T>()) {}
};

template <typename T>
class SumProdRealSemiring : public Impl<T, generic::SumProdRealSemiringOp<T>> {
 public:
  SumProdRealSemiring() : Impl<T, generic::SumProdRealSemiringOp<T>>(
      generic::SumProdRealSemiringOp<T>()) {}
};

}  // namespace cpu


// Factory class for CPU-based implementations
template <typename T>
class ImplFactory<T, c10::DeviceType::CPU> {
 public:
  typedef typename generic::Impl<T> Impl;

  static std::unique_ptr<Impl> CreateImpl(const std::string& name) {
    if (name == "sum_prod_real") {
      return std::unique_ptr<Impl>(new cpu::SumProdRealSemiring<T>());
    } else if (name == "sum_prod_log") {
      return std::unique_ptr<Impl>(new cpu::SumProdLogSemiring<T>());
    } else {
      return nullptr;
    }
  }
};

}  // namespace prob_phoc

#endif // PROB_PHOC_CPU_H_
