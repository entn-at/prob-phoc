#ifndef PROB_PHOC_CPU_H_
#define PROB_PHOC_CPU_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "./factory.h"
#include "./generic.h"

namespace prob_phoc {
namespace cpu {

template <typename T>
static inline T logsumexp(T a, T b) {
  if (b > a) { std::swap(a, b); }
  return a + std::log1p(std::exp(b - a));
}

template <typename T>
class Impl : public generic::Impl<T> {
 public:
  typedef typename generic::Impl<T>::SType SType;

  using generic::Impl<T>::compute_for_pair;

  void cphoc(const c10::Device& device, const long int na, const long int nb, const long int d, const SType* xa, const SType* xb, SType* y) const override {
    #pragma omp parallel for collapse(2)
    for (auto i = 0; i < na; ++i) {
      for (auto j = 0; j < nb; ++j) {
        const auto* xa_i = xa + i * d;
        const auto* xb_j = xb + j * d;
        y[i * nb + j] = compute_for_pair(d, xa_i, xb_j);
      }
    }
  }

  void pphoc(const c10::Device& device, const long int n, const long int d, const SType* x, SType* y) const override {
    #pragma omp parallel for schedule(static, 128) collapse(2)
    for (auto i = 0; i < n; ++i) {
      for (auto j = 0; j < n; ++j) {
        if (j > i) {
          const auto x_i = x + i * d;
          const auto x_j = x + j * d;
          const auto k = i * (2 * n - i - 1) / 2 + (j - i - 1);
          y[k] = compute_for_pair(d, x_i, x_j);
        }
      }
    }
  }
};

template <typename T>
class SumProdLogSemiring : public Impl<T> {
 public:
  typedef typename Impl<T>::SType SType;

  SType compute_for_pair(const long int d, const SType* pa, const SType* pb) const override {
    SType result = 0;
    for (auto i = 0; i < d; ++i) {
      const auto pa0 = -std::expm1(pa[i]), log_pa1 = pa[i];
      const auto pb0 = -std::expm1(pb[i]), log_pb1 = pb[i];
      const auto log_ph0 =
          (pa0 > 0 && pb0 > 0)
          ?  std::log(pa0 * pb0)
          : -std::numeric_limits<T>::infinity();
      const auto log_ph1 = log_pa1 + log_pb1;
      result += logsumexp(log_ph0, log_ph1);
    }
    return result;
  }
};

template <typename T>
using SumProdRealSemiring = generic::SumProdRealSemiring<cpu::Impl<T>>;

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
