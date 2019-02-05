#ifndef PROB_PHOC_GENERIC_H_
#define PROB_PHOC_GENERIC_H_

#include <torch/extension.h>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace prob_phoc {
namespace generic {

template <typename T>
class PairwiseOp {
 public:
  __host__ __device__
  PairwiseOp() {}

  __host__ __device__
  virtual ~PairwiseOp() {}

  __host__ __device__
  virtual T operator()(const long int n, const T* a, const T* b) const = 0;
};

template <typename T>
class SumProdRealSemiringOp : public PairwiseOp<T> {
 public:
  __host__ __device__
  T operator()(const long int n, const T* pa, const T* pb) const override {
    T result = 1;
    for (auto i = 0; i < n; ++i) {
      const auto pa0 = 1 - pa[i], pa1 = pa[i];
      const auto pb0 = 1 - pb[i], pb1 = pb[i];
      const auto ph0 = pa0 * pb0;
      const auto ph1 = pa1 * pb1;
      result *= ph0 + ph1;
    }
    return result;
  }
};

template <typename T>
class Impl {
 public:
  Impl() {}

  virtual ~Impl() {}

  virtual void cphoc(const c10::Device& device, const long int na, const long int nb, const long int d, const T* xa, const T* xb, T* y) const = 0;

  virtual void pphoc(const c10::Device& device, const long int n, const long int d, const T* x, T* y) const = 0;
};

}  // namespace generic
}  // namespace prob_phoc

#endif // PROB_PHOC_GENERIC_H_
