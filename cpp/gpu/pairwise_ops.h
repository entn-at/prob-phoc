#ifndef PROB_PHOC_GPU_PAIRWISE_OPS_H_
#define PROB_PHOC_GPU_PAIRWISE_OPS_H_

#include <limits>

#include <thrust/swap.h>

#include "../generic.h"

#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif

namespace prob_phoc {
namespace gpu {

template <typename T>
__host__ __device__
T infinity() { return std::numeric_limits<T>::infinity(); }

#ifdef __CUDA_ARCH__
template <>
__host__ __device__
float infinity<float>() { return CUDART_INF_F; }

template <>
__host__ __device__
double infinity<double>() { return CUDART_INF; }
#endif

template <typename T>
__host__ __device__
inline T logsumexp(T a, T b) {
  if (b > a) { thrust::swap(a, b); }
  return a + log1p(exp(b - a));
}

template <typename T>
class SumProdLogSemiringOp : public generic::PairwiseOp<T> {
 public:
  using generic::PairwiseOp<T>::PairwiseOp;

  __host__ __device__
  T operator()(const long int n, const T* pa, const T* pb) const override {
    T result = 0;
    for (auto i = 0; i < n; ++i) {
      const T pa0 = -expm1(pa[i]), log_pa1 = pa[i];
      const T pb0 = -expm1(pb[i]), log_pb1 = pb[i];
      const T log_ph0 =
          (pa0 > 0 && pb0 > 0)
          ?  log(pa0 * pb0)
          : -infinity<T>();
      const T log_ph1 = log_pa1 + log_pb1;
      result += logsumexp(log_ph0, log_ph1);
    }
    return result;
  }
};

}  // namespace gpu
}  // namespace prob_phoc

#endif  // PROB_PHOC_GPU_PAIRWISE_OPS_H_
