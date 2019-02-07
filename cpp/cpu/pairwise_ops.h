#ifndef PROB_PHOC_CPU_PAIRWISE_OPS_H_
#define PROB_PHOC_CPU_PAIRWISE_OPS_H_

#include <algorithm>
#include <cmath>
#include <limits>

#include "../generic.h"

namespace prob_phoc {
namespace cpu {

template <typename T>
inline T logsumexp(T a, T b) {
  if (b > a) { std::swap(a, b); }
  return a + std::log1p(std::exp(b - a));
}

template <typename T>
class SumProdLogSemiringOp : public generic::PairwiseOp<T> {
 public:
  using generic::PairwiseOp<T>::PairwiseOp;

  T operator()(const long int n, const T* pa, const T* pb) const override {
    T result = 0;
    for (auto i = 0; i < n; ++i) {
      const T pa0 = -std::expm1(pa[i]), log_pa1 = pa[i];
      const T pb0 = -std::expm1(pb[i]), log_pb1 = pb[i];
      const T log_ph0 =
          (pa0 > 0 && pb0 > 0)
          ?  std::log(pa0 * pb0)
          : -std::numeric_limits<T>::infinity();
      const T log_ph1 = log_pa1 + log_pb1;
      result += logsumexp(log_ph0, log_ph1);
    }
    return result;
  }
};


}  // namespace cpu
}  // namespace prob_phoc

#endif  // PROB_PHOC_CPU_PAIRWISE_OPS_H_
