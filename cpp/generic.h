#ifndef PROB_PHOC_GENERIC_H_
#define PROB_PHOC_GENERIC_H_

#include <torch/extension.h>

namespace prob_phoc {
namespace generic {

template <typename T>
class Impl {
 public:
  typedef T SType;

  Impl() {}

  virtual ~Impl() {}

  virtual SType compute_for_pair(const long int d, const SType* a, const SType* b) const = 0;

  virtual void cphoc(const c10::Device& device, const long int na, const long int nb, const long int d, const SType* xa, const SType* xb, SType* y) const = 0;

  virtual void pphoc(const c10::Device& device, const long int n, const long int d, const SType* x, SType* y) const = 0;
};

template <typename B>
class SumProdRealSemiring : public B {
 public:
  typedef typename B::SType SType;

  SType compute_for_pair(const long int d, const SType* pa, const SType* pb) const override {
    SType result = 1;
    for (auto i = 0; i < d; ++i) {
      const auto pa0 = 1 - pa[i], pa1 = pa[i];
      const auto pb0 = 1 - pb[i], pb1 = pb[i];
      const auto ph0 = pa0 * pb0;
      const auto ph1 = pa1 * pb1;
      result *= ph0 + ph1;
    }
    return result;
  }

  static_assert(std::is_base_of<Impl<SType>, B>::value,
                "B must be descendant of generic::Impl");
};

}  // namespace generic
}  // namespace prob_phoc

#endif // PROB_PHOC_GENERIC_H_
