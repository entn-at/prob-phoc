#ifndef PROB_PHOC_GPU_H_
#define PROB_PHOC_GPU_H_

#include <memory>

#include <c10/Device.h>
#include <c10/DeviceType.h>

#include "./factory.h"
#include "./generic.h"
#include "./gpu/pairwise_ops.h"


namespace prob_phoc {
namespace gpu {

template <typename T, typename O>
class Impl : public generic::Impl<T> {
 public:
  Impl(const O& pairwise_op) : op_(pairwise_op) {}

  void cphoc(const c10::Device& device, const long int na, const long int nb, const long int d, const T* xa, const T* xb, T* y) const override;

  void pphoc(const c10::Device& device, const long int n, const long int d, const T* x, T* y) const override;

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

}  // namespace gpu


// Factory class for GPU-based implementations
template <typename T>
class ImplFactory<T, c10::DeviceType::CUDA> {
 public:
  typedef typename generic::Impl<T> Impl;

  static std::unique_ptr<Impl> CreateImpl(const std::string& name) {
    if (name == "sum_prod_real") {
      return std::unique_ptr<Impl>(new gpu::SumProdRealSemiring<T>());
    } else if (name == "sum_prod_log") {
      return std::unique_ptr<Impl>(new gpu::SumProdLogSemiring<T>());
    } else {
      return nullptr;
    }
  }
};

}  // namespace prob_phoc

#endif // PROB_PHOC_gPU_H_
