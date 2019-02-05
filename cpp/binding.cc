#include <torch/extension.h>

#include "./common.h"
#include "./cpu.h"
#ifdef WITH_CUDA
#include "./gpu.h"
#endif  // WITH_CUDA

namespace prob_phoc {

#define ERROR_IMPL(func, method, type) do {                             \
    const std::string t = (type).toString();                            \
    AT_ERROR(func " with method \"" + (method) + "\" not implemented for type " + t); \
  } while(0)

#define CHECK_IMPL(func, method, type, impl)  do {     \
    if (!(impl)) ERROR_IMPL(func, method, type);       \
  } while(0)

void cphoc(const at::Tensor& xa, const at::Tensor& xb, at::Tensor& y, const std::string& method) {
  CHECK_SAME_DEVICE(xa, xb);
  CHECK_SAME_DEVICE(xa, y);
  CHECK_SAME_SCALAR_TYPE(xa, xb);
  CHECK_SAME_SCALAR_TYPE(xa, y);
  CHECK_NDIM(xa, 2);
  CHECK_NDIM(xb, 2);
  CHECK_SAME_NUM_OUTPUTS(xa, xb);

  const auto na = xa.size(0);
  const auto nb = xb.size(0);
  const auto d = xa.size(1);
  y.resize_({na, nb});

#define DEFINE_SWITCH_CASE_OP(scalar_t, device_type)                    \
  case device_type: {                                                   \
    auto impl = ImplFactory<scalar_t, device_type>::CreateImpl(method); \
    CHECK_IMPL("cphoc", method, y.type(), impl);                        \
    impl->cphoc(                                                        \
        y.device(), na, nb, d, xa.data<scalar_t>(), xb.data<scalar_t>(), \
        y.data<scalar_t>());                                            \
  }                                                                     \
  break

  AT_DISPATCH_FLOATING_TYPES(y.type(), "cphoc", [&]{
      switch (y.device().type()) {
        DEFINE_SWITCH_CASE_OP(scalar_t, c10::Device::Type::CPU);
#ifdef WITH_CUDA
        DEFINE_SWITCH_CASE_OP(scalar_t, c10::Device::Type::CUDA);
#endif  // WITH_CUDA
        default:
          ERROR_IMPL("cphoc", method, y.type());
      }
    });

#undef DEFINE_SWITCH_CASE_OP
}

void pphoc(const at::Tensor& x, at::Tensor& y, const std::string& method) {
  CHECK_SAME_DEVICE(x, y);
  CHECK_SAME_SCALAR_TYPE(x, y);
  CHECK_NDIM(x, 2);

  const auto n = x.size(0);
  const auto d = x.size(1);
  y.resize_({n * (n - 1) / 2});

#define DEFINE_SWITCH_CASE_OP(scalar_t, device_type)                    \
  case device_type: {                                                   \
    auto impl = ImplFactory<scalar_t, device_type>::CreateImpl(method); \
    CHECK_IMPL("pphoc", method, y.type(), impl);                        \
    impl->pphoc(y.device(), n, d, x.data<scalar_t>(), y.data<scalar_t>()); \
  }                                                                     \
  break

  AT_DISPATCH_FLOATING_TYPES(y.type(), "pphoc", [&]{
      switch (y.device().type()) {
        DEFINE_SWITCH_CASE_OP(scalar_t, c10::Device::Type::CPU);
#ifdef WITH_CUDA
        DEFINE_SWITCH_CASE_OP(scalar_t, c10::Device::Type::CUDA);
#endif  // WITH_CUDA
        default:
          ERROR_IMPL("pphoc", method, y.type());
      }
    });

#undef DEFINE_SWITCH_CASE_OP
}

} // namespace prob_phoc

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "cphoc",
      prob_phoc::cphoc,
      "Probabilistic PHOC between each pair of the two collections of inputs.",
      pybind11::arg("xa"),
      pybind11::arg("xb"),
      pybind11::arg("y"),
      pybind11::arg("method") = "sum_prod_log");
  m.def(
      "pphoc",
      prob_phoc::pphoc,
      "Probabilistic PHOC between all pairs in the given input.",
      pybind11::arg("x"),
      pybind11::arg("y"),
      pybind11::arg("method") = "sum_prod_log");
}
