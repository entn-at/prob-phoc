#include <THC/THC.h>

#include "./../gpu.h"
#include "./defines.h"
#include "./pairwise_ops.h"


namespace prob_phoc {
namespace gpu {

template <typename T, typename O>
__global__
void cphoc_kernel(const long int na, const long int nb, const long int d, const T* xa, const T* xb, T* y, const O op) {
  for (long int i = thGy; i < na; i += NTGy) {
    for (long int j = thGx; j < nb; j += NTGx) {
      const T* xa_i = xa + i * d;
      const T* xb_j = xb + j * d;
      y[i * nb + j] = op(d, xa_i, xb_j);
    }
  }
}

template <typename T, typename O>
__global__
void pphoc_kernel(const long int n, const long int d, const T* x, T* y, const O op) {
  for (long int i = thGy; i < n; i += NTGy) {
    for (long int j = thGx; j < n; j += NTGx) {
      if (j > i) {
        const T* x_i = x + i * d;
        const T* x_j = x + j * d;
        const long k = i * (2 * n - i - 1) / 2 + (j - i - 1);
        y[k] = op(d, x_i, x_j);
      }
    }
  }
}

template <typename T, typename O>
void Impl<T, O>::cphoc(const c10::Device& device, const long int na, const long int nb, const long int d, const T* xa, const T* xb, T* y) const {
  c10::DeviceGuard device_guard(device);
  auto stream = THCState_getCurrentStream(at::globalContext().getTHCState());
  const dim3 block_size(32, 32);
  const dim3 grid_size(NUM_BLOCKS(na, 32),
                       NUM_BLOCKS(nb, 32));
  cphoc_kernel<T, O><<<grid_size, block_size, 0, stream>>>(na, nb, d, xa, xb, y, op_);
  if (stream == nullptr) {
    CHECK_LAST_CUDA_CALL();
  }
}

template <typename T, typename O>
void Impl<T, O>::pphoc(const c10::Device& device, const long int n, const long int d, const T* x, T* y) const {
  c10::DeviceGuard device_guard(device);
  auto stream = THCState_getCurrentStream(at::globalContext().getTHCState());
  const dim3 block_size(32, 32);
  const dim3 grid_size(NUM_BLOCKS(n, 32),
                       NUM_BLOCKS(n, 32));
  //pphoc_kernel<T, O><<<grid_size, block_size, 0, stream>>>(n, d, x, y, op_);
  if (stream == nullptr) {
    CHECK_LAST_CUDA_CALL();
  }
}

template class SumProdLogSemiring<float>;
template class SumProdLogSemiring<double>;

template class SumProdRealSemiring<float>;
template class SumProdRealSemiring<double>;

}  // namespace gpu
}  // namespace prob_phoc
