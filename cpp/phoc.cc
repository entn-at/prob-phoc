#include <algorithm>
#include <cmath>
#include <limits>

#include <Python.h>
#include <THW/THTensor.h>

extern "C" {
#include <phoc.h>
}

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;
using nnutils::THW::TensorTraits;

template <typename T>
static inline T logsumexp(T a, T b) {
   if (b > a) { std::swap(a, b); }
   return a + std::log1p(std::exp(b - a));
}

class ComputePair {
 public:
  template <typename Int, typename T>
  inline T operator()(Int n, const T* a, const T* b) {
    T result = 0;
    for (Int i = 0; i < n; ++i) {
      const T pa0 = -std::expm1(a[i]);
      const T pb0 = -std::expm1(b[i]);
      const T lh1 = a[i] + b[i];
      const T lh0 = (pa0 > 0 && pb0 > 0)
          ? std::log(pa0) + std::log(pb0)
          : -std::numeric_limits<T>::infinity();
      result += logsumexp(lh0, lh1);
    }
    return result;
  }
};

class ComputePairMin {
 public:
  template <typename Int, typename T>
  inline T operator()(Int n, const T* a, const T* b) {
    T result = std::numeric_limits<T>::max();
    for (Int i = 0; i < n; ++i) {
      const T pa0 = -std::expm1(a[i]);
      const T pb0 = -std::expm1(b[i]);
      const T lh1 = a[i] + b[i];
      const T lh0 = (pa0 > 0 && pb0 > 0)
          ? std::log(pa0) + std::log(pb0)
          : -std::numeric_limits<T>::infinity();
      result = std::min(result, logsumexp(lh0, lh1));
    }
    return result;
  }
};

template <typename TT, typename Callable>
static inline int cphoc(const ConstTensor<TT>& X,
                        const ConstTensor<TT>& Y,
                        MutableTensor<TT>* R,
                        Callable func) {
  if (X.Dims() != 2 || Y.Dims() != 2) {
    PyErr_SetString(PyExc_ValueError, "Input arrays must be matrices");
    return 1;
  }
  if (X.Size(1) != Y.Size(1)) {
    PyErr_SetString(PyExc_ValueError, "Input matrices must have the same "
                                      "number of columns");
    return 1;
  }
  if (!X.IsContiguous() || !Y.IsContiguous()) {
    PyErr_SetString(PyExc_ValueError, "Input matrices must be continuous");
    return 1;
  }
  R->Resize({X.Size(0), Y.Size(0)});
  if (!R->IsContiguous()) {
    PyErr_SetString(PyExc_SystemError,
                    "Output matrix is not contiguous, THIS IS A BUG!");
    return 1;
  }

  const auto NX = X.Size(0);
  const auto NY = Y.Size(0);
  const auto ND = X.Size(1);
  #pragma omp parallel for collapse(2)
  for (auto i = 0; i < NX; ++i) {
    for (auto j = 0; j < NY; ++j) {
      const auto xi = X.Data() + i * ND;
      const auto yj = Y.Data() + j * ND;
      R->Data()[i * NY + j] = func(ND, xi, yj);
    }
  }

  return 0;
}

template <typename TT, typename Callable>
static inline int pphoc(const ConstTensor<TT>& X,
                        MutableTensor<TT>* R,
                        Callable func) {
  if (X.Dims() != 2) {
    PyErr_SetString(PyExc_ValueError, "Input array must be a matrix");
    return 1;
  }
  if (!X.IsContiguous()) {
    PyErr_SetString(PyExc_ValueError, "Input matrix must be continuous");
    return 1;
  }
  const long nelem = X.Size(0) * (X.Size(0) - 1) / 2;
  R->Resize(std::vector<long>{nelem});
  if (!R->IsContiguous()) {
    PyErr_SetString(PyExc_SystemError,
                    "Output matrix is not contiguous, THIS IS A BUG!");
    return 1;
  }

  const auto NX = X.Size(0);
  const auto ND = X.Size(1);
  #pragma omp parallel for schedule(static, 128) collapse(2)
  for (auto i = 0; i < NX; ++i) {
    for (auto j = 0; j < NX; ++j) {
      if (j > i) {
        const auto xi = X.Data() + i * ND;
        const auto yj = X.Data() + j * ND;
        const auto k = i * (2 * NX - i - 1) / 2 + (j - i - 1);
        R->Data()[k] = func(ND, xi, yj);
      }
    }
  }

  return 0;
}

#define DEFINE_WRAPPER(STYPE, TTYPE)                                    \
  int cphoc_##STYPE(const TTYPE* X, const TTYPE* Y, TTYPE* R) {         \
    ConstTensor<TTYPE> tX(X);                                           \
    ConstTensor<TTYPE> tY(Y);                                           \
    MutableTensor<TTYPE> tR(R);                                         \
    return cphoc(tX, tY, &tR, ComputePair());                           \
  }                                                                     \
                                                                        \
  int pphoc_##STYPE(const TTYPE* X, TTYPE* R) {                         \
    ConstTensor<TTYPE> tX(X);                                           \
    MutableTensor<TTYPE> tR(R);                                         \
    return pphoc(tX, &tR, ComputePair());                               \
  }                                                                     \
                                                                        \
  int cphoc_min_##STYPE(const TTYPE* X, const TTYPE* Y, TTYPE* R) {     \
    ConstTensor<TTYPE> tX(X);                                           \
    ConstTensor<TTYPE> tY(Y);                                           \
    MutableTensor<TTYPE> tR(R);                                         \
    return cphoc(tX, tY, &tR, ComputePairMin());                        \
  }                                                                     \
                                                                        \
  int pphoc_min_##STYPE(const TTYPE* X, TTYPE* R) {                     \
    ConstTensor<TTYPE> tX(X);                                           \
    MutableTensor<TTYPE> tR(R);                                         \
    return pphoc(tX, &tR, ComputePairMin());                            \
  }

DEFINE_WRAPPER(f32, THFloatTensor)
DEFINE_WRAPPER(f64, THDoubleTensor)
