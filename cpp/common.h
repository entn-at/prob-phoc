#ifndef PROB_PHOC_COMMON_H_
#define PROB_PHOC_COMMON_H_

#define CHECK_CONTIGUOUS(x)                                     \
  AT_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CHECK_NDIM(x, d) do {                                           \
    const auto xd = (x).dim();                                          \
    AT_CHECK(xd == (d), #x " has the wrong number of dimensions "       \
             "(expected: " #d ", actual: " + std::to_string(xd) + ")"); \
  } while(0)

#define CHECK_SAME_NUM_SAMPLES(t1, t2) do {                                   \
    const auto s1 = (t1).size(0);                                             \
    const auto s2 = (t2).size(0);                                             \
    AT_CHECK(s1 == s2, "First dimension (number of samples in the batch) of " \
             #t1 " and " #t2 " must be equal "                                \
             "(" + std::to_string(s1) + " vs. " + std::to_string(s2) + ")");  \
  } while(0)

#define CHECK_SAME_NUM_OUTPUTS(t1, t2) do {                                  \
    const auto s1 = (t1).size(1);                                            \
    const auto s2 = (t2).size(1);                                            \
    AT_CHECK(s1 == s2, "Second dimension (number of outputs per sample) of " \
             #t1 " and " #t2 " must be equal "                               \
             "(" + std::to_string(s1) + " vs. " + std::to_string(s2) + ")"); \
  } while(0)

#define CHECK_SAME_SCALAR_TYPE(t1, t2) do {                             \
    const auto st1 = (t1).scalar_type();                                \
    const auto st2 = (t2).scalar_type();                                \
    const std::string sst1 = toString(st2);                             \
    const std::string sst2 = toString(st2);                             \
    AT_CHECK(st1 == st2, "Scalar type of " #t1 " and " #t2 " must be equal " \
             "(" + sst1 + " vs. " + sst2 + ")");                        \
  } while(0)

#define CHECK_SAME_DEVICE(t1, t2) do {                                  \
    std::ostringstream t1d, t2d;                                        \
    t1d << (t1).device(); t2d << (t2).device();                         \
    AT_CHECK((t1).device() == (t2).device(),                            \
             #t1 " and " #t2 " must be allocated in the same device "   \
             "(" + t1d.str() + " vs. " + t2d.str() + ")");              \
  } while(0)

#endif
