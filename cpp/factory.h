#ifndef PROB_PHOC_FACTORY_H_
#define PROB_PHOC_FACTORY_H_

#include <memory>
#include <string>

namespace prob_phoc {

namespace generic { template <typename T> class Impl; }

template <typename T, c10::DeviceType D>
class ImplFactory {
 public:
  typedef typename generic::Impl<T> Impl;

  static std::unique_ptr<Impl> CreateImpl(const std::string& name);
};

}  // namespace prob_phoc

#endif  // PROB_PHOC_FACTORY_H_
