
#include <poplar/Graph.hpp>

poplar::Tensor vstack(const std::vector<poplar::Tensor>& vectors);
poplar::Tensor hstack(const std::vector<poplar::Tensor>& vectors);
