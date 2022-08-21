// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <unistd.h>
#include "utils.hpp"

poplar::Tensor vstack(const std::vector<poplar::Tensor>& vectors) {
  std::vector<poplar::Tensor> rowVectors;
  rowVectors.reserve(vectors.size());
  for (const auto& v : vectors) {
    if (v.rank() != 1) {
      throw std::logic_error("vstack operates only on vectors or batches of vectors.");
    }
    if (v.rank() == 1) {
      rowVectors.push_back(v.reshape({1, v.numElements()}));
    } else if (v.rank() == 2) {
      rowVectors.push_back(v);
    }
  }

  return poplar::concat(rowVectors, 0);
}

poplar::Tensor hstack(const std::vector<poplar::Tensor>& vectors) {
  std::vector<poplar::Tensor> colVectors;
  colVectors.reserve(vectors.size());
  for (const auto& v : vectors) {
    if (v.rank() != 1 && v.rank() != 2) {
      throw std::logic_error("hstack operates only on vectors or batches of vectors.");
    }
    if (v.rank() == 1) {
      colVectors.push_back(v.reshape({v.numElements(), 1}));
    } else if (v.rank() == 2) {
      colVectors.push_back(v.transpose());
    }
  }

  return poplar::concat(colVectors, 1);
}
