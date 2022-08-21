// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <poplar/Graph.hpp>

poplar::Tensor vstack(const std::vector<poplar::Tensor>& vectors);
poplar::Tensor hstack(const std::vector<poplar::Tensor>& vectors);

template <class T>
std::vector<T> slice(const std::vector<T>& v, std::size_t start, std::size_t end) {
  return std::vector<T>(v.begin() + start, v.begin() + end);
}