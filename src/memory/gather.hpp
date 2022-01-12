#pragma once

#include "ipu_utils.hpp"

#include <poputil/TileMapping.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Encoding.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/codelets.hpp>

#include <vector>

namespace gather {

struct MultiSlice {
  const std::string name;
  std::size_t inputSize;
  std::size_t featureSize;
  std::size_t outputSize;
  poplar::OptionFlags optionFlags;
  popops::SlicePlan slicePlan;
  const bool planned;

  MultiSlice(const std::string& name,
             std::size_t inputs, std::size_t dimension, std::size_t outputs, bool usePlan) :
              name(name),
              inputSize(inputs),
              featureSize(dimension),
              outputSize(outputs),
              planned(usePlan)
  {}

  void plan(poplar::Graph& graph) {
    if (planned) {
      optionFlags = {{"availableMemoryProportion", "0.1"}, {"usedForUpdate", "false"}};
      slicePlan = popops::embedding::plan(graph, poplar::FLOAT, inputSize, featureSize, {outputSize, 1}, optionFlags);
    }
  }

  poplar::Tensor createValues(poplar::Graph& graph) {
    ipu_utils::logger()->info("inputSize: {} feature Size: {}", inputSize, featureSize);
    return popops::createSliceableTensor(graph, poplar::FLOAT, {inputSize, featureSize}, {0}, {1}, slicePlan, optionFlags, name + "/values");
  }

  poplar::Tensor createIndices(poplar::Graph& graph) {
    return popops::createIndicesTensor(graph, {0}, outputSize, slicePlan, optionFlags, name + "/indices");
  }

  poplar::Tensor createOutput(poplar::Graph& graph, const poplar::Tensor& values, const poplar::Tensor& indices, poplar::program::Sequence& program) {
    return popops::multiSlice(graph, values, indices, {0}, {1}, program, slicePlan, optionFlags, name + "/output");
  }
};

} // end namespace gather
