// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include "ipu_utils.hpp"

#include <poputil/TileMapping.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Encoding.hpp>
#include <poputil/TileMapping.hpp>
#include <popops/codelets.hpp>
#include <popops/Fill.hpp>
#include <popops/Cast.hpp>

#include <vector>
#include <random>
#include <algorithm>

namespace scatter {

struct MultiUpdate {
  const std::string name;
  const poplar::Tensor valuesToUpdate;
  const std::size_t featureCount;
  const std::size_t featureSize;
  const std::size_t count;
  poplar::OptionFlags optionFlags;
  popops::SlicePlan slicePlan;
  const bool planned;

  MultiUpdate(const std::string& name,
              poplar::Tensor destination,
              std::size_t updateCount, bool usePlan)
  :
    valuesToUpdate(destination),
    featureCount(destination.dim(0)),
    featureSize(destination.dim(1)),
    count(updateCount),
    planned(usePlan)
  {}

  void plan(poplar::Graph& graph) {
    if (planned) {
      optionFlags = {{"availableMemoryProportion", "0.2"}, {"usedForUpdate", "true"}};
      // multiUpdate doesn't support slice plans at the moment:
      //slicePlan = popops::embedding::plan(graph, poplar::FLOAT, featureCount, featureSize, {count, 1}, optionFlags);
    }
  }

  poplar::Tensor createSource(poplar::Graph& graph) {
    auto sliceFrom = popops::createSliceTensor(graph, valuesToUpdate, {0}, {1}, count, name + "/source");
    return sliceFrom;
  }

  poplar::Tensor createIndices(poplar::Graph& graph) {
    return popops::createIndicesTensor(graph, {0}, count, slicePlan, optionFlags, name + "/indices");
  }

  void createProgram(poplar::Graph& graph,
                     const poplar::Tensor& valuesToInsert,
                     const poplar::Tensor& indicesToUpdate,
                     poplar::program::Sequence& program) {
    ipu_utils::logger()->debug("Feature size: {}", featureSize);
    ipu_utils::logger()->debug("Dst shape: ({})", valuesToUpdate.shape());
    ipu_utils::logger()->debug("Src shape: ({})", valuesToInsert.shape());
    ipu_utils::logger()->debug("Indices shape: ({})", indicesToUpdate.shape());

    popops::multiUpdate(graph, valuesToUpdate, valuesToInsert, indicesToUpdate, {0}, {1}, program, slicePlan, optionFlags, name + "/output");
  }
};

} // end namespace scatter
