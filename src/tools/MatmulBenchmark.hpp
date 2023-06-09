// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include "ipu_utils.hpp"
#include "io_utils.hpp"
#include "tool_registry.hpp"

#include <boost/program_options.hpp>

struct MatmulBenchmark :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  MatmulBenchmark();
  virtual ~MatmulBenchmark();

  // Builder interface:
  void build(poplar::Graph& g, const poplar::Target&) override;
  void execute(poplar::Engine& engine, const poplar::Device& device) override;

  // Tool interface:
  void addToolOptions(boost::program_options::options_description& desc) override;
  void init(const boost::program_options::variables_map& args) override {}

private:
  std::size_t lhsRows;
  std::size_t lhsCols;
  std::size_t rhsCols;
  std::size_t iterations;
  std::string dataTypeString;
  std::string partialsType;
  poplar::Type dtype;
  float availableMemoryProportion;
  ipu_utils::StreamableTensor lhsMatrices;
  ipu_utils::StreamableTensor rhsMatrices;
  ipu_utils::StreamableTensor results;
  ipu_utils::StreamableTensor cycleCount;
  std::size_t tilesUsed;
};
