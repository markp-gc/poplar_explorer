// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#pragma once

#include "ipu_utils.hpp"
#include "io_utils.hpp"
#include "tool_registry.hpp"

#include <boost/program_options.hpp>

struct KNNBenchmark :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  KNNBenchmark();
  virtual ~KNNBenchmark();

  // Builder interface:
  void build(poplar::Graph& g, const poplar::Target&) override;
  void execute(poplar::Engine& engine, const poplar::Device& device) override;

  // Tool interface:
  void addToolOptions(boost::program_options::options_description& desc) override;
  void init(const boost::program_options::variables_map& args) override {}

private:
  std::size_t batchSize;
  std::size_t k;
  std::size_t D;
  std::size_t numVecs;
  std::size_t iterations;
  ipu_utils::StreamableTensor query;
  ipu_utils::StreamableTensor vecs;
  ipu_utils::StreamableTensor results;
};
 