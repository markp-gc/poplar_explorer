// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include "ipu_utils.hpp"
#include "io_utils.hpp"
#include "tool_registry.hpp"
#include <memory/SoftwareCache.hpp>

#include <boost/program_options.hpp>
#include <memory>

struct SoftwareCacheBenchmark :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  SoftwareCacheBenchmark() {}
  virtual ~SoftwareCacheBenchmark() {}

  // Builder interface:
  void build(poplar::Graph& g, const poplar::Target&) override;
  void execute(poplar::Engine& engine, const poplar::Device& device) override;

  // Tool interface:
  void addToolOptions(boost::program_options::options_description& desc) override;
  void init(const boost::program_options::variables_map& args) override;

private:
  std::unique_ptr<SoftwareCache> cache;
  ipu_utils::RuntimeConfig runConfig;
  ipu_utils::ProgramManager programs;
  std::size_t residentSetSize;
  std::size_t cacheableSetSize;
  std::size_t lineSize;
  std::size_t fetchCount;
  std::size_t iterations;
  std::size_t seed;
  bool optimiseCycles;
};
