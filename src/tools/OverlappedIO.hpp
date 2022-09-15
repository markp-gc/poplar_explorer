// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <ipu_utils.hpp>
#include <io_utils.hpp>
#include <tool_registry.hpp>

#include <poputil/TileMapping.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <gcl/TileAllocation.hpp>

#include <tuple>

/// This tool shows how to use an overlapped IO pipeline so
/// that the IPU can compute and communicate with the host in
/// parallel.
class OverlappedIO :
  public ipu_utils::BuilderInterface, public ToolInterface
{
public:
  OverlappedIO() {}
  virtual ~OverlappedIO() {}

  /// Tool interface:
  void addToolOptions(boost::program_options::options_description& desc) override;
  void init(const boost::program_options::variables_map& args) override;

  /// Builder interface:
  void build(poplar::Graph& graph, const poplar::Target& target) override;
  void execute(poplar::Engine& engine, const poplar::Device& device) override;

private:

  std::tuple<poplar::ComputeSet, poplar::Tensor, poplar::Tensor>
  buildComputeGraph(poplar::Graph& compute_graph, poplar::Type dtype);

  std::tuple<poplar::Tensor, poplar::Tensor>
  buildIOGraph(poplar::Graph& io_graph, const poplar::Target& target, poplar::Type elementType);

  unsigned numTilesForIO;
  std::vector<unsigned> ioTiles;
  std::vector<unsigned> computeTiles;
  std::size_t numWorkerContexts;
  std::size_t sizePerWorker;
  std::size_t numIterations;
  std::size_t numComputeTiles;
  std::size_t numTransferInElements;
  std::size_t numTransferOutElements;
};
