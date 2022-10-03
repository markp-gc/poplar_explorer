// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <cstdlib>
#include <vector>
#include <limits>
#include <chrono>

#include "SoftwareCacheBenchmark.hpp"

#include <poprand/codelets.hpp>
#include <poprand/RandomGen.hpp>
#include <gcl/TileAllocation.hpp>

template <class T>
std::vector<T> slice(const std::vector<T>& v, std::size_t start, std::size_t end) {
  return std::vector<T>(v.begin() + start, v.begin() + end);
}

struct AsyncIoGraphs {
  AsyncIoGraphs(poplar::Graph& graph, unsigned numTilesForIO)
  : ioTiles(gcl::perIPUTiles(graph, 0, numTilesForIO)),
    computeTiles(gcl::perIPUTiles(graph, numTilesForIO, graph.getTarget().getNumTiles() - numTilesForIO))
  {
    ioGraph = graph.createVirtualGraph(ioTiles);
    computeGraph = graph.createVirtualGraph(computeTiles);
  }

  std::vector<unsigned> ioTiles;
  std::vector<unsigned> computeTiles;
  poplar::Graph ioGraph;
  poplar::Graph computeGraph;
};

AsyncIoGraphs makeIoGraph(poplar::Graph& graph, unsigned numTilesForIO) {
  // Get two disjoint sets of tiles to use for compute and IO:
  const auto minIOTiles = gcl::getMinIoTiles(graph);
  numTilesForIO = std::max(minIOTiles, numTilesForIO);

  return AsyncIoGraphs(graph, numTilesForIO);
}

void SoftwareCacheBenchmark::init(const boost::program_options::variables_map& args) {
  cache.reset(
    new SoftwareCache(
      "on_chip_cache", poplar::INT,
      cacheableSetSize, residentSetSize, lineSize, fetchCount)
  );
}

void SoftwareCacheBenchmark::build(poplar::Graph& graph, const poplar::Target&) {
  using namespace poplar::program;

  popops::addCodelets(graph);
  poprand::addCodelets(graph);

  auto graphs = makeIoGraph(graph, numIoTiles);
  ipu_utils::logger()->info("Reserved {} tiles for asynchronous IO", graphs.ioTiles.size());

  // Build the graph for the cache:
  bool optimiseMemoryUse = !optimiseCycles;
  ipu_utils::logger()->info("Optimise memory use: {}", optimiseMemoryUse);
  cache->build(graphs.computeGraph, graphs.ioGraph, optimiseMemoryUse);

  // Make a program that generates random remote buffer indices and random scatter indices:

  // Randomise the scatter indices for next fetch:
  Sequence randomiseIndicesProg;
  auto randIndicesLocal = poprand::uniform(graphs.computeGraph, nullptr, 0u, cache->cacheScatterOffsets, poplar::INT, 0, cache->residentSet.numElements(), randomiseIndicesProg, "gen_rand_scatter_indices");

  // Randomise the remote buffer fetch indices:
  auto randIndicesRemote = graphs.computeGraph.addVariable(poplar::INT, cache->remoteFetchOffsets.shape(), "compute_tile_fetch_buffer");
  poputil::mapTensorLinearly(graphs.computeGraph, randIndicesRemote);
  randIndicesRemote = poprand::uniform(graphs.computeGraph, nullptr, 0u, randIndicesRemote, poplar::INT, 0, cache->cacheableSetSize, randomiseIndicesProg, "gen_rand_fetch_indices");

  // Cast the new indices:
  auto castComputeSet = graphs.ioGraph.addComputeSet("indices_cast_cs");
  popops::cast(graphs.computeGraph, randIndicesLocal, cache->cacheScatterOffsets, castComputeSet);
  auto newFetchOffsets = popops::cast(graphs.computeGraph, randIndicesRemote, poplar::UNSIGNED_INT, castComputeSet, "cast_");
  randomiseIndicesProg.add(Execute(castComputeSet));
  // Add a program to copy the remote fetch indices from the compute tiles back to the IO tiles:
  randomiseIndicesProg.add(Copy(newFetchOffsets, cache->remoteFetchOffsets, "exchange_new_fetch_offsets"));

  // Create the asynchronous I/O pipeline:
  auto mainSequence = Sequence {
    cache->updateResidentSetProg,
    cache->readMemoryProg,
    cache->cacheExchangeProg,
    randomiseIndicesProg,
  };

  auto pipeline = Sequence {
    randomiseIndicesProg,
    cache->readMemoryProg,
    cache->cacheExchangeProg,
    Repeat(iterations - 1, mainSequence),
    cache->updateResidentSetProg
  };

  // Register programs:
  getPrograms().add("write_indices", cache->offsetStreamSequence);
  getPrograms().add("cache_io_pipeline", pipeline);
  getPrograms().add("copy_cache_to_host", cache->cacheReadProg);
}

void SoftwareCacheBenchmark::execute(poplar::Engine& engine, const poplar::Device& device) {
  if (!device.supportsRemoteBuffers()) {
    throw std::runtime_error("Remote buffers are not supported on this device.");
  }

  ipu_utils::logger()->info("Execution starts");

  // Fill the entire remote buffer with data:
  auto fillStartTime = std::chrono::steady_clock::now();
  std::vector<std::int32_t> featureVector(lineSize, 0);
  for (std::size_t i = 0; i < cache->cacheableSetSize; ++i) {
    std::fill(featureVector.begin(), featureVector.end(), i);
    engine.copyToRemoteBuffer(featureVector.data(), cache->getRemoteBufferName(), i);
  }
  auto fillEndTime = std::chrono::steady_clock::now();
  auto seconds = std::chrono::duration<double>(fillEndTime - fillStartTime).count();
  auto gigaBytesPerSec = (1e-9 / seconds) * lineSize * cacheableSetSize * sizeof(float);
  ipu_utils::logger()->info("Remote-buffer rows: {}", cacheableSetSize);
  ipu_utils::logger()->info("Remote-buffer fill time (host to remote-buffer): {} secs rate: {} GB/sec", seconds, gigaBytesPerSec);

  // Make a list of indices of the remote buffer to fetch. Gen unique
  // random set of random indices to fetch:
  std::vector<std::uint32_t> remoteBufferIndices(cacheableSetSize);
  std::mt19937 g(seed);
  std::iota(remoteBufferIndices.begin(), remoteBufferIndices.end(), 0);
  std::shuffle(remoteBufferIndices.begin(), remoteBufferIndices.end(), g);
  remoteBufferIndices.resize(fetchCount);

  // List of locations in the cache for the fetched lines. Again use a random
  // permutation:
  std::vector<std::uint32_t> cacheDestinationIndices(residentSetSize);
  std::iota(cacheDestinationIndices.begin(), cacheDestinationIndices.end(), 0);
  std::shuffle(cacheDestinationIndices.begin(), cacheDestinationIndices.end(), g);
  cacheDestinationIndices.resize(fetchCount);

  if (remoteBufferIndices.size() < 10) {
    ipu_utils::logger()->info("Remote buffer indices to fetch:\n{}", remoteBufferIndices);
  }
  if (cacheDestinationIndices.size() < 10) {
    ipu_utils::logger()->info("Indices of destination in cache:\n{}", cacheDestinationIndices);
  }

  // Buffer to read back the cache at end:
  std::vector<std::int32_t> cacheContents(residentSetSize * lineSize);

  // Connect the streams to the buffers we just created:
  cache->connectStreams(engine, remoteBufferIndices, cacheDestinationIndices, cacheContents);

  // Set cache indices form the host:
  // TODO: generate random cache fetch indices on IPU.
  const auto& progs = getPrograms();
  progs.run(engine, "write_indices");

  // Repeatedly fetch data into the cache:
  ipu_utils::logger()->info("Running {} iterations of cache fetches", iterations);
  auto cacheFetchStartTime = std::chrono::steady_clock::now();
  progs.run(engine, "cache_io_pipeline");
  auto cacheFetchEndTime = std::chrono::steady_clock::now();
  seconds = std::chrono::duration<double>(cacheFetchEndTime - cacheFetchStartTime).count();
  double bytesPerCacheFetch = lineSize * fetchCount * sizeof(float);
  gigaBytesPerSec = (1e-9 / seconds) * bytesPerCacheFetch * iterations;
  ipu_utils::logger()->info("Cache fetch time (remote-buffer to IPU): {} secs rate: {} GB/sec", seconds, gigaBytesPerSec);

  // For debug/test read back the cache:
  progs.run(engine, "copy_cache_to_host");

  if (residentSetSize < 100) {
    ipu_utils::logger()->info("Cache state:\n");
    for (auto r = 0u; r < residentSetSize; ++r) {
      auto pos = r * lineSize;
      ipu_utils::logger()->info("Line {}: {}", r, slice(cacheContents, pos, pos + lineSize));
    }
  } else {
    ipu_utils::logger()->info("Supressed output: too large ({} elements).", cacheContents.size());
  }

  // Check cache contents is correct:
  std::vector<std::uint32_t> expected(residentSetSize);
  for (auto r = 0u; r < residentSetSize; ++r) {
    // Find the cache line index in the scattering indices:
    auto itr = std::find(cacheDestinationIndices.begin(), cacheDestinationIndices.end(), r);
    auto expected = 0u; // If its not there expect 0
    if (itr != cacheDestinationIndices.end()) {
      // Found the index so the expected value can be looked up in the corresponding position in the remote buffer. 
      expected = remoteBufferIndices[itr - cacheDestinationIndices.begin()];
    }
    auto pos = r * lineSize;
    auto line = slice(cacheContents, pos, pos + lineSize);
    if (!std::all_of(line.begin(), line.end(), [expected](std::uint32_t v) { return v == expected; })) {
      ipu_utils::logger()->error("Expected cache line {} to contain {} but saw {}", r, expected, line);
    }
  }
}

void SoftwareCacheBenchmark::addToolOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("resident-set-size", po::value<std::size_t>(&residentSetSize)->default_value(10000),
   "Number of cache lines stored in on chip-memory."
  )
  ("remote-buffer-size", po::value<std::size_t>(&cacheableSetSize)->default_value(100000),
   "Number of cacheable lines in total (in remote buffer)."
  )
  ("line-size", po::value<std::size_t>(&lineSize)->default_value(1024),
   "Number of elements in a cache line."
  )
  ("fetch-count", po::value<std::size_t>(&fetchCount)->required(),
   "Number of lines to fetch from remote buffer in a single cache update."
  )
  ("iterations", po::value<std::size_t>(&iterations)->default_value(1000),
   "Number of pull-to-cache iterations."
  )
  ("seed", po::value<std::size_t>(&seed)->default_value(10142),
   "Seed used to generate random indices."
  )
  ("num-io-tiles", po::value<std::size_t>(&numIoTiles)->default_value(32u),
   "Number of tiles to reserve for asynchronous I/O."
  )
  ("optimise-cycles", po::bool_switch(&optimiseCycles)->default_value(false))
  ;
}
