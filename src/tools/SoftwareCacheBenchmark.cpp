// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <cstdlib>
#include <vector>
#include <limits>
#include <chrono>

#include "SoftwareCacheBenchmark.hpp"

#include <poprand/codelets.hpp>
#include <poprand/RandomGen.hpp>

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

  // Build the graph for the cache:
  bool optimiseMemoryUse = !optimiseCycles;
  ipu_utils::logger()->info("Optimise memory use: {}", optimiseMemoryUse);
  cache->build(graph, optimiseMemoryUse);

  // Register programs:
  getPrograms().add("write_indices", cache->offsetStreamSequence);
  getPrograms().add("cache_fetch", cache->cacheFetchProg);
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
  for (auto i = 0u; i < iterations; ++i) {
    progs.run(engine, "cache_fetch");
  }
  auto cacheFetchEndTime = std::chrono::steady_clock::now();
  seconds = std::chrono::duration<double>(cacheFetchEndTime - cacheFetchStartTime).count();
  gigaBytesPerSec = (1e-9 / seconds) * lineSize * fetchCount * iterations * sizeof(float);
  ipu_utils::logger()->info("Cache fetch time (remote-buffer to IPU): {} secs rate: {} GB/sec", seconds, gigaBytesPerSec);

  if (cacheContents.size() < 100) {
    progs.run(engine, "copy_cache_to_host");
    ipu_utils::logger()->info("Cache state:\n{}", cacheContents);
  } else {
    ipu_utils::logger()->info("Supressed output: too large ({} elements).", cacheContents.size());
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
  ("optimise-cycles", po::bool_switch(&optimiseCycles)->default_value(false))
  ;
}
