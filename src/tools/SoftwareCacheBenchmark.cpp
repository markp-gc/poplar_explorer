// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <cstdlib>
#include <vector>
#include <limits>
#include <chrono>

#include "SoftwareCacheBenchmark.hpp"

void SoftwareCacheBenchmark::init(const boost::program_options::variables_map& args) {
  slicer.reset(
    new gather::MultiSlice(
      "embedding",
      deviceFeatureCount,
      featureDim,
      gatherCount,
      true)
  );
  cache.reset(
    new FeatureCache(
      "feature_cache",
      poplar::FLOAT,
      totalFeatureCount,
      slicer->featureSize,
      fetchCount,
      true)
  );
}

ipu_utils::RuntimeConfig SoftwareCacheBenchmark::getRuntimeConfig() const {
  return runConfig;
}

void SoftwareCacheBenchmark::build(poplar::Graph& graph, const poplar::Target&) {
  using namespace poplar::program;

  popops::addCodelets(graph);

  // Create gather inputs:
  bool optimiseCopyMemoryUse = false;
  slicer->plan(graph);
  auto features = slicer->createValues(graph);
  auto indices = slicer->createIndices(graph);
  auto gatherIndicesStream =
    graph.addHostToDeviceFIFO("write_indices", indices.elementType(), indices.numElements());
  programs.add("write_gather_indices", Copy(gatherIndicesStream, indices, optimiseCopyMemoryUse, "write_gather_indices"));

  // Program to do the on IPU gather:
  Sequence gatherProg;
  auto result = slicer->createOutput(graph, features, indices, gatherProg);
  programs.add("gather", gatherProg);
  auto resultStream =
    graph.addDeviceToHostFIFO("read_gather_result",
                              result.elementType(),
                              result.numElements());
  programs.add("read_result", Copy(result, resultStream));

  // Build the cache:
  cache->build(graph, features, optimiseCopyMemoryUse);
  programs.add("write_cache_indices", cache->offsetStreamSequence);
  programs.add("fill_cache", cache->fillCache);
}

ipu_utils::ProgramManager& SoftwareCacheBenchmark::getPrograms() { return programs; }

void SoftwareCacheBenchmark::execute(poplar::Engine& engine, const poplar::Device& device) {
  if (!device.supportsRemoteBuffers()) {
    throw std::runtime_error("Remote buffers are not supported on this device.");
  }

  ipu_utils::logger()->info("Execution starts");

  const auto& progs = getPrograms();

  // Fill the remote buffer with the entire feature table:
  auto fillStartTime = std::chrono::steady_clock::now();
  std::vector<float> featureVector(slicer->featureSize, 0);
  for (std::size_t i = 0; i < cache->totalFeatureCount; ++i) {
    std::fill(featureVector.begin(),featureVector.end(), i);
    engine.copyToRemoteBuffer(featureVector.data(), cache->getRemoteBufferName(), i);
  }
  auto hostCachePushEndTime = std::chrono::steady_clock::now();
  auto seconds = std::chrono::duration<double>(hostCachePushEndTime - fillStartTime).count();
  auto gigaBytesPerSec = (1e-9 / seconds) * slicer->featureSize * cache->totalFeatureCount * sizeof(float);
  ipu_utils::logger()->info("Remote-buffer fill time (host to remote-buffer): {} secs rate: {} GB/sec", seconds, gigaBytesPerSec);

  // List of feature indices to cache on the IPU and list of where to cache them:
  std::vector<std::uint32_t> remoteFeatureIndices(cache->fetchCount);
  std::vector<std::uint32_t> cacheDestinationIndices(cache->fetchCount);
  for (std::size_t i = 0; i < remoteFeatureIndices.size(); ++i) {
    remoteFeatureIndices[i] = 10 + i;
  }
  for (std::size_t i = 0; i < cacheDestinationIndices.size(); ++i) {
    cacheDestinationIndices[i] = cacheDestinationIndices.size() - 1 - i;
  }
  ipu_utils::connectStream(engine, cache->remoteFetchOffsets.getWriteHandle(), remoteFeatureIndices);
  ipu_utils::connectStream(engine, cache->cacheScatterOffsets.getWriteHandle(), cacheDestinationIndices);

  // List of indices to gather from the cache on the IPU:
  std::vector<std::uint32_t> gatherIndices(slicer->outputSize);
  for (std::size_t i = 0; i < gatherIndices.size(); ++i) {
    gatherIndices[i] = i;
  }

  ipu_utils::connectStream(engine, "write_indices", gatherIndices);

  std::vector<float> result(slicer->outputSize * slicer->featureSize);
  ipu_utils::connectStream(engine, "read_gather_result", result);

  ipu_utils::logger()->info("Running {} iterations", iterations);
  auto startTime = std::chrono::steady_clock::now();
  for (auto i = 0u; i < iterations; ++i) {
    progs.run(engine, "write_cache_indices");
    progs.run(engine, "fill_cache");
  }
  auto ipuCachePullEndTime = std::chrono::steady_clock::now();
  seconds = std::chrono::duration<double>(ipuCachePullEndTime - startTime).count();
  gigaBytesPerSec = (1e-9 / seconds) * slicer->featureSize * cache->fetchCount * iterations * sizeof(float);
  ipu_utils::logger()->info("Remote-buffer pull time (remote-buffer to IPU): {} secs rate: {} GB/sec", seconds, gigaBytesPerSec);

  progs.run(engine, "write_gather_indices");
  progs.run(engine, "gather");
  progs.run(engine, "read_result");
  auto endTime = std::chrono::steady_clock::now();
  auto totalTime = std::chrono::duration<double>(endTime - startTime).count();
  ipu_utils::logger()->info("Total execution time: {} secs.", totalTime);

  std::cerr << "Result:\n";
  if (slicer->featureSize < 16 && slicer->outputSize < 50) {
    auto v = result.begin();
    for (auto f = 0u; f < slicer->outputSize; ++f) {
      for (auto i = 0u; i < slicer->featureSize; ++i) {
        std::cerr << *v << " ";
        v += 1;
      }
      std::cerr << "\n";
    }
  } else {
    std::cerr << "Output supressed (too large).\n";
  }
}

void SoftwareCacheBenchmark::setRuntimeConfig(const ipu_utils::RuntimeConfig& cfg) {
  runConfig = cfg;
}

void SoftwareCacheBenchmark::addToolOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("device-feature-count", po::value<std::size_t>(&deviceFeatureCount)->default_value(10000),
   "Number of feature vectors in lookup on device."
  )
  ("total-feature-count", po::value<std::size_t>(&totalFeatureCount)->default_value(100000),
   "Number of feature vectors in total (in remote buffer)."
  )
  ("feature-dim", po::value<std::size_t>(&featureDim)->default_value(1024),
   "Dimension of feature vectors."
  )
  ("gather-count", po::value<std::size_t>(&gatherCount)->default_value(256),
   "Number of feature vectors to gather in on device embedding lookup."
  )
  ("fetch-count", po::value<std::size_t>(&fetchCount)->required(),
   "Number of features to fetch onto to device in a single remote buffer read."
  )
  ("iterations", po::value<std::size_t>(&iterations)->default_value(1000),
   "Number of pull-to-cache iterations."
  )
  ;
}