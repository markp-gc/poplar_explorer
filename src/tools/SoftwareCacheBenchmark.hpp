#pragma once

#include "ipu_utils.hpp"
#include "io_utils.hpp"
#include "tool_registry.hpp"

#include <memory/gather.hpp>
#include <memory/scatter.hpp>

#include <memory>

#include <boost/program_options.hpp>

/// A cache provides a local table of variables that can be filled from
/// a larger table of variables stored in a remote buffer (i.e. DRAM).
struct SoftwareCache {
  SoftwareCache(std::string cacheName,
                poplar::Type type,            // Element type of cache lines.
                std::size_t numLinesOffChip,  // Total number of cacheable lines in remote buffer.
                std::size_t maxCached,        // Number of cache lines held on chip.
                std::size_t lineSize,         // Number of elements in each cache line.
                std::size_t remoteFetchCount) // Number of lines fetched in each cache update.
  :
    name(cacheName),
    dataType(type),
    cacheableSetSize(numLinesOffChip),
    totalCacheLines(maxCached),
    cacheLineSize(lineSize),
    fetchCount(remoteFetchCount),
    residentSet(cacheName + "/resident_set"),
    remoteFetchOffsets(cacheName + "/fetch_offsets"),
    cacheScatterOffsets(cacheName + "/scatter_offsets")
  {}

  std::string getRemoteBufferName() const { return name + "/remote_feature_buffer"; }

  void build(poplar::Graph& graph, bool optimiseCopyMemoryUse = true) {
    using namespace poplar::program;

    ipu_utils::logger()->info("Cache '{}': Building cache of {} lines of size {}.", name, totalCacheLines, cacheLineSize);

    // Create remote buffer for the feature store:
    ipu_utils::logger()->info("Cache '{}': Building remote buffer with {} remote features", name, cacheableSetSize);
    remoteFeatures = graph.addRemoteBuffer(getRemoteBufferName(), dataType, cacheLineSize, cacheableSetSize);

    // Create variables needed for the cache:
    residentSet = popops::createSliceableTensor(graph, dataType, {totalCacheLines, cacheLineSize}, {0}, {1}, {}, {}, name + "/resident_set");
    cacheReadProg.add(residentSet.buildRead(graph, optimiseCopyMemoryUse));

    remoteFetchOffsets = graph.addVariable(poplar::UNSIGNED_INT, {fetchCount}, poplar::VariableMappingMethod::LINEAR);
    offsetStreamSequence.add(remoteFetchOffsets.buildWrite(graph, optimiseCopyMemoryUse));

    scatter::MultiUpdate scatterToCache(name + "/scatter_to_cache", residentSet, fetchCount, false);
    scatterToCache.plan(graph);

    // Create a temporary tensor for holding the IPU side cache.
    auto fetchBuffer = scatterToCache.createSource(graph);
    cacheScatterOffsets = scatterToCache.createIndices(graph);
    offsetStreamSequence.add(cacheScatterOffsets.buildWrite(graph, optimiseCopyMemoryUse));

    const std::vector<std::size_t> fetchShape = {fetchCount, cacheLineSize};

    // The fetch program will read fromt the remote buffer into the
    // fetch buffer and then scatter from the fetch buffer into the cache:
    ipu_utils::logger()->info("Cache '{}': Building cache fetch program (fetches {} features)", name, fetchCount);
    cacheFetchProg = Sequence();
    cacheFetchProg.add(Copy(remoteFeatures, fetchBuffer.reshape(fetchShape), remoteFetchOffsets, name + "/copy_host_features_to_cache"));
    cacheFetchProg.add(WriteUndef(remoteFetchOffsets, name + "/unlive_feature_offsets"));

    ipu_utils::logger()->info("Cache '{}': Building update (scatter {} features from fetchbuffer into residentSet).", name, fetchCount);
    scatterToCache.createProgram(graph, fetchBuffer, cacheScatterOffsets, cacheFetchProg);
    cacheFetchProg.add(WriteUndef(fetchBuffer, name + "/unlive_fetch_buffer"));
    cacheFetchProg.add(WriteUndef(cacheScatterOffsets, name + "/unlive_fetch_buffer_indices"));

    ipu_utils::logger()->info("Done building cache");
  }

  void connectStreams(
    poplar::Engine& e,
    std::vector<std::uint32_t>& remoteIndices,
    std::vector<std::uint32_t>& localIndices,
    std::vector<std::int32_t>& cacheData
  ) {
    ipu_utils::connectStream(e, remoteFetchOffsets.getWriteHandle(), remoteIndices);
    ipu_utils::connectStream(e, cacheScatterOffsets.getWriteHandle(), localIndices);
    ipu_utils::connectStream(e, residentSet.getReadHandle(), cacheData);
  }

  const std::string name;
  poplar::Type dataType;
  const std::size_t cacheableSetSize; // Total number of lines stored in the remote buffer.
                                      // Only a subset can be on chip at once.
  const std::size_t totalCacheLines;  // Number of cache lines on chip.
  const std::size_t cacheLineSize;    // Number of elements on each cache line.
  const std::size_t fetchCount;       // Number of lines that can be fetched from the
                                      // remote buffer in one copy. (If this is too big it results
                                      // in excessive internal exchange code so large fetches should
                                      // be broken down into a series of smaller fetches).

  // Remote buffer where all the cacheable data is stored:
  poplar::RemoteBuffer remoteFeatures;

  // Tensor that holds the on chip cached data. (Actually a multi-set in
  // general since nothing enforces it as a set).
  ipu_utils::StreamableTensor residentSet;

  // Tensor that describes which feature indices to fetch from the remote buffer into the on
  // IPU cache. These could be updated by host or IPU itself (push or pull to cache).
  ipu_utils::StreamableTensor remoteFetchOffsets;

  // Tensor that describes where the features fetched from the remote-buffer
  // should be scattered to in the on-device cache.
  ipu_utils::StreamableTensor cacheScatterOffsets;

  // A program to update the offsets before updating the cache: it streams from
  // the host all the offsets that describe the cache update.
  poplar::program::Sequence offsetStreamSequence;

  // Program to update the cache by using the offset indices to read from the
  // remote buffer then scatter to the cache. This program invalidates the indices.
  poplar::program::Sequence cacheFetchProg;

  // Program to read back the entire cache to the host (mainly intended for debugging):
  poplar::program::Sequence cacheReadProg;
};

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
};
