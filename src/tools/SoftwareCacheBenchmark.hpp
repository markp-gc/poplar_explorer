#pragma once

#include "ipu_utils.hpp"
#include "io_utils.hpp"
#include "tool_registry.hpp"

#include <memory/gather.hpp>
#include <memory/scatter.hpp>

#include <memory>

#include <boost/program_options.hpp>

/// A cache provides a local table of features that can be filled
/// from a larger table of features stored in a remote buffer.
struct FeatureCache {
  FeatureCache(std::string cacheName, poplar::Type type,
               std::size_t featureCount,      // Total number of features in remote buffer.
               std::size_t featureSize,       // Length of each feature vector.
               std::size_t remoteFetchCount,  // Number of features fetched in each cache update.
               bool useSlicePlanner)          // Whether the scatter should be planned.
  :
    name(cacheName),
    dataType(type),
    totalFeatureCount(featureCount),
    featureDimension(featureSize),
    fetchCount(remoteFetchCount),
    useSlicePlan(useSlicePlanner),
    remoteFetchOffsets(cacheName + "/fetch_offsets"),
    cacheScatterOffsets(cacheName + "/scatter_offsets")
  {}

  std::string getRemoteBufferName() const { return name + "/remote_feature_buffer"; }

  void build(poplar::Graph& graph, poplar::Tensor& deviceFeatures, bool optimiseCopyMemoryUse) {
    using namespace poplar::program;

    // Create remote buffer for the feature store:
    ipu_utils::logger()->info("Building remote buffer with {} remote features", totalFeatureCount);
    remoteFeatures = graph.addRemoteBuffer(getRemoteBufferName(), poplar::FLOAT, featureDimension, totalFeatureCount);

    // Create the Tensor which indexes into the remote buffer and a stream so the host can update the indices:
    remoteFetchOffsets.buildTensor(graph, poplar::UNSIGNED_INT, {fetchCount}, poplar::VariableMappingMethod::LINEAR);
    offsetStreamSequence.add(remoteFetchOffsets.buildWrite(graph, optimiseCopyMemoryUse));

    ipu_utils::logger()->info("Building cache of {} features ({} features fetched per cache update).", deviceFeatures.dim(0), fetchCount);
    scatter::MultiUpdate cacheUpdate(name + "/scatter_to_cache", deviceFeatures, fetchCount, useSlicePlan);
    cacheUpdate.plan(graph);

    // Create a temporary tensor for holding the IPU side cache.
    auto fetchBuffer = cacheUpdate.createSource(graph);
    cacheScatterOffsets = cacheUpdate.createIndices(graph);
    offsetStreamSequence.add(cacheScatterOffsets.buildWrite(graph, optimiseCopyMemoryUse));

    // Program to fill pull the features currently indexed in remoteFetchOffsets into the cache:
    ipu_utils::logger()->info("Building cache fill program (fetches {} features)", fetchCount);
    fillCache = Sequence();
    fillCache.add(Copy(remoteFeatures, fetchBuffer.reshape({fetchCount, featureDimension}), remoteFetchOffsets, name + "/copy_host_features_to_cache"));
    fillCache.add(WriteUndef(remoteFetchOffsets, name + "/unlive_feature_offsets"));

    ipu_utils::logger()->info("Building cache update (scatter {} features from fetchbuffer into the on device cache).", fetchCount);
    cacheUpdate.createProgram(graph, fetchBuffer, cacheScatterOffsets, fillCache);

    // TODO: need to loop over a set of fetch indices to do larger transfers with low memory overhead.
    fillCache.add(WriteUndef(fetchBuffer, name + "/unlive_fetch_buffer"));
    fillCache.add(WriteUndef(cacheScatterOffsets, name + "/unlive_fetch_buffer_indices"));
    ipu_utils::logger()->info("Done building cache");
  }

  const std::string name;
  poplar::Type dataType;
  const std::size_t totalFeatureCount; // Total number stored in remote buffers.
  const std::size_t featureDimension;  // Dimension of feature vector space
  const std::size_t fetchCount;        // Number of features that can be fetched in one copy. (If this is too big it results in excessive internal exchange code).
  const bool useSlicePlan;
  poplar::RemoteBuffer remoteFeatures; // Remote buffer where all features are stored.
  ipu_utils::StreamableTensor remoteFetchOffsets; // Tensor that describes which feature indices to fetch from the remote buffer into the on IPU cache.
                                                  // Could be updated by host (push to cache) or IPU itself (pull to cache).
  ipu_utils::StreamableTensor cacheScatterOffsets; // Tensor that describes where the features fetched from the remote-buffer should be scattered to in the on-device cache.
  poplar::program::Sequence offsetStreamSequence; // A program to update the offsets before updating the cache: it streams from the host all the offsets that describe the cache update.
  poplar::program::Sequence fillCache; // Program to update the cache by using the offset indices to read from the remote buffer then scatter to the cache. This program invalidates the indices.
};

struct SoftwareCacheBenchmark :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  SoftwareCacheBenchmark() {}
  virtual ~SoftwareCacheBenchmark() {}

  // Builder interface:
  ipu_utils::RuntimeConfig getRuntimeConfig() const override;
  void build(poplar::Graph& g, const poplar::Target&) override;
  ipu_utils::ProgramManager& getPrograms() override;
  void execute(poplar::Engine& engine, const poplar::Device& device) override;

  // Tool interface:
  void setRuntimeConfig(const ipu_utils::RuntimeConfig& cfg) override;
  void addToolOptions(boost::program_options::options_description& desc) override;
  void init(const boost::program_options::variables_map& args) override;

private:
  std::unique_ptr<gather::MultiSlice> slicer;
  std::unique_ptr<FeatureCache> cache;
  ipu_utils::RuntimeConfig runConfig;
  ipu_utils::ProgramManager programs;
  std::size_t deviceFeatureCount;
  std::size_t totalFeatureCount;
  std::size_t featureDim;
  std::size_t gatherCount;
  std::size_t fetchCount;
  std::size_t iterations;
};
