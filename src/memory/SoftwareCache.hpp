// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <ipu_utils.hpp>
#include <io_utils.hpp>

#include "gather.hpp"
#include "scatter.hpp"

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

  void build(poplar::Graph& computeGraph, poplar::Graph& ioGraph, bool optimiseCopyMemoryUse = true) {
    using namespace poplar::program;

    ipu_utils::logger()->info("Cache '{}': Building cache of {} lines of size {}.", name, totalCacheLines, cacheLineSize);

    // Create remote buffer for the feature store:
    ipu_utils::logger()->info("Cache '{}': Building remote buffer with {} rows/lines", name, cacheableSetSize);
    remoteFeatures = computeGraph.addRemoteBuffer(getRemoteBufferName(), dataType, cacheLineSize, cacheableSetSize);

    // Resident set lives on the compute tiles:
    residentSet = popops::createSliceableTensor(computeGraph, dataType, {totalCacheLines, cacheLineSize}, {0}, {1}, {}, {}, name + "/resident_set");
    cacheReadProg.add(residentSet.buildRead(computeGraph, optimiseCopyMemoryUse));

    // The variables used for remote buffer fetches need to live on the IO tiles:
    remoteFetchOffsets = ioGraph.addVariable(poplar::UNSIGNED_INT, {fetchCount}, poplar::VariableMappingMethod::LINEAR, "rb_fetch_offsets");

    // Scattering data from the fetch buffer into the resident set happens on the compute tiles:
    scatter::MultiUpdate scatterToCache(name + "/scatter_to_cache", residentSet, fetchCount, false);
    scatterToCache.plan(computeGraph);
    cacheScatterOffsets = scatterToCache.createIndices(computeGraph, "");

    // We need two "fetch buffers". One on the IO tiles to receive from the
    // remote buffer and a duplicate on the compute tiles:
    auto fetchBuffer = scatterToCache.createSource(computeGraph, "compute_fetch_buffer");
    auto ioFetchBuffer = ioGraph.addVariable(fetchBuffer.elementType(), fetchBuffer.shape(), "io_fetch_buffer");
    poputil::mapTensorLinearly(ioGraph, ioFetchBuffer);

    offsetStreamSequence.add(remoteFetchOffsets.buildWrite(ioGraph, optimiseCopyMemoryUse));
    offsetStreamSequence.add(cacheScatterOffsets.buildWrite(computeGraph, optimiseCopyMemoryUse));

    // The fetch program will read from the remote buffer into the
    // fetch buffer and then scatter from the fetch buffer into the cache:
    const std::vector<std::size_t> fetchShape = {fetchCount, cacheLineSize};
    ipu_utils::logger()->info("Cache '{}': Building cache fetch program (fetches {} lines)", name, fetchCount);
    Sequence ioReadRemoteBuffer;
    ioReadRemoteBuffer.add(Copy(remoteFeatures, ioFetchBuffer.reshape(fetchShape), remoteFetchOffsets, name + "/copy_rb_features_to_io_tiles"));
    // ioReadRemoteBuffer.add(WriteUndef(remoteFetchOffsets, name + "/unlive_rb_feature_offsets"));

    ipu_utils::logger()->info("Cache '{}': Building update (scatter {} lines from fetchbuffer into residentSet).", name, fetchCount);

    // Before we can scatter to the full cache we need to move the
    // fetched data from the IO tiles to a temporary buffer on the
    // compute tiles:
    readMemoryProg = Sequence();
    readMemoryProg.add(ioReadRemoteBuffer);

    // Program that copies between IO and compute tiles:
    cacheExchangeProg = poplar::program::Sequence();
    cacheExchangeProg.add(poplar::program::Copy(ioFetchBuffer, fetchBuffer));
    // cacheExchangeProg.add(WriteUndef(fetchBuffer, name + "/unlive_fetch_buffer"));
    // cacheExchangeProg.add(WriteUndef(cacheScatterOffsets, name + "/unlive_fetch_buffer_indices"));

    // Now build the scatter program:
    updateResidentSetProg = Sequence();
    scatterToCache.createProgram(computeGraph, fetchBuffer, cacheScatterOffsets, updateResidentSetProg);
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

  // This program reads new cache lines from the remote buffer:
  poplar::program::Sequence readMemoryProg;

  // Progam that exchanges cache lines from IO tiles to compute tiles:
  poplar::program::Sequence cacheExchangeProg;

  // This program scatters the cache lines that were fetched
  // from the IO tiles into the resident set:
  poplar::program::Sequence updateResidentSetProg;

  // Program to read back the entire cache to the host (mainly intended for debugging):
  poplar::program::Sequence cacheReadProg;
};
