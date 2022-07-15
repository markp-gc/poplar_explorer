// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "RemoteBufferBenchmark.hpp"

#include <popops/codelets.hpp>
#include <popops/Encoding.hpp>

#include <chrono>
#include <map>

RemoteBufferBenchmark::RemoteBufferBenchmark() {}
RemoteBufferBenchmark::~RemoteBufferBenchmark() {}

void RemoteBufferBenchmark::build(poplar::Graph& g, const poplar::Target&) {
  using namespace poplar::program;

  // Add codelets
  popops::addCodelets(g);

  auto dtype = getBufferType();

  // Create the remote buffer:
  ipu_utils::logger()->info("Building remote buffer with {} elements and {} repeats", bufferElements, bufferRepeats);
  ipu_utils::logger()->info("Total elements in remote buffer: {}", totalBufferSize());
  const bool optimiseMemory = false;
  buffer = g.addRemoteBuffer(
    "remote_buffer", dtype, bufferElements, bufferRepeats,
    rearrangeOnHost, optimiseMemory);

  // Create a tensor to hold contents:
  auto tensor = g.addVariable(dtype, {bufferRepeats, bufferElements},
                              poplar::VariableMappingMethod::LINEAR, "ipu_buffer");

  // Create a tensor to index all repeats of the remote buffer:
  Sequence setup;
  auto indices = g.addVariable(poplar::UNSIGNED_INT, {bufferRepeats},
                               poplar::VariableMappingMethod::LINEAR, "ipu_buffer");
  popops::iota(g, indices, 0u, setup, "create_buffer_indices");
  getPrograms().add("setup", setup);

  Sequence ipuReadFromBuffer;
  ipuReadFromBuffer.add(Copy(buffer, tensor, indices));

  auto loop = poplar::program::Repeat(iterations, ipuReadFromBuffer);
  getPrograms().add("repeat_loop", loop);
}

void RemoteBufferBenchmark::execute(poplar::Engine& engine, const poplar::Device& device) {
  // Time transfer from host to remote buffer:
  std::vector<std::vector<float>> hostBuffers(bufferRepeats, std::vector<float>(bufferElements, .5f));

  // Warm-up. For some reason this makes a difference to first host transfer:
  engine.copyToRemoteBuffer(hostBuffers[0].data(), "remote_buffer", 0);

  for (auto& v : hostBuffers) {
    std::iota(v.begin(), v.end(), 0.f);
  }

  auto startTime = std::chrono::steady_clock::now();
  for (auto i = 0u; i < bufferRepeats; i += 1) {
    engine.copyToRemoteBuffer(hostBuffers[i].data(), "remote_buffer", i);
  }
  auto endTime = std::chrono::steady_clock::now();
  auto seconds = std::chrono::duration<double>(endTime - startTime).count();

  const auto elementBytes = getBufferElementSizeInBytes();
  const double gigaBytesTransferred = 1e-9 * elementBytes * totalBufferSize();

  double hostGigaBytesPerSecond = gigaBytesTransferred / seconds;
  ipu_utils::logger()->info("Host to remote-buffer time: {}", seconds);
  ipu_utils::logger()->info("Host to Remote-buffer bandwidth: {} GB/sec", hostGigaBytesPerSecond);

  // Initialise stuff on IPU:
  const auto& progs = getPrograms();
  progs.run(engine, "setup");

  // Time transfer from remote buffer to IPU:
  startTime = std::chrono::steady_clock::now();
  progs.run(engine, "repeat_loop");
  endTime = std::chrono::steady_clock::now();

  seconds = std::chrono::duration<double>(endTime - startTime).count();
  double secondsPerTransfer = seconds / iterations;
  ipu_utils::logger()->info("Remote-buffer to IPU time: {}", secondsPerTransfer);

  double gigaBytesPerSecond = gigaBytesTransferred / secondsPerTransfer;
  ipu_utils::logger()->info("Remote-buffer to IPU bandwidth: {} GB/sec", gigaBytesPerSecond);

  // Time transfer from remote buffer to host:
  startTime = std::chrono::steady_clock::now();
  for (auto i = 0u; i < bufferRepeats; i += 1) {
    engine.copyFromRemoteBuffer("remote_buffer", hostBuffers[i].data(), i);
  }
  endTime = std::chrono::steady_clock::now();
  seconds = std::chrono::duration<double>(endTime - startTime).count();
  hostGigaBytesPerSecond = gigaBytesTransferred / seconds;
  ipu_utils::logger()->info("Remote-buffer to host time: {}", seconds);
  ipu_utils::logger()->info("Remote-buffer to host bandwidth: {} GB/sec", hostGigaBytesPerSecond);
}

std::size_t RemoteBufferBenchmark::totalBufferSize() const {
  return bufferRepeats * bufferElements;
}

poplar::Type RemoteBufferBenchmark::getBufferType() const {
  std::map<std::string, poplar::Type> convert = {
    {"half", poplar::HALF},
    {"float", poplar::FLOAT}
  };
  return interpretType(convert);
}

std::size_t RemoteBufferBenchmark::getBufferElementSizeInBytes() const {
  std::map<std::string, std::size_t> convert = {
    {"half", 2u},
    {"float", 4u}
  };
  return interpretType(convert);
}

void RemoteBufferBenchmark::addToolOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("repeats", po::value<std::size_t>(&bufferRepeats)->default_value(4096),
   "Number of 'rows' in remote buffer."
  )
  ("elements", po::value<std::size_t>(&bufferElements)->default_value(256),
   "Size of each 'row' in remote buffer"
  )
  ("iterations", po::value<std::size_t>(&iterations)->default_value(1000),
   "Number of iterations for benchmarking."
  )
  ("host-rearrange", po::value<bool>(&rearrangeOnHost)->default_value(false),
   "Rearrange remote-buffer data on host if necessary."
  )
  ("data-type", po::value<std::string>(&bufferType)->default_value("float"),
   "Element type. 'float' or 'half'."
  )
  ;
}
