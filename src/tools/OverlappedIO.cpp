// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "OverlappedIO.hpp"

#include <pvti/pvti.hpp>

static pvti::TraceChannel traceChannel{"streams"};

class StreamInCallback : public poplar::StreamCallback {
public:
  StreamInCallback(const std::vector<float>& data) : data_(data), completeCount(0) {};

  void fetch(void *p) {
    std::memcpy(p, &data_[0], data_.size() * sizeof(data_[0]));
    pvti::Tracepoint::begin(&traceChannel, "stream_in_data_ready");
  }

  poplar::StreamCallback::Result prefetch(void* p) {
    fetch(p);
    return poplar::StreamCallback::Result::Success;
  }

  void complete() {
    pvti::Tracepoint::end(&traceChannel, "stream_in_data_ready");
    completeCount += 1;
  }
  void invalidatePrefetched() {
    pvti::Tracepoint::end(&traceChannel, "stream_in_data_ready");
  }

private:
  const std::vector<float>& data_;
  std::size_t completeCount;
};

class StreamOutCallback : public poplar::StreamCallback {
public:
  StreamOutCallback(std::vector<float>& data) : data_(data) {};

  void fetch(void *p) {
    std::memcpy(&data_[0], p, data_.size() * sizeof(data_[0]));
    pvti::Tracepoint::begin(&traceChannel, "stream_out_data_ready");
  }

  poplar::StreamCallback::Result prefetch(void* p) {
    fetch(p);
    return poplar::StreamCallback::Result::Success;
  }

  void complete() {
    pvti::Tracepoint::end(&traceChannel, "stream_out_data_ready");
  }

  void invalidatePrefetched() {
    pvti::Tracepoint::end(&traceChannel, "stream_out_data_ready");
  }

private:
  std::vector<float>& data_;
};

void OverlappedIO::addToolOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("num-io-tiles", po::value<unsigned>(&numTilesForIO)->default_value(0u),
    "Number of tiles to use for IO. Defaults to the minimum number.")
  ("work-size", po::value<std::size_t>(&sizePerWorker)->default_value(128u),
    "Amount of work to give each worker thread.")
  ("iterations", po::value<std::size_t>(&numIterations)->default_value(100u),
    "Number of iterations of the IO pipeline.")
  ;
}

void OverlappedIO::init(const boost::program_options::variables_map& args) {}

void OverlappedIO::build(poplar::Graph& graph, const poplar::Target& target) {
  popops::addCodelets(graph);
  graph.addCodelets("../src/codelets/simple.cpp");

  // Get two disjoint sets of tiles to use for compute and IO:
  const auto numTotalTiles = target.getNumTiles();
  const auto minIOTiles = gcl::getMinIoTiles(graph);
  numTilesForIO = std::max(minIOTiles, numTilesForIO);

  ioTiles = gcl::perIPUTiles(graph, 0, numTilesForIO);
  numWorkerContexts = target.getNumWorkerContexts();
  sizePerWorker = 128;

  computeTiles = gcl::perIPUTiles(graph, numTilesForIO, numTotalTiles - numTilesForIO);
  numComputeTiles = computeTiles.size();

  ipu_utils::logger()->info("Minimum number of tiles that can be used for IO: {}", minIOTiles);
  ipu_utils::logger()->info("Number of tiles used for IO: {}", numTilesForIO);
  ipu_utils::logger()->info("Number of tiles used for compute: {}", numComputeTiles);
  ipu_utils::logger()->info("numWorkerContexts: {}", numWorkerContexts);
  ipu_utils::logger()->info("sizePerWorker: {}", sizePerWorker);

  const auto elementType = poplar::FLOAT;

  // Create two virtual graphs from the two disjoint sets of tiles.
  // These graphs can run in parallel:
  auto compute_graph = graph.createVirtualGraph(computeTiles);
  poplar::ComputeSet cs_compute_0;
  poplar::Tensor compute_tensor_in, compute_tensor_out;
  std::tie(cs_compute_0, compute_tensor_in, compute_tensor_out) = buildComputeGraph(compute_graph, elementType);

  auto io_graph = graph.createVirtualGraph(ioTiles);
  poplar::Tensor io_tensor_in, io_tensor_out;
  std::tie(io_tensor_in, io_tensor_out) = buildIOGraph(io_graph, target, elementType);

  // Create the input and output data FIFOs:
  auto stream_in = io_graph.addHostToDeviceFIFO(
    "stream_in",
    elementType,
    numTransferInElements,
    poplar::ReplicatedStreamMode::REPLICATE,
    {{"bufferingDepth", "4"}});

  auto stream_out = io_graph.addDeviceToHostFIFO(
    "stream_out",
    elementType,
    numTransferOutElements);

  // Create the copy programs:
  const bool doNotOutline = true;

  // These copies do the transfers between the host and the IO tiles:
  auto program_host_exchange_in = poplar::program::Copy(stream_in, io_tensor_in);
  auto program_host_exchange_out = poplar::program::Copy(io_tensor_out, stream_out);

  // These copies exchange data between the IO tiles and the compute tiles:
  auto program_internal_exchange_in = poplar::program::Copy(
    io_tensor_in.flatten(),
    compute_tensor_in.flatten(),
    doNotOutline);
  auto program_internal_exchange_out = poplar::program::Copy(
    compute_tensor_out.flatten(),
    io_tensor_out.flatten(),
    doNotOutline);

  // Create the complete graph program:

  // Execute the compute sets for the compute tiles:
  auto program_compute = poplar::program::Execute(cs_compute_0);

  // This is the main pipeline sequence that runs in a loop after ramp-up:
  auto main_sequence = poplar::program::Sequence {
    program_host_exchange_out,
    program_host_exchange_in,
    program_compute,
    program_internal_exchange_out,
    program_internal_exchange_in
  };

  // Create the pipeline (first ramp up, then run main-sequence loop, then ramp-down):
  auto pipeline = poplar::program::Sequence{
    // Ramp up/Priming stage
    // I/O input buffer = data[0]
    program_host_exchange_in,
    // Compute input buffer = data[0]
    program_internal_exchange_in,
    // [
    //   Compute output buffer = output[0],
    //   I/O input buffer = data[1]
    // ]
    poplar::program::Sequence{program_compute, program_host_exchange_in},
    // I/O output buffer = output[0]
    program_internal_exchange_out,
    // Compute input buffer = data[1]
    program_internal_exchange_in,
    // At this point the state is:
    // [
    //   I/O input = data[1], (stale)
    //   Compute input = data[1],
    //   Compute output = data[0],  (stale)
    //   I/O output = output[0]
    // ]
    poplar::program::Repeat(numIterations - 2, main_sequence),
    // At this point the state is:
    // [
    //   I/O input = data[-1], (stale)
    //   Compute input = data[-1],
    //   Compute output = output[-2], (stale)
    //   I/O output = output[-2],
    // ]
    // Next steps:
    // [
    //   outfeed output[-2],
    //   Compute output = output[-1]
    // ]
    poplar::program::Sequence{program_host_exchange_out, program_compute},
    // I/O output = output[-1]
    program_internal_exchange_out,
    // outfeed output[-1]
    program_host_exchange_out
  };

  // Register the completed pipeline program with the program manager:
  getPrograms().add("io_pipeline", pipeline);
}

void OverlappedIO::execute(poplar::Engine& engine, const poplar::Device& device) {

  ipu_utils::logger()->info("Num compute tiles: {}", numComputeTiles);
  std::vector<float> host_in(numTransferInElements);
  for (unsigned i = 0; i < host_in.size(); ++i) {
      host_in[i] = i/sizePerWorker;
  }
  for (unsigned i = 0; i < std::min(3ul*sizePerWorker,host_in.size()); ++i) {
      std::cout << "host_in[" << i << "] = " << host_in[i] << std::endl;
  }
  std::cout << "..." << std::endl;

  engine.connectStreamToCallback(
      "stream_in",
      0,
      std::make_unique<StreamInCallback>(host_in));

  std::vector<float> host_out(numTransferOutElements);
  for (unsigned i = 0; i < host_out.size(); ++i) {
      host_out[i] = -1.0;
  }

  engine.connectStreamToCallback(
      "stream_out",
      std::make_unique<StreamOutCallback>(host_out));

  getPrograms().run(engine, "io_pipeline");

  for (unsigned i = 0; i < std::min(8ul,host_out.size()); ++i) {
      std::cout << "host_out[" << i << "] = " << host_out[i] << std::endl;
  }
  std::cout << "..." << std::endl;
}

std::tuple<poplar::ComputeSet, poplar::Tensor, poplar::Tensor>
OverlappedIO::buildComputeGraph(poplar::Graph& compute_graph, poplar::Type dtype) {
  // Construct the compute graph
  auto compute_tensor_in = compute_graph.addVariable(
    dtype,
    {numComputeTiles, numWorkerContexts, sizePerWorker},
    "compute_tensor_in");

  for (unsigned tile = 0; tile < numComputeTiles; ++tile) {
    compute_graph.setTileMapping(compute_tensor_in[tile], tile);
  }

  numTransferInElements = compute_tensor_in.numElements();
  ipu_utils::logger()->debug("numTransferInElements: {}", numTransferInElements);

  numTransferOutElements = numComputeTiles * numWorkerContexts;
  ipu_utils::logger()->debug("numTransferOutElements: {}", numTransferOutElements);
  auto compute_tensor_out = compute_graph.addVariable(
    dtype,
    {numComputeTiles, numWorkerContexts},
    "compute_tensor_out");

  for (unsigned tile = 0; tile < numComputeTiles; ++tile) {
    compute_graph.setTileMapping(compute_tensor_out[tile], tile);
  }

  auto cs_compute_0 = compute_graph.addComputeSet("cs_compute_0");
  for (unsigned tile = 0; tile < numComputeTiles; ++tile) {
    for (unsigned worker = 0; worker < numWorkerContexts; ++worker) {
      auto vertex = compute_graph.addVertex(cs_compute_0, "ComputeVertex");

      compute_graph.connect(vertex["in"], compute_tensor_in[tile][worker]);
      compute_graph.connect(vertex["out"], compute_tensor_out[tile][worker]);

      compute_graph.setTileMapping(vertex, tile);
    }
  }

  return std::make_tuple(cs_compute_0, compute_tensor_in, compute_tensor_out);
}

std::tuple<poplar::Tensor, poplar::Tensor>
OverlappedIO::buildIOGraph(poplar::Graph& io_graph, const poplar::Target& target, poplar::Type elementType) {
  // Construct the IO graph
  if ((numTransferInElements % numTilesForIO) != 0) {
      throw std::runtime_error(
          "Number of io tiles does not divide compute elements");
  }

  const auto num_elements_in_per_io_tile = numTransferInElements / numTilesForIO;
  const auto num_elements_out_per_io_tile = numTransferOutElements / numTilesForIO;

  ipu_utils::logger()->debug("num_elements_in_per_io_tile: {}", num_elements_in_per_io_tile);
  ipu_utils::logger()->debug("num_elements_out_per_io_tile: {}", num_elements_out_per_io_tile);

  if ((num_elements_in_per_io_tile + num_elements_out_per_io_tile) * target.getTypeSize(elementType) > target.getBytesPerTile()) {
    throw std::runtime_error("Too many bytes requested per io tile");
  }

  auto io_tensor_in = io_graph.addVariable(
    elementType,
    {numTilesForIO, num_elements_in_per_io_tile},
    "io_tensor_in");
  for (unsigned tile = 0; tile < numTilesForIO; ++tile) {
    io_graph.setTileMapping(io_tensor_in[tile], tile);
  }

  auto io_tensor_out = io_graph.addVariable(
      elementType,
      {numTilesForIO, num_elements_out_per_io_tile},
      "io_tensor_out");
  for (unsigned tile = 0; tile < numTilesForIO; ++tile) {
      io_graph.setTileMapping(io_tensor_out[tile], tile);
  }

  return std::make_tuple(io_tensor_in, io_tensor_out);
}
