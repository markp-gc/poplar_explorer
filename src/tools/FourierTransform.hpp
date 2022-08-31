// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <ipu_utils.hpp>
#include <io_utils.hpp>
#include <tool_registry.hpp>

#include <poplar/CycleCount.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

#include "../fft/complex.hpp"
#include "../fft/utils.hpp"

#include <boost/program_options.hpp>

/// Example computes a 1D Fourier transform using the Cooley-Tukey algorithm for fast
/// Fourier transforms (FFT). The the discrete Fourier transform (DFT) matrix is factorised
/// into a base matrix multiply of some dimension (the radix size) followed by 'twiddles' or
/// 'butterflies' that compute the second linear transformation in the factorisation (without
/// the computational cost of the original large DFT matrix multiply).
struct FourierTransform :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  FourierTransform() : size(0), batchSize(0), radixSize(0),
                       serialisation(0), availableMemoryProportion(-1.f) {}
  virtual ~FourierTransform() {}

  void build(poplar::Graph& graph, const poplar::Target&) override {
    popops::addCodelets(graph);
    poplin::addCodelets(graph);
    poplar::program::Sequence prog;

    poplar::program::Sequence fftSeq;
    complex::FFTBuilder builder(graph, fftSeq, "fft_builder");
    auto input = complex::ComplexTensor(graph, poplar::FLOAT, {batchSize, size}, "a");
    input.mapLinearly(graph);

    ipu_utils::logger()->info("Building FFT of input-size {} batch-size {} radix-size {}", size, batchSize, radixSize);
    builder.setAvailableMemoryProportion(availableMemoryProportion);
    auto output = builder.fft2d(input, radixSize, serialisation);
    ipu_utils::logger()->info("FFT estimated FLOP count: {}", builder.getFlopEstimate());

    auto cycleCount = poplar::cycleCount(graph, fftSeq, 0, poplar::SyncType::INTERNAL);
    prog.add(fftSeq);

    graph.createHostWrite("input_real", input.real);
    graph.createHostWrite("input_imag", input.imag);
    graph.createHostRead("output_real", output.real);
    graph.createHostRead("output_imag", output.imag);
    graph.createHostRead("cycle_count", cycleCount);

    getPrograms().add("fft", prog);
  }

  void execute(poplar::Engine& engine, const poplar::Device& device) override {
    // Create input values and write to the device:
    auto x = 0u;
    for (auto b = 0u; b < batchSize; ++b) {
      // Each item in a batch is identical:
      for (auto i = 0u; i < size; ++i) {
        x += 1;
        realData[b*size + i] = x;
        imagData[b*size + i] = x;
      }
    }

    ipu_utils::writeTensor(engine, "input_real", realData);
    ipu_utils::writeTensor(engine, "input_imag", imagData);
    if (size < 32u && batchSize < 5u) {
      for (auto b = 0u; b < batchSize; ++b) {
        ipu_utils::logger()->info("1D FFT input[{}] Re:\n{}\n", b, slice(realData, b * size, (b + 1) * size));
        ipu_utils::logger()->info("1D FFT input[{}] Im:\n{}\n", b, slice(imagData, b * size, (b + 1) * size));
      }
    }

    ipu_utils::logger()->info("Running program");
    getPrograms().run(engine, "fft");

    ipu_utils::readTensor(engine, "output_real", realData);
    ipu_utils::readTensor(engine, "output_imag", imagData);

    uint64_t cycleCount = 0u;
    ipu_utils::readScalar(engine, "cycle_count", cycleCount);
    ipu_utils::logger()->info("FFT completed in {} cycles.", cycleCount);
    if (size <= 16u && batchSize <= 8u) {
      for (auto b = 0u; b < batchSize; ++b) {
        ipu_utils::logger()->info("1D FFT result[{}] Re:\n{}\n", b, slice(realData, b * size, (b + 1) * size));
        ipu_utils::logger()->info("1D FFT result[{}] Im:\n{}\n", b, slice(imagData, b * size, (b + 1) * size));
      }
    }
  }

  void addToolOptions(boost::program_options::options_description& desc) override {
    namespace po = boost::program_options;
    desc.add_options()
    ("fft-size", po::value<std::size_t>(&size)->default_value(1024),
     "Dimension of input vector to 1D FFT.")
    ("batch-size", po::value<std::size_t>(&batchSize)->default_value(1),
     "Batch size for 1D FFT (i.e. number of input vectors).")
    ("radix-size", po::value<std::size_t>(&radixSize)->default_value(0),
     "Choose radix size (base case size at which DFT matrix-multiply is performed). The default (0) automatically "
     "sets the radix to half the input size (i.e. no FFT recursion).")
    ("serialisation-factor", po::value<std::size_t>(&serialisation)->default_value(1),
     "For FFT-2D controls how many chunks the input is split into. Higher values trade performance for reduced memory use.")
    ("available-memory-proportion", po::value<float>(&availableMemoryProportion)->default_value(-1.f),
     "Set the memory proportion available for the inner DFT matrix multiplies. Default: use the Poplar default.");
  }

  void init(const boost::program_options::variables_map& args) override {
    if (size % 2) {
      throw std::logic_error("FFT input size must be a multiple of 2.");
    }
    if (radixSize == 0) {
      radixSize = size / 2;
    }
    if (radixSize > size / 2) {
      throw std::runtime_error("Radix size can not be greater than half the input size.");
    }
    realData.resize(size * batchSize);
    imagData.resize(size * batchSize);
  }

private:
  std::size_t size;
  std::size_t batchSize;
  std::size_t radixSize;
  std::size_t serialisation;
  float availableMemoryProportion;
  std::vector<float> realData;
  std::vector<float> imagData;
};
