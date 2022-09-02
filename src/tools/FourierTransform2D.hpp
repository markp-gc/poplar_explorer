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

/// Example computes a 2D Fourier transform using the Cooley-Tukey algorithm for fast
/// Fourier transforms (FFT). Then discrete Fourier transform (DFT) matrix is factorised
/// into a base matrix multiply of some dimension (the radix size) followed by 'twiddles' or
/// 'butterflies' that compute the second linear transformation in the factorisation (without
/// the computational cost of the original large DFT matrix multiply).
struct FourierTransform2D :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  FourierTransform2D() {}
  virtual ~FourierTransform2D() {}

  void build(poplar::Graph& graph, const poplar::Target&) override {
    popops::addCodelets(graph);
    poplin::addCodelets(graph);
    poplar::program::Sequence prog;

    poplar::program::Sequence fftSeq;
    complex::FFTBuilder builder(graph, fftSeq, "fft_builder");
    // keep this. 1d input
    auto input = complex::ComplexTensor(graph, poplar::FLOAT, {batchSize, size}, "a");
    //ipu_utils::logger()->info("Shape: {}", input.imag.shape());   
    auto inputMatrixX = complex::ComplexTensor(graph, poplar::FLOAT, {batchSize, size, size}, "b");

    input.mapLinearly(graph);
    
    ipu_utils::logger()->info("Building FFT of input-size {} batch-size {} radix-size {}", size, batchSize, radixSize);

    // inputMatrix.mapLinearlyy(graph);

    auto graphFunctionOutput = builder.fft1d(input, radixSize);
    auto graphFunc = graph.addFunction(builder.getProgram());

    // Call graph function in a loop:

    auto inputMatrix = complex::ComplexTensor(
      inputMatrixX.real.dimShuffle({1,0,2}),
      inputMatrixX.imag.dimShuffle({1,0,2})
    );
   
    ipu_utils::logger()->info("Input Shape: {}", inputMatrix.imag.shape());   


    for (int n = 0u; n < size; n++) {
      
      //slice rows of input matrix
      auto inputSliceReal = inputMatrix.real.slice(n, n+1, 0);
      auto inputSliceImag = inputMatrix.imag.slice(n, n+1, 0);
      auto inputSlice = complex::ComplexTensor(inputSliceReal, inputSliceImag);
      // Map input slice within loop so whole matrix isn't remapped each time
      inputSlice.mapLinearly(graph);

      auto outputSliceReal = inputMatrix.real.slice(n, n+1, 0);
      auto outputSliceImag = inputMatrix.imag.slice(n, n+1, 0);
      auto outputSlice = complex::ComplexTensor(outputSliceReal, outputSliceImag);

      // 1. Add program to sequence Copy input slice from matrix into input
      // 2. call graph function
      // 3. Add program to sequence to copy output to slice of result matrix (input and result
      prog.add(poplar::program::Copy(inputSlice.imag, input.imag));
      prog.add(poplar::program::Copy(inputSlice.real, input.real));
      prog.add(poplar::program::Call(graphFunc));
      prog.add(poplar::program::Copy(graphFunctionOutput.imag, outputSlice.imag));
      prog.add(poplar::program::Copy(graphFunctionOutput.real, outputSlice.real));
    }

    auto inputRealTranspose = inputMatrix.real.dimShuffle({2,1,0});
    auto inputImagTranspose = inputMatrix.imag.dimShuffle({2,1,0});
  
    for (int n = 0u; n < size; n++) {

      //slice rows of transposed input matrix
      auto inputSliceReal = inputRealTranspose.slice(n, n+1, 0);
      auto inputSliceImag = inputImagTranspose.slice(n, n+1, 0);
      auto inputSlice = complex::ComplexTensor(inputSliceReal, inputSliceImag);
      // Map input slice within loop so whole matrix isn't remapped each time
      inputSlice.mapLinearly(graph);

      auto outputSliceReal = inputRealTranspose.slice(n, n+1, 0);
      auto outputSliceImag = inputImagTranspose.slice(n, n+1, 0);
      auto outputSlice = complex::ComplexTensor(outputSliceReal, outputSliceImag);
      
      prog.add(poplar::program::Copy(inputSlice.imag, input.imag));
      prog.add(poplar::program::Copy(inputSlice.real, input.real));
      prog.add(poplar::program::Call(graphFunc));
      prog.add(poplar::program::Copy(graphFunctionOutput.imag, outputSlice.imag));
      prog.add(poplar::program::Copy(graphFunctionOutput.real, outputSlice.real));
    }
    
    ipu_utils::logger()->info("FFT estimated FLOP count: {}", builder.getFlopEstimate());
    auto cycleCount = poplar::cycleCount(graph, prog, 0, poplar::SyncType::INTERNAL);

    auto outputMatrix = complex::ComplexTensor(
      inputMatrix.real.dimShuffle({1,0,2}),
      inputMatrix.imag.dimShuffle({1,0,2}));

    graph.createHostWrite("input_real", inputMatrixX.real);
    graph.createHostWrite("input_imag", inputMatrixX.imag);
    graph.createHostRead("output_real", outputMatrix.real);
    graph.createHostRead("output_imag", outputMatrix.imag);
    
    graph.createHostRead("cycle_count", cycleCount);

    getPrograms().add("fft", prog);
    
  }

  void execute(poplar::Engine& engine, const poplar::Device& device) override {
    // Create input values and write to the device:
    // auto step = 1.f / (size*size);
      // Each item in a batch is identical:
    auto x = 0u;
    for (auto b = 0u; b < batchSize; ++b) {
      for (auto i = 0u; i < (size*size); ++i) {
        // auto x = i * step;
        x += 1;
        realData[(b*size*size) + i] = x;
        imagData[(b*size*size) + i] = x;
      }
    }

    ipu_utils::writeTensor(engine, "input_real", realData);
    ipu_utils::writeTensor(engine, "input_imag", imagData);
    if (size < 8u && batchSize < 4u) {
      for (auto b = 0u; b < batchSize; ++b) {
        ipu_utils::logger()->info("FFT input[{}] Re:\n{}\n", b, slice(realData, b * size * size, (b + 1) * size * size));
        ipu_utils::logger()->info("FFT input[{}] Im:\n{}\n", b, slice(imagData, b * size * size, (b + 1) * size * size));
      }
    }
    
    ipu_utils::logger()->info("Running program");
    getPrograms().run(engine, "fft");

    ipu_utils::readTensor(engine, "output_real", realData);
    ipu_utils::readTensor(engine, "output_imag", imagData);

    uint64_t cycleCount = 0u;
    ipu_utils::readScalar(engine, "cycle_count", cycleCount);
    ipu_utils::logger()->info("FFT completed in {} cycles.", cycleCount);
    if (size < 8u && batchSize < 4u) {
      for (auto b = 0u; b < batchSize; ++b) {
        auto sliceOutputReal = slice(realData, b * size * size, (b + 1) * size * size);
        auto sliceOutputImag = slice(imagData, b * size * size, (b + 1) * size * size);
        ipu_utils::logger()->info("FFT result[{}] Re:\n{}\n", b, sliceOutputReal);
        ipu_utils::logger()->info("FFT result[{}] Im:\n{}\n", b, sliceOutputImag);
      
      }
    }
  }

  void addToolOptions(boost::program_options::options_description& desc) override {
    namespace po = boost::program_options;
    desc.add_options()
    ("fft-size", po::value<std::size_t>(&size)->default_value(1024),
     "Size of square input matrix to 2D FFT.")
    ("batch-size", po::value<std::size_t>(&batchSize)->default_value(1),
     "Batch size for FFT (i.e. number of input vectors).")
    ("radix-size", po::value<std::size_t>(&radixSize)->default_value(0),
     "Choose radix size (base case size at which DFT matrix-multiply is performed). The default (0) automatically "
     "sets the radix to half the input size (i.e. no FFT recursion).");
  }

  void init(const boost::program_options::variables_map& args) override {
    if (size % 2) {
      throw std::logic_error("FFT input size must be a multiple of 2.");
    }
    if (radixSize == 0) {
      radixSize = size / 2;
    }
    realData.resize(size * size * batchSize);
    imagData.resize(size * size * batchSize);
  }

private:
  std::size_t size;
  std::size_t batchSize;
  std::size_t radixSize;
  std::vector<float> realData;
  std::vector<float> imagData;
};
