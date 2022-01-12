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

/// Exampklel computes a 1D Fourier transform using the Cooley-Tukey algorithm for fast
/// Fourier transforms (FFT). The the discrete Fourier transform (DFT) matrix is factorised
/// into a base matrix multiply of some dimension (the radix size) followed by 'twiddles' or
/// 'butterflies' that compute the second linear transformation in the factorisation (without
/// the computational cost of the original large DFT matrix multiply).
struct FourierTransform :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  FourierTransform() {}
  virtual ~FourierTransform() {}

  ipu_utils::RuntimeConfig getRuntimeConfig() const override { return runConfig; }

  void build(poplar::Graph& graph, const poplar::Target&) override {
    popops::addCodelets(graph);
    poplin::addCodelets(graph);
    poplar::program::Sequence prog;

    poplar::program::Sequence fftSeq;
    complex::FFTBuilder builder(graph, fftSeq, "fft_builder");
    auto input = complex::ComplexTensor(graph, poplar::FLOAT, {size}, "a");
    input.mapLinearly(graph);
    auto output = builder.fft1d(input);

    auto cycleCount = poplar::cycleCount(graph, fftSeq, 0, poplar::SyncType::INTERNAL);
    prog.add(fftSeq);

    graph.createHostWrite("input_real", input.real);
    graph.createHostWrite("input_imag", input.imag);
    graph.createHostRead("output_real", output.real);
    graph.createHostRead("output_imag", output.imag);
    graph.createHostRead("cycle_count", cycleCount);

    programs.add("fft", prog);
  }

  ipu_utils::ProgramManager& getPrograms() override { return programs; }

  void execute(poplar::Engine& engine, const poplar::Device& device) override {
    // Create input values and write to the device:
    auto step = 1.f / realData.size();
    for (auto i = 0u; i < realData.size(); ++i) {
      auto x = i * step;
      realData[i] = x * x;
      imagData[i] = (1 - x) * x;
    }
    ipu_utils::writeTensor(engine, "input_real", realData);
    ipu_utils::writeTensor(engine, "input_imag", imagData);
    if (size < 32u) {
      ipu_utils::logger()->info("1D FFT input Re:\n{}\n", realData);
      ipu_utils::logger()->info("1D FFT input Im:\n{}\n", imagData);
    }

    ipu_utils::logger()->info("Running program");
    programs.run(engine, "fft");

    ipu_utils::readTensor(engine, "output_real", realData);
    ipu_utils::readTensor(engine, "output_imag", imagData);

    uint64_t cycleCount = 0u;
    ipu_utils::readScalar(engine, "cycle_count", cycleCount);
    ipu_utils::logger()->info("1D FFT of size {} completed in {} cycles.", size, cycleCount);
    if (size < 32u) {
      ipu_utils::logger()->info("1D FFT result Re:\n{}\n", realData);
      ipu_utils::logger()->info("1D FFT result Im:\n{}\n", imagData);
    }
  }

  void addToolOptions(boost::program_options::options_description& desc) override {
    namespace po = boost::program_options;
    desc.add_options()
    ("fft-size", po::value<std::size_t>(&size)->default_value(1024),
     "Dimension of input vector to 1D FFT.");
  }

  void setRuntimeConfig(const ipu_utils::RuntimeConfig& cfg) override { runConfig = cfg; };

  void init(const boost::program_options::variables_map& args) override {
    realData.resize(size);
    imagData.resize(size);
  }

private:
  std::size_t size;
  std::vector<float> realData;
  std::vector<float> imagData;
  ipu_utils::RuntimeConfig runConfig;
  ipu_utils::ProgramManager programs;
};
