#pragma once

#include "../ipu_utils.hpp"
#include "../io_utils.hpp"
#include "../tool_registry.hpp"

#include <boost/program_options.hpp>

struct RemoteBufferBenchmark :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  RemoteBufferBenchmark();
  virtual ~RemoteBufferBenchmark();

  // Builder interface:
  ipu_utils::RuntimeConfig getRuntimeConfig() const override;
  void build(poplar::Graph& g, const poplar::Target&) override;
  ipu_utils::ProgramManager& getPrograms() override;
  void execute(poplar::Engine& engine, const poplar::Device& device) override;

  // Tool interface:
  void setRuntimeConfig(const ipu_utils::RuntimeConfig& cfg) override;
  void addToolOptions(boost::program_options::options_description& desc) override;

private:
  std::size_t totalBufferSize() const;
  poplar::Type getBufferType() const;
  std::size_t getBufferElementSizeInBytes() const;

  template <typename T>
  T interpretType(const std::map<std::string, T>& convert) const {
    if (convert.count(bufferType) == 0) {
      throw std::runtime_error("Data type unsupported by benchmark: " + bufferType);
    }
    return convert.at(bufferType);
  }

  ipu_utils::RuntimeConfig runConfig;
  boost::program_options::variables_map args;
  ipu_utils::ProgramManager programs;
  poplar::RemoteBuffer buffer;
  std::string bufferType;
  std::size_t bufferRepeats;
  std::size_t bufferElements;
  std::size_t iterations;
  bool rearrangeOnHost;
};
