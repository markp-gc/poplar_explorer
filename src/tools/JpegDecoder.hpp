// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <ipu_utils.hpp>
#include <io_utils.hpp>
#include <tool_registry.hpp>

#include <poputil/TileMapping.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>

#include <jpeg/jpeg.hpp>

// Load bytes from a file into a vector. No attempt at error checking.
std::vector<std::uint8_t> fileToBytes(const std::string& inFile) {
  std::ifstream ifs(inFile, std::ios::in | std::ios::binary);
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

void cpuJpegDecode(const std::string& inFile, const std::string& outFile) {
  auto vbytes = fileToBytes(inFile);
  ipu_utils::logger()->info("CPU decoder: read {} bytes from '{}'", vbytes.size(), inFile);

  Jpeg::Decoder decoder(vbytes.data(), vbytes.size(), malloc, free);
  if (decoder.GetResult() != Jpeg::Decoder::OK)
  {
    throw std::runtime_error("Error in CPU JPEG decoding.");
  }

  // Write a PPM or PGM of the decoded image:
  auto suffix = (decoder.IsColor() ? "_cpu.ppm" : "_cpu.pgm");
  auto pmFileName = outFile + suffix;
  std::ofstream file(pmFileName, std::ios::binary);
  file << "P" << (decoder.IsColor() ? 6 : 5) << "\n";
  file << decoder.GetWidth() << " " << decoder.GetHeight() << "\n255\n";
  file.write((const char*)decoder.GetImage(), decoder.GetImageSize());
  ipu_utils::logger()->info("CPU decoder: saved decoded image '{}'", pmFileName);
}

/// Experimental in IPU tile JPEG decoder.
struct JpegDecoder :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  JpegDecoder() {}
  virtual ~JpegDecoder() {}

  /// Tool interface:

  void addToolOptions(boost::program_options::options_description& desc) override {
    namespace po = boost::program_options;
    desc.add_options()
    ("in", po::value<std::string>(&inFile)->required(),
     "Input JPEG file.")
    ("out", po::value<std::string>(&outFile)->default_value("decoded"),
     "Output file name prefix for decoded image (extension will automatically be set as .pgm or .ppm).")
    ;
  }

  void init(const boost::program_options::variables_map& args) override {}

  /// Builder interface:

  void build(poplar::Graph& graph, const poplar::Target& target) override {
    graph.addCodelets("../src/codelets/JpegDecoder/jpeg.cpp", poplar::CodeletFileType::Auto, "-O3");

    auto decodeCs = graph.addComputeSet("decoder");
    auto decodeVert = graph.addVertex(decodeCs, "JpegDecode");
    graph.setTileMapping(decodeVert, 0u);

    poplar::program::Sequence prog;
    prog.add(poplar::program::Execute(decodeCs));

    getPrograms().add("decode", prog);
  }

  void execute(poplar::Engine& engine, const poplar::Device& device) override {

    // Run CPU decoder:
    cpuJpegDecode(inFile, outFile);

    // Run IPU decoder:
    getPrograms().run(engine, "decode");
  }

  std::string inFile;
  std::string outFile;
};
