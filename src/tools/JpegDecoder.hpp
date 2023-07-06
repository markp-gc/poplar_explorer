// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <ipu_utils.hpp>
#include <io_utils.hpp>
#include <tool_registry.hpp>

#include <poputil/TileMapping.hpp>
#include <popops/Fill.hpp>
#include <popops/codelets.hpp>

#include <jpeg/jpeg.hpp>

// Load bytes from a file into a vector. No attempt at error checking.
std::vector<std::uint8_t> fileToBytes(const std::string& inFile) {
  std::ifstream ifs(inFile, std::ios::in | std::ios::binary);
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

std::vector<std::uint8_t> cpuJpegDecode(const std::string& inFile, const std::string& outFile) {
  auto vbytes = fileToBytes(inFile);
  ipu_utils::logger()->info("CPU decoder: read {} bytes from '{}'", vbytes.size(), inFile);

  Jpeg::Decoder::Context context;
  Jpeg::Decoder decoder(context, vbytes.data(), vbytes.size());
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

  return vbytes;
}

/// Experimental in IPU tile JPEG decoder.
struct JpegDecoder :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  JpegDecoder() : input("jpeg_input_buffer") {}
  virtual ~JpegDecoder() {}

  /// Tool interface:

  void addToolOptions(boost::program_options::options_description& desc) override {
    namespace po = boost::program_options;
    desc.add_options()
    ("in", po::value<std::string>(&inFile)->required(),
     "Input JPEG file.")
    ("out", po::value<std::string>(&outFile)->default_value("decoded"),
     "Output file name prefix for decoded image (extension will automatically be set as .pgm or .ppm).")
    ("heap", po::value<std::uint32_t>(&tileHeapSizeKiB)->default_value(32), "Size of heap in KiB for on tile dynamic allocation.")
    ;
  }

  void init(const boost::program_options::variables_map& args) override {
    // Run CPU decoder:
    inputBuffer = cpuJpegDecode(inFile, outFile);
    if (inputBuffer.empty()) {
      throw std::runtime_error("Empty input buffer.");
    }
  }

  /// Builder interface:

  void build(poplar::Graph& graph, const poplar::Target& target) override {
    popops::addCodelets(graph);
    graph.addCodelets("../src/codelets/JpegDecoder/jpeg.cpp", poplar::CodeletFileType::Auto, "-O3");

    auto heap = graph.addVariable(poplar::UNSIGNED_CHAR, {tileHeapSizeKiB * 1024}, "tile_heap");

    auto decodeCs = graph.addComputeSet("decoder");
    auto decodeVert = graph.addVertex(decodeCs, "JpegDecode");

    input.buildTensor(graph, poplar::UNSIGNED_CHAR, {inputBuffer.size()});

    graph.connect(decodeVert["buffer"], input);
    graph.connect(decodeVert["heap"], heap);

    graph.setTileMapping(input, 0u);
    graph.setTileMapping(heap, 0u);
    graph.setTileMapping(decodeVert, 0u);

    auto uploadJpeg = input.buildWrite(graph, true);

    poplar::program::Sequence prog;
    prog.add(uploadJpeg);
    popops::fill(graph, heap, prog, 0u, "zero_heap");
    prog.add(poplar::program::Execute(decodeCs));

    getPrograms().add("decode", prog);
  }

  void execute(poplar::Engine& engine, const poplar::Device& device) override {
    // Run IPU decoder:
    input.connectWriteStream(engine, inputBuffer);
    getPrograms().run(engine, "decode");
  }

  std::string inFile;
  std::string outFile;

  std::vector<std::uint8_t> inputBuffer;
  ipu_utils::StreamableTensor input;

  std::uint32_t tileHeapSizeKiB;
};
