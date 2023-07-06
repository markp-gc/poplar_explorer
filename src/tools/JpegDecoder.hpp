// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <ipu_utils.hpp>
#include <io_utils.hpp>
#include <tool_registry.hpp>

#include <poputil/TileMapping.hpp>
#include <popops/Fill.hpp>
#include <popops/codelets.hpp>

#define PRINT_DEBUG_MSGS
#include "jpeg/debug_print.hpp"
#include <jpeg/jpeg.hpp>

// Load bytes from a file into a vector. No attempt at error checking.
std::vector<std::uint8_t> fileToBytes(const std::string& inFile) {
  std::ifstream ifs(inFile, std::ios::in | std::ios::binary);
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

void writeImage(const std::string& outFile, const std::vector<std::uint8_t>& bytes, std::size_t width, std::size_t height, bool isColour) {
  if (width * height >= bytes.size()) {
    throw std::runtime_error("Invalid width and height for given number of bytes.");
  }

  // Write a PPM or PGM image:
  auto suffix = isColour ? ".ppm" : ".pgm";
  std::size_t components = isColour ? 3 : 1;
  auto fileName = outFile + suffix;
  std::ofstream file(fileName, std::ios::binary);
  file << "P" << (isColour ? 6 : 5) << "\n";
  file << width << " " << height << "\n255\n";
  file.write((const char*)bytes.data(), width * height * components);
  ipu_utils::logger()->info("Saved image '{}'", fileName);
}

std::tuple<std::vector<std::uint8_t>, std::vector<std::uint8_t>, std::size_t, std::size_t, bool>
cpuJpegDecode(const std::string& inFile, const std::string& outFile, std::uint32_t heapSizeInBytes) {
  auto inBytes = fileToBytes(inFile);
  ipu_utils::logger()->info("CPU decoder: read {} bytes from '{}'", inBytes.size(), inFile);

  std::vector<std::uint8_t> heap(heapSizeInBytes);
  Allocator alloc(heap.data(), heap.size());

  Jpeg::Decoder::Context context;
  Jpeg::Decoder decoder(context, alloc, inBytes.data(), inBytes.size());
  if (decoder.GetResult() != Jpeg::Decoder::OK) {
    throw std::runtime_error("Error in CPU JPEG decoding.");
  }

  std::vector<std::uint8_t> decodedByteStorage(decoder.GetImageSize());
  std::copy(decoder.GetImage(), decoder.GetImage() + decoder.GetImageSize(), decodedByteStorage.begin());
  ipu_utils::logger()->info("CPU decoder: decoded image size {} bytes", decodedByteStorage.size());

  writeImage(outFile + "_cpu", decodedByteStorage, decoder.GetWidth(), decoder.GetHeight(), decoder.IsColor());

  return {inBytes, decodedByteStorage, decoder.GetWidth(), decoder.GetHeight(), decoder.IsColor()};
}

/// Experimental in IPU tile JPEG decoder.
struct JpegDecoder :
  public ipu_utils::BuilderInterface, public ToolInterface
{
  JpegDecoder() : input("jpeg_input_buffer"), output("jpeg_output_buffer") {}
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
    // Run CPU decoder, save results, and store size info for use in graph building:
    std::tie(inputBuffer, outputBuffer, decodedWidth, decodedHeight, decodedIsColor) =
      cpuJpegDecode(inFile, outFile, tileHeapSizeKiB * 1024);

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
    output.buildTensor(graph, poplar::UNSIGNED_CHAR, {outputBuffer.size()});

    graph.connect(decodeVert["buffer"], input);
    graph.connect(decodeVert["heap"], heap);
    graph.connect(decodeVert["result"], output);

    graph.setTileMapping(input, 0u);
    graph.setTileMapping(output, 0u);
    graph.setTileMapping(heap, 0u);
    graph.setTileMapping(decodeVert, 0u);

    auto uploadJpeg = input.buildWrite(graph, true);
    auto downloadResult = output.buildRead(graph, true);

    poplar::program::Sequence prog;
    prog.add(uploadJpeg);
    popops::fill(graph, heap, prog, 0u, "zero_heap");
    prog.add(poplar::program::Execute(decodeCs));
    prog.add(downloadResult);

    getPrograms().add("decode", prog);
  }

  void execute(poplar::Engine& engine, const poplar::Device& device) override {
    // Clear the output buffer so we don't get the right result by accident:
    std::fill(outputBuffer.begin(), outputBuffer.end(), 0);

    // Run IPU decoder:
    input.connectWriteStream(engine, inputBuffer);
    output.connectReadStream(engine, outputBuffer);
    getPrograms().run(engine, "decode");

    writeImage(outFile + "_ipu", outputBuffer, decodedWidth, decodedHeight, decodedIsColor);
  }

  std::string inFile;
  std::string outFile;

  std::size_t decodedWidth, decodedHeight;
  bool decodedIsColor;

  std::vector<std::uint8_t> inputBuffer;
  std::vector<std::uint8_t> outputBuffer;
  ipu_utils::StreamableTensor input;
  ipu_utils::StreamableTensor output;

  std::uint32_t tileHeapSizeKiB;
};
