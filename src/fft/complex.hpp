// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/TileMapping.hpp>

namespace complex {

/// Complex tensor contains two separate tensors: one
/// for the real and imaginary parts (i.e. it imposes
/// a planar rather than interleaved storage format).
struct ComplexTensor {
  poplar::Tensor real;
  poplar::Tensor imag;
  ComplexTensor() {}

  /// Make a complex tensor from existing tensors.
  ComplexTensor(poplar::Tensor re, poplar::Tensor im)
  : real(re), imag(im) {
    if (real.shape() != imag.shape()) {
      throw std::logic_error(
        "ComplexTensor: Real and Imaginary tensors "
        "must have the same shape.");
    }
  }

  /// Make a complex tensor by specifying type and shape.
  ComplexTensor(poplar::Graph& graph,
                poplar::Type type,
                poplar::ArrayRef<std::size_t> shape,
                const std::string& debugPrefix) {
    real = graph.addVariable(type, shape, debugPrefix + "/real");
    imag = graph.addVariable(type, shape, debugPrefix + "/imag");
  }

  /// Return the shape of the complex Tensor.
  poplar::Type elementType() const { return real.elementType(); }
  std::vector<std::size_t> shape() const { return real.shape(); }
  std::size_t rank() const { return real.rank(); }
  std::size_t dim(unsigned i) const { return real.dim(i); }
  ComplexTensor transpose() const { return ComplexTensor(real.transpose(), imag.transpose()); }
  ComplexTensor slice(std::size_t begin, std::size_t end, unsigned axis) const {
    return ComplexTensor(real.slice(begin, end, axis), imag.slice(begin, end, axis));
  }

  /// Map the real and imaginary parts linearly
  /// (and separately) across tiles. 
  void mapLinearly(poplar::Graph& graph) {
    poputil::mapTensorLinearly(graph, real);
    poputil::mapTensorLinearly(graph, imag);
  }

  /// Make a new complex tensor that clones this one's real
  /// and imaginary tensors:
  ComplexTensor clone(poplar::Graph& graph, const std::string& debugPrefix, poplar::TensorCloneMethod method) const {
    return ComplexTensor(
      graph.clone(real, debugPrefix, method),
      graph.clone(imag, debugPrefix, method)
    );
  }

  /// Concatenate the real and imaginary parts as row vectors.
  /// Excepts if this is not a vector.
  poplar::Tensor asRowVectors();

  /// Concatenate the real and imaginary parts as column vectors.
  /// Excepts if this is not a vector.
  poplar::Tensor asColumnVectors();

  /// Split real and imaginary vectors by their odd and even indices.
  /// Excpets if this is not a vector.
  std::pair<ComplexTensor, ComplexTensor> splitEvenOdd();

  void multiplyInPlace(poplar::Graph& graph,
                       const ComplexTensor v,
                       poplar::program::Sequence& prog,
                       const std::string& debugPrefix="");
};

/// Element-wise multiply of two complex tensors.
ComplexTensor multiply(poplar::Graph& graph,
                        const ComplexTensor v1,
                        const ComplexTensor v2,
                        poplar::program::Sequence& prog,
                        const std::string& debugPrefix="");

/// Create copy program for both real and imaginary parts:
poplar::program::Sequence copy(const ComplexTensor& src, const ComplexTensor& dst);

/// Class to aid graph construction of a 1D Fast-Fourier-Transform.
class FFTBuilder {
  poplar::Graph &graph;
  const std::string debugPrefix;

public:
  /// Make an fft builder object.
  /// When fft1d is called the FFT program will
  /// be appended to the specified sequence.
  FFTBuilder(poplar::Graph &graph, const std::string debugName)
    : graph(graph), debugPrefix(debugName),
      availableMemoryProportion(-1.f), flopEstimate(0) {}

  /// Set the proportion of memory available for the inner DFT matrix-multiplies.
  void setAvailableMemoryProportion(float proportion) { availableMemoryProportion = proportion; }

  /// Build the compute graph that applies FFT to the given complex vector.
  /// The program will be appended to the sequence specified in construction
  /// of this object.
  ComplexTensor fft1d(poplar::program::Sequence& prog, ComplexTensor input, std::size_t radix = 0);

  /// Build a compute graph that applies a 2D-FFT to a complex matrix.
  /// The computation will be serialised into chunks determined by the
  /// serialisation factor. For large FFTs you will need to increase
  /// the serialisation factor to reduce memory consumption.
  ///
  /// Unlike fft1d the transform is computed in-place (returned tensor is
  /// the input tensor).
  ///
  /// The program will be appended to the sequence specified in
  /// construction of this object.
  ComplexTensor fft2d(poplar::program::Sequence& prog, ComplexTensor input, std::size_t radix, std::size_t serialisationFactor=1);

  /// Return the sum of FLOPs counted during all FFT building performed by this object.
  /// The counts are coarse estimates, not the exact number of FLOPs executed by the hardware.
  std::size_t getFlopEstimate() const { return flopEstimate; }

private:
  float availableMemoryProportion;
  std::size_t flopEstimate;

  // Utility functions used in construction of the FFT graph program.
  ComplexTensor multiplyMatrixByVectorBatch(poplar::program::Sequence& fftSeq, const ComplexTensor matrix, ComplexTensor vectors);
  ComplexTensor dft1d(poplar::program::Sequence& fftSeq, ComplexTensor fourierMatrix, ComplexTensor even, ComplexTensor odd);
  std::pair<ComplexTensor, ComplexTensor> splitEvenOdd(ComplexTensor input);
  ComplexTensor inverseFourierMatrices(std::size_t length, poplar::Type elemType);
  ComplexTensor twiddleCoefficients(std::size_t N, poplar::Type elemType);

  /// Internal utility that holds a graph function together with
  // input and output tensors and implements a callable interface.
  struct FunctionClosure {
    poplar::Function function;
    ComplexTensor input;
    ComplexTensor output;

    /// Apply the graph function to the specified arguments. argIn is copied into the input tensors
    /// and the result is copied to argOut. (The graph function input and output tensors are captured
    /// in the closure). Returns a graph program that executes the function call.
    poplar::program::Program operator () (ComplexTensor& argIn, ComplexTensor& argOut);
  };

  FunctionClosure fft1dMakeGraphFunction(std::size_t radix,
                                         poplar::Type elementType,
                                         const std::vector<std::size_t>& shape);
};

} // namespace complex
