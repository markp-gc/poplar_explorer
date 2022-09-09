// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include "complex.hpp"

/// Class to aid graph construction of a 1D Fast-Fourier-Transform.
class FFTBuilder {
  poplar::Graph &graph;
  const std::string debugPrefix;

public:
  /// Make an fft builder object.
  FFTBuilder(poplar::Graph &graph, const std::string debugName)
    : graph(graph), debugPrefix(debugName),
      availableMemoryProportion(-1.f), flopEstimate(0) {}

  /// Set the proportion of memory available for the inner DFT matrix-multiplies.
  void setAvailableMemoryProportion(float proportion) { availableMemoryProportion = proportion; }

  /// Build the compute graph that applies FFT to the given complex vector.
  /// The program will be appended to the sequence specified in construction
  /// of this object. The FFT program will be appended to the sequence /p prog.
  complex::ComplexTensor fft1d(poplar::program::Sequence& prog, complex::ComplexTensor input, std::size_t radix = 0);

  /// Build a compute graph that applies a 2D-FFT to a complex matrix.
  /// The computation will be serialised into chunks determined by the
  /// serialisation factor. For large FFTs you will need to increase
  /// the serialisation factor to reduce memory consumption.
  ///
  /// Unlike fft1d the transform is computed in-place (returned tensor is
  /// the input tensor).
  ///
  /// The program will be appended to the sequence /p prog.
  complex::ComplexTensor fft2d(poplar::program::Sequence& prog, complex::ComplexTensor input, std::size_t radix, std::size_t serialisationFactor=1);

  /// Return the sum of FLOPs counted during all FFT building performed by this object.
  /// The counts are coarse estimates, not the exact number of FLOPs executed by the hardware.
  std::size_t getFlopEstimate() const { return flopEstimate; }

private:
  float availableMemoryProportion;
  std::size_t flopEstimate;

  // Utility functions used in construction of the FFT graph program.
  complex::ComplexTensor multiplyMatrixByVectorBatch(poplar::program::Sequence& fftSeq, const complex::ComplexTensor matrix, complex::ComplexTensor vectors);
  complex::ComplexTensor dft1d(poplar::program::Sequence& fftSeq, complex::ComplexTensor fourierMatrix, complex::ComplexTensor even, complex::ComplexTensor odd);
  std::pair<complex::ComplexTensor, complex::ComplexTensor> splitEvenOdd(complex::ComplexTensor input);
  complex::ComplexTensor inverseFourierMatrices(std::size_t length, poplar::Type elemType);
  complex::ComplexTensor twiddleCoefficients(std::size_t N, poplar::Type elemType);

  /// Internal utility that holds a graph function together with
  // input and output tensors and implements a callable interface.
  struct FunctionClosure {
    poplar::Function function;
    complex::ComplexTensor input;
    complex::ComplexTensor output;

    /// Apply the graph function to the specified arguments. argIn is copied into the input tensors
    /// and the result is copied to argOut. (The graph function input and output tensors are captured
    /// in the closure). Returns a graph program that executes the function call.
    poplar::program::Program operator () (complex::ComplexTensor& argIn, complex::ComplexTensor& argOut);
  };

  FunctionClosure fft1dMakeGraphFunction(std::size_t radix,
                                         poplar::Type elementType,
                                         const std::vector<std::size_t>& shape);
};
