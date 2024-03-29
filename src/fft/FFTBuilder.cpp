// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "FFTBuilder.hpp"

#include <cmath>
#include <map>

#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

#include <ipu_utils.hpp>
#include <io_utils.hpp>
#include "utils.hpp"

using namespace complex;

ComplexTensor FFTBuilder::multiplyMatrixByVectorBatch(
    poplar::program::Sequence& fftSeq,
    const ComplexTensor matrix, ComplexTensor vectors) {
  // The intended use of this function is to do the matmuls for all the base FFT
  // radixes in two real matmuls by batching all the components and multiplying
  // by the FFT matrix's real and imaginary parts separately, then recombining
  // the result. This is just the matrix equivalent of complex multiplication:
  // M * V = [Re(M)*Re(V) - Im(M)*Im(V)] + j[Im(M)*Re(V) + Re(M)*Im(V)]
  // but performed with only two matmuls by concatenating the vecrtors on the
  // right hand sides into a batches like this:
  // Re(M) * [ Re(V) : Im(V) ] + Im(M) * [ -Im(V) : Re(V) ]
  if (vectors.real.shape() != vectors.imag.shape()) {
    throw std::logic_error("Real and imaginary shapes must match.");
  }

  auto elemType = vectors.real.elementType();
  auto numVectors = vectors.real.dim(1);
  auto debugStr = debugPrefix + "/complex_mul_mat_vec";
  auto negIm = popops::neg(graph, vectors.imag, fftSeq, debugStr);

  // Batch together all vectors that are multiplied
  // by the real part of the matrix:
  auto realBatch = poplar::concat(vectors.real, vectors.imag, 1);

  // Batch together all vectors that are multiplied
  // by the imaginary part of the matrix:
  auto imagBatch = poplar::concat(negIm, vectors.real, 1);

  // Build the matmuls:
  ipu_utils::logger()->debug("DFT Re-Matmul shape: {} x {}", matrix.real.shape(), realBatch.shape());
  ipu_utils::logger()->debug("DFT Im-Matmul shape: {} x {}", matrix.imag.shape(), imagBatch.shape());

  poplar::OptionFlags matmulOptions;
  if (availableMemoryProportion > 0.f) {
    matmulOptions.set("availableMemoryProportion", std::to_string(availableMemoryProportion));
  }

  // Re-map the Fourier matrices here:
  auto matmulMapping = poplin::createMatMulInputLHS(graph, elemType, matrix.shape(), realBatch.shape(),
                                                    debugStr + "/fourier_matrix_mapping", matmulOptions);
  graph.setTileMapping(matrix.real, graph.getTileMapping(matmulMapping));
  graph.setTileMapping(matrix.imag, graph.getTileMapping(matmulMapping));

  poplar::Tensor partial =
    poplin::matMul(graph, matrix.real, realBatch, fftSeq,
                    elemType, debugStr + "/real_matmul", matmulOptions);

  poplin::matMulAcc(graph, partial, 1.f, matrix.imag, imagBatch, fftSeq,
                    debugStr + "/imag_matmul", matmulOptions);

  // FLOP estimates for matrix multiplies:
  flopEstimate += 2 * matrix.dim(0) * matrix.dim(1) * realBatch.dim(1) * 2;

  return ComplexTensor(partial.slice(0, numVectors, 1), partial.slice(numVectors, 2 * numVectors, 1));
}

ComplexTensor FFTBuilder::dft1d(poplar::program::Sequence& fftSeq,
                                ComplexTensor fourierMatrix,
                                ComplexTensor even, ComplexTensor odd) {
  // Combine the odd and even chunks into real and imaginary batches:
  auto real = hstack({even.real, odd.real});
  auto imag = hstack({even.imag, odd.imag});
  return multiplyMatrixByVectorBatch(fftSeq, fourierMatrix, ComplexTensor(real, imag));
}

ComplexTensor FFTBuilder::fft1d(poplar::program::Sequence& fftSeq, ComplexTensor input, std::size_t radix) {
  // Compute the 1D-FFT by decomposing the
  // Fourier matrix into an FFT of half the size
  // then compute final result using Cooley-Tukey
  // algorithm. To get the half size FT problem extract
  // odd and even, real and imaginary, coefficients:
  const auto elemType = input.real.elementType();

  // This is a 1D FFT on a batch of vectors so choose
  // the correct axis for the vector length:
  const auto batchSize = input.rank() == 1 ? 1 : input.dim(0);
  const auto fftSize = input.rank() == 1 ? input.dim(0) : input.dim(1);

  ipu_utils::logger()->debug("FFT-1D input shape: {}", input.shape());

  if (fftSize % 2) {
    throw std::logic_error("FFT size must be a multiple of 2.");
  }
  const auto splitPoint = fftSize / 2;

  complex::ComplexTensor even, odd;
  std::tie(even, odd) = input.splitEvenOdd();
  complex::ComplexTensor fftSubResult;

  // Decide whether to execute a DFT or recursiuvely apply Cooley-Tukey-FFT:
  if (radix == 0 || radix > splitPoint) {
    // Radix of zero means automatically choose radix as half size of input:
    radix = splitPoint;
  }

  if (splitPoint == radix) {
    // We have reached the specified radix size so
    // can finish by applying the DFT matrices (ending any
    // recursion):
    auto invF = inverseFourierMatrices(splitPoint, elemType);
    fftSubResult = dft1d(fftSeq, invF, even, odd);
    ipu_utils::logger()->debug("DFT-1D result shape: {}", fftSubResult.shape());
  } else {
    // Recursively construct two FFTs of half the size
    // but fold them into a single batched call to fft1d:
    auto recursiveInput = complex::ComplexTensor(
      vstack({even.real, odd.real}),
      vstack({even.imag, odd.imag})
    );
    ipu_utils::logger()->debug("Recursive FFT-1D. Sub-problem input shape: {}", recursiveInput.shape());
    auto fftResult = fft1d(fftSeq, recursiveInput, radix);
    fftSubResult = fftResult.transpose();
    ipu_utils::logger()->debug("Sub-FFT-1D result shape: {}", fftResult.shape());
  }

  // Now apply the remaining part of factorised
  // inverse Fourier matrix to get the final
  // result. First get the coefficients:
  auto w = twiddleCoefficients(fftSize, elemType);
  poputil::mapTensorLinearly(graph, w.real);
  poputil::mapTensorLinearly(graph, w.imag);

  // Reconstruct the result by slicing from columns:
  // results come out in the same even/odd order that
  // we packed the input vectors:
  auto result_even = fftSubResult.transpose().slice(0, batchSize, 0);
  auto result_odd = fftSubResult.transpose().slice(batchSize, 2*batchSize, 0);
  ipu_utils::logger()->debug("Twiddle coeff shape: {} and multiply shape: {}", w.shape(), result_odd.shape());

  // Copy the DFT results to a linear layout if there are enough
  // elements for this to make sense (this heuristic is very approximate):
  if (result_even.real.numElements() > graph.getTarget().getNumTiles()) {
    ipu_utils::logger()->debug("Re-mapping DFT result ({} > {}).",
                                result_even.real.numElements(), graph.getTarget().getNumTiles());
    auto result_even_remapped = ComplexTensor(graph, result_even.elementType(), result_even.shape(), "dft_even_remapped");
    result_even_remapped.mapLinearly(graph);
    fftSeq.add(copy(result_even, result_even_remapped));
    result_even = result_even_remapped;

    auto result_odd_remapped = ComplexTensor(graph, result_even.elementType(), result_even.shape(), "dft_even_remapped");
    result_odd_remapped.mapLinearly(graph);
    fftSeq.add(copy(result_odd, result_odd_remapped));
    result_odd = result_odd_remapped;
  }

  // Element-wise multiply odd components by coefficients:
  auto twiddlePrefix = debugPrefix + "/twiddle";
  result_odd.multiplyInPlace(graph, w, fftSeq, twiddlePrefix);
  auto tmp = result_odd;
  // FLOP estimate for complex multiply:
  flopEstimate += 6 * tmp.real.numElements();

  // Elementwise add for the twiddles (butterflies):
  poplar::Tensor lowerRe =
    popops::add(graph, result_even.real, tmp.real,
                fftSeq, twiddlePrefix + "/lower_real");
  poplar::Tensor lowerIm =
    popops::add(graph, result_even.imag, tmp.imag,
                fftSeq, twiddlePrefix + "/lower_imag");
  poplar::Tensor upperRe =
    popops::sub(graph, result_even.real, tmp.real,
                fftSeq, twiddlePrefix + "/upper_real");
  poplar::Tensor upperIm =
    popops::sub(graph, result_even.imag, tmp.imag,
                fftSeq, twiddlePrefix + "/upper_imag");

  // FLOP estimate for element-wise ops:
  flopEstimate += 4 * tmp.real.numElements();

  return ComplexTensor(
    poplar::concat(lowerRe, upperRe, 1),
    poplar::concat(lowerIm, upperIm, 1)
  );
}

poplar::program::Program
FFTBuilder::FunctionClosure::operator () (ComplexTensor& argIn,ComplexTensor& argOut) {
  poplar::program::Sequence seq;
  seq.add(copy(argIn, input));
  seq.add(poplar::program::Call(function));
  seq.add(copy(output, argOut));
  return seq;
}

FFTBuilder::FunctionClosure
FFTBuilder::fft1dMakeGraphFunction(std::size_t radix,
                                    poplar::Type elementType,
                                    const std::vector<std::size_t>& shape) {
  poplar::program::Sequence fft1dSeq;
  auto functionInput = ComplexTensor(graph, elementType,
                                      shape, debugPrefix + "/fft1d_fn_input");
  functionInput.mapLinearly(graph);
  auto functionOutput = fft1d(fft1dSeq, functionInput, radix);
  auto fft1dFunc = graph.addFunction(fft1dSeq);
  return FunctionClosure{fft1dFunc, functionInput, functionOutput};
}

ComplexTensor FFTBuilder::fft2d(poplar::program::Sequence& prog, ComplexTensor input, std::size_t radix, std::size_t serialisationFactor) {

  if (input.rank() != 2) {
    throw std::runtime_error("fft2d only supports inputs with rank 2 and batch-size 1 (i.e. a single matrix).");
  }

  if (input.dim(0) != input.dim(1)) {
    throw std::runtime_error("fft2d only supports square matrices as input.");
  }

  if (input.dim(0) % serialisationFactor) {
    std::stringstream ss;
    ss << "The number of rows in the input (" << input.dim(0)
        << ") must be divisible by the serialisation factor (" << serialisationFactor << ")";
    ipu_utils::logger()->error(ss.str());
    throw std::runtime_error(ss.str());
  }

  // Work out the size of each slice determined by the serialisationFactor:
  std::size_t rowsPerCall = input.dim(0) / serialisationFactor;

  // Make a graph function that can be called to process each slice of the input with a 1D-FFT:
  auto fft1dFunc = fft1dMakeGraphFunction(radix, input.elementType(), {rowsPerCall, input.dim(1)});
  ipu_utils::logger()->info("FFT-2D input shape: {}", input.shape());
  ipu_utils::logger()->debug("Serialised FFT input shape: {} serialisation-factor: {}", fft1dFunc.input.shape(), serialisationFactor);
  ipu_utils::logger()->debug("Serialised FFT FLOPS per call: {}", flopEstimate);

  // FLOP estimates have been accumulated by the fft1d: account for the number of calls:
  flopEstimate *= 2 * serialisationFactor;

  // 2D FFT is done in-place in two passes:

  // First pass 1D FFT for each row. Rows are processed
  // in-place in serialisationFactor chunks:
  for (auto i = 0u; i < serialisationFactor; ++i) {
    // Work on slices of the input, result slice overwrites input slice:
    auto slicedRows = input.slice(i * rowsPerCall, (i + 1) * rowsPerCall, 0);
    prog.add(fft1dFunc(slicedRows, slicedRows));
  }

  // Now repeat applying 1D-FFT to columns:
  input = input.transpose();
  for (auto i = 0u; i < serialisationFactor; ++i) {
    // Work on slices of the input, result slice overwrites input slice:
    auto slicedRows = input.slice(i * rowsPerCall, (i + 1) * rowsPerCall, 0);
    prog.add(fft1dFunc(slicedRows, slicedRows));
  }

  // We have calculated the result in-place so we must
  // transpose back before returning:
  return input.transpose();
}

ComplexTensor FFTBuilder::inverseFourierMatrices(
    std::size_t length, poplar::Type elemType) {
  const double twoPi_over_length = (2.0L / length) * 3.141592653589793238462643383279502884L;
  std::vector<float> real(length * length, 0.f);
  std::vector<float> imag(length * length, 0.f);
  for (std::size_t row = 0; row < length; ++row) {
    for (std::size_t col = 0; col < length; ++col) {
      real[row * length + col] = std::cos(twoPi_over_length * col * row);
      imag[row * length + col] = -std::sin(twoPi_over_length * col * row);
    }
  }

  auto reInvF =
    graph.addConstant<float>(elemType, {length, length}, real);
  auto imInvF =
    graph.addConstant<float>(elemType, {length, length}, imag);

  return ComplexTensor(reInvF, imInvF);
}

ComplexTensor FFTBuilder::twiddleCoefficients(std::size_t N, poplar::Type elemType) {
  // Return the complex coeeficients that recombine the partial results
  // of the FFT (I.e. coefficients that appear in left hand side of the
  // inverse Fourier matrix's FFT factorization).
  if (N % 2) {
    throw std::logic_error("FFT size must be a multiple of 2.");
  }
  auto baseSize = N / 2;
  const double s = ((2.0L * (N-1)) / N) * 3.141592653589793238462643383279502884L;
  std::vector<float> real(baseSize, 0.f);
  std::vector<float> imag(baseSize, 0.f);

  for (auto n = 0u; n < baseSize; ++n) {
    real[n] = std::cos(s * n);
    imag[n] = std::sin(s * n);
  }

  return ComplexTensor(
    graph.addConstant<float>(elemType, {baseSize}, real),
    graph.addConstant<float>(elemType, {baseSize}, imag)
  );
}
