# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# This script generates output for 1D and 2D FFTs for comparing against the IPU implementation.

import numpy as np
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='1D-FFT Test Program.')
  parser.add_argument('--fft-size', default=8, type=int,
                      help='Length of (complex) input vector for 1D FFT (C2C).')
  args = parser.parse_args()

  N = args.fft_size
  batchSize = args.fft_size
  input = np.ones(batchSize * N) * [0 + 0J]
  step = 1.0 / (N * N)

  x = 0
  for b in np.arange(batchSize):
      for i in np.arange(N):
          x += 1
          input.real[b * N + i] = x
          input.imag[b * N + i] = x

  input_matrix = np.reshape(input, (batchSize, N))
  input_matrix = np.reshape(input_matrix, (batchSize, N))

  print("Input Real:\n", input_matrix.real)
  print("Input Imag:\n", input_matrix.imag)

  output = np.fft.fft(input_matrix)
  print("1D batched output real:\n", output.real)
  print("1D batched output imag:\n", output.imag)

  output2 = np.fft.fft2(input_matrix)
  print("2D output real:\n", output2.real)
  print("2D output imag:\n", output2.imag)
