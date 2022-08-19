# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# This Python example contains a reference implementation of the DFT
# algorithm that is implemented in Poplar. Can be used to check
# correctness of the algorithm against scipy's implementation.

from scipy.fft import fft
import numpy as np
from timeit import default_timer as timer
from typing import List
import argparse


def inverse_fourier_matrix(N):
  """
  It is the inverse Fourier matrix which computes
  the forward FFT (as we are performing a linear
  solve for the coefficients that express the input
  vector in the Fourier basis: i.e. solving y = Fc
  for c by forming inv(F)):
  :param N: Size of the 1D-FFT we want to solve. 
  :return: Two inverse Fourier matrices: one to solve for the Real
  and one to solve for the Imaginary part of the
  coefficients.
  """
  two_pi = 2 * np.pi
  re_inv_F = np.ones([N, N])
  im_inv_F = np.ones([N, N])
  for r, _ in enumerate(re_inv_F):
    for c, _ in enumerate(re_inv_F):
      re_inv_F[r][c] = np.cos(two_pi * c * r/N)
      im_inv_F[r][c] = -np.sin(two_pi * c * r/N)
  return re_inv_F, im_inv_F


def fft_recombine_coeffs(N):
  """
  Return the complex coeeficients that recombine the partial results
  of the FFT (I.e. coefficients that appear in left hand side of the
  inverse Fourier matrix's FFT factorization).
  :param N: Size of the 1D-FFT we want to solve.
  :return: Two real arrays containing Re and Im parts of the complex
            coefficients from the LHS of the FFT decompostion.
  """
  base_size = N // 2
  two_pi = 2 * np.pi
  re_w = np.zeros(shape=[base_size])
  im_w = np.zeros(shape=[base_size])
  for n in range(base_size):
    re_w[n] = np.cos(two_pi * n * ((N-1)/N))
    im_w[n] = np.sin(two_pi * n * ((N-1)/N))
  return re_w, im_w


def complex_multiply(re_v1, im_v1, re_v2, im_v2):
  """
  Element wise multiply of two complex vectors where each
  vector v is represented as two real arrays (Re(v) and Im(v)).
  """
  re = np.multiply(re_v1, re_v2) - np.multiply(im_v1, im_v2)
  im = np.multiply(re_v1, im_v2) + np.multiply(im_v1, re_v2)
  return re, im


def complex_matvecs_mul(re_M, im_M, re_v, im_v):
  """
  Matrix multiply of complex matrix by a list of complex vectors where
  the operands' real and imaginary parts are kept separate throughout
  and the results' real and imaginary parts are returned in separate
  vectors packed into columns of the result.

  The intended use of this function is to do the matmuls for all the base FFT
  radixes in two real matmuls by batching all the components and multiplying
  by the FFT matrix's real and imaginary parts separately, then recombining
  the result. This is just the matrix equivalent of complex multiplication:
  M * V = [Re(M)*Re(V) - Im(M)*Re(V)] + j[Im(M)*Re(V) + Re(M)*Im(V)]

  The vectors are represented by two lists of real arrays (Re(v) and Im(v)),
  and the matrix M by two matrices: Re(M) and Im(M).
  """
  if not len(re_v) == len(im_v):
    raise RuntimeError("Number of real and imaginary vectors must match.")

  # Extract the imaginary components into an np array because
  # we need to negate them in one of the expressions:
  neg_im_v = -np.vstack([*im_v])

  # Batch together all components that are multiplied by Re(M):
  re_M_batch = np.vstack(([*re_v], [*im_v])).T

  # Batch together all components that are multiplied by Im(M):
  im_M_batch = np.vstack([*neg_im_v, *re_v]).T

  # Do the matmuls, accumulating results:
  partial = np.matmul(re_M, re_M_batch)
  partial += np.matmul(im_M, im_M_batch)
  return partial[:,:len(re_v)], partial[:,len(re_v):]


def my_dft(re_v: List[np.array], im_v: List[np.array]):
  """
  Apply DFT to a batch of inputs as independent matmuls for real and
  imaginary component vectors.
  :param re_v: The real components of a batch of vectors to compute DFT on.
  :param im_v: The imaginary components of a batch of vectors to compute DFT on.
  :return: Tuple containing the batch of real and batch of imaginary component
            results. 
  """
  if not len(re_v) == len(im_v):
    raise RuntimeError("Number of real and imaginary vectors must match.")
  if len(re_v) > 0:
    fft_size = re_v[0].size
    re_inv_F, im_inv_F = inverse_fourier_matrix(fft_size)
    re_results, im_results = complex_matvecs_mul(re_inv_F, im_inv_F, re_v, im_v)
    return re_results, im_results

  return None, None


def my_fft(y):
  """
  Fast Fourier transform of a complex input vector.
  :param y: Input vector (real or complex).
  :return: The discrete Fourier transform of the input
            (complex vector).
  """
  # Compute the 1D-FFT by decomposing the
  # Fourier matrix into an FFT of half the size
  # then compute final result using Cooley-Tukey
  # algorithm. To get the half size FT problem extract
  # odd and even, real and imaginary, coefficients:
  re_y_even = y.real[0:][::2]
  re_y_odd = y.real[1:][::2]
  im_y_even = y.imag[0:][::2]
  im_y_odd = y.imag[1:][::2]

  # Apply DFT to the odd and even problems. In order
  # to perform a complex matrix multiply with only
  # real arithmetic that is easily vectoriseable
  # we batch the real and imaginary components carefully
  # so that only two real matmuls are needed in
  # the DFT:
  re_results, im_results = my_dft([re_y_even, re_y_odd], [im_y_even, im_y_odd])

  # The results come out with the same list structure as we put them in:
  re_even, re_odd = re_results[:,0], re_results[:,1]
  im_even, im_odd = im_results[:,0], im_results[:,1]

  # Now apply the remaining part of factorised
  # inverse Fourier matrix to get the final
  # result. First get the coefficients:
  re_w, im_w = fft_recombine_coeffs(y.size)

  # Element-wise multiply by coefficients:
  re_tmp, im_tmp = complex_multiply(re_w, im_w, re_odd, im_odd)

  # Elementwise add for the twiddles (butterflies):
  re_lower = re_even + re_tmp
  im_lower = im_even + im_tmp
  re_upper = re_even - re_tmp
  im_upper = im_even - im_tmp

  # Concat lower and upper parts into result:
  re_result = np.concatenate([re_lower, re_upper])
  im_result = np.concatenate([im_lower, im_upper])

  # Check result against scipy:
  result = re_result + (1j * im_result)
  return result


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='1D-FFT Test Program.')
  parser.add_argument('--fft-size', default=1024, type=int,
                      help='Length of (complex) input vector for 1D FFT (C2C).')
  args = parser.parse_args()

  length = args.fft_size
  X = np.arange(0, 1, 1.0/length)
  Y = np.multiply(X, X + (1-X) * 1j)
  print(f"x: {X}")
  print(f"y: {Y}\n")

  # Compute 1D-FFT and compare result against scipy:
  N = len(Y)

  start = timer()
  re_inv_F, im_inv_F = inverse_fourier_matrix(N)
  re = np.matmul(re_inv_F, Y)
  im = np.matmul(im_inv_F, Y)
  result = re + (1j * im)
  end = timer()
  ft_time = end - start

  print(f"Real part of FT: {re}")
  print(f"Imaginary part of FT: {im}\n")
  print(f"FT: {result}")

  if np.allclose(fft(Y), result):
    print(f"FT Results match.\n")
  else:
    raise RuntimeError("FT Results do not match.\n")

  start = timer()
  result = my_fft(Y)
  end = timer()
  fft_time = end - start

  print(f"Cooley-Tukey FFT: {result}")
  if np.allclose(fft(Y), result):
    print(f"FFT Results match.\n")
  else:
    raise RuntimeError("FFT Results do not match.\n")

  print(f"FT time: {ft_time}, FFT Time: {fft_time}")
