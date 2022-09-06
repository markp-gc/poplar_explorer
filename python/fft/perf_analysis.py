# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pva
import argparse
import re


# Find the name and bytes used for the program step with
# highest not-always-live memory usage.
def getPeakLivenessStep(report):
  peak_name = ""
  peak_live_memory = 0
  for step in report.compilation.livenessProgramSteps:
    if step.notAlwaysLiveMemory.bytes > peak_live_memory:
      peak_name = step.program.name
      peak_live_memory = step.notAlwaysLiveMemory.bytes
  return peak_name, peak_live_memory


# Parse the log of the `multi-tool FourierTransform` program:
def getFFTInfoFromLog(log_file):
  sizes_regx = re.compile(r"Building ([0-9])D-FFT of input-size ([-+]?[0-9]+) (?:\bbatch-size\b|x) ([-+]?[0-9]+) radix-size ([-+]?[0-9]+)")
  cycles_regx = re.compile(r"FFT completed in ([-+]?[0-9]+) cycles.")
  flop_regx = re.compile(r"estimated FLOP count: ([-+]?[0-9]+)")
  matmul_regx = re.compile(r"DFT Re-Matmul shape: [-+]?[0-9]+ [-+]?[0-9]+  x [-+]?[0-9]+ ([-+]?[0-9]+)")
  input_size = None
  batch_size = None
  radix_size = None
  fft_cycles = None
  flops = None
  dft_batch_size = None
  with open(args.log_file) as f:
    for line in f:
        match1 = sizes_regx.search(line)
        match2 = cycles_regx.search(line)
        match3 = flop_regx.search(line)
        match4 = matmul_regx.search(line)
        if match1:
          fft_type = match1.group(1)
          input_size = match1.group(2)
          batch_size = match1.group(3)
          radix_size = match1.group(4)
        if match2:
          fft_cycles = match2.group(1)
        if match3:
          flops = match3.group(1)
        if match4:
          dft_batch_size = match4.group(1)

  if input_size is None:
    raise RuntimeError("Could not parse log file.")

  if fft_cycles is None or flops is None:
    # Was probably out of memory or other error in this case so still return the sizes:
    return int(fft_type), int(input_size), int(batch_size), int(radix_size), None, None, int(dft_batch_size)
  return int(fft_type), int(input_size), int(batch_size), int(radix_size), int(fft_cycles), int(flops), int(dft_batch_size)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='1D-FFT Test Program.')
  parser.add_argument('--report-file', required=True, type=str,
                    help='PopVision report to analyse.')
  parser.add_argument('--log-file', required=True, type=str,
                    help='Output log of multi-tool FourierTransform (to read cycles and FLOP estimate).')
  parser.add_argument('--clock-speed-ghz', default=1.85, type=float,
                    help='Clock speed in GHz used to compute the FLOP rate. Default is 1.85GHz (BOW-2000)')   
  parser.add_argument('--csv-out', default=None, type=str,
                    help='Optional CSV filename to which stats will be appended.')
  parser.add_argument('--csv-write-headers', action="store_true",
                    help='Write the column headers to the file specified by `--csv-out`.')
  args = parser.parse_args()

  if args.csv_write_headers:
    if args.csv_out:
      print("Writing CSV headers")
      with open(args.csv_out, "w") as f:
        f.write(f"FFT-Type, Input-size,Batch-size,Radix-size,DFT Batch-size,Cycles,FLOPS Estimate,FLOPS/Cycle,GFLOPS/sec,Memory Including Gaps (bytes),Peak Live Memory Step, Peak Live Memory (bytes)\n")
      exit()
    else:
      raise RuntimeError("Can't write headers: no output file specified.")

  report = pva.openReport(args.report_file)
  print(f"Poplar Version: {report.poplarVersion.string}")

  total_memory = sum(tile.memory.total.includingGaps for tile in report.compilation.tiles)
  print(f"Total memory use (bytes): {total_memory}")

  peak_name, peak_live_memory = getPeakLivenessStep(report)
  print(f"Program step consuming peak memory: {peak_name} {peak_live_memory}")

  fft_type, size, bs, radix, cycles, flops, dft_batch_size = getFFTInfoFromLog(args.log_file)
  flops_per_cycle = flops/cycles if flops else None
  gflops_per_second = flops_per_cycle * args.clock_speed_ghz if flops_per_cycle else None
  print(f"FFT type: {fft_type}D")
  print(f"Input size: {size}")
  print(f"Batch size: {bs}")
  print(f"Radix size: {radix}")
  print(f"FFT cycles: {cycles}")
  print(f"Estimated FLOP count: {flops}")
  print(f"DFT batch size: {dft_batch_size}")
  print(f"FLOPS per cycle: {flops_per_cycle}")
  print(f"GFLOPS/sec: {gflops_per_second}")

  # Collate everything into one line of CSV and append to file if specififed:
  if args.csv_out:
    with open(args.csv_out, "a") as f:
      f.write(f"{fft_type}D,{size},{bs},{radix},{dft_batch_size},{cycles},{flops},{flops_per_cycle},{gflops_per_second},{total_memory},{peak_name},{peak_live_memory}\n")
