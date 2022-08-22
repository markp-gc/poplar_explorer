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
  regx = re.compile("FFT of input-size ([-+]?[0-9]+) batch-size ([-+]?[0-9]+) radix-size ([-+]?[0-9]+) completed in ([-+]?[0-9]+) cycles.")
  flop_regx = re.compile("estimated FLOP count: ([-+]?[0-9]+)")
  input_size = None
  batch_size = None
  radix_size = None
  fft_cycles = None
  flops = None
  with open(args.log_file) as f:
    for line in f:
        match1 = regx.search(line)
        match2 = flop_regx.search(line)
        if match1:
          input_size = match1.group(1)
          batch_size = match1.group(2)
          radix_size = match1.group(3)
          fft_cycles = match1.group(4)
        if match2:
          flops = match2.group(1)

  if fft_cycles is None or flops is None:
    return None, None, None, None, None
  return int(input_size), int(batch_size), int(radix_size), int(fft_cycles), int(flops)


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
  args = parser.parse_args()
  report = pva.openReport(args.report_file)

  print(f"Poplar Version: {report.poplarVersion.string}")

  total_memory = sum(tile.memory.total.includingGaps for tile in report.compilation.tiles)
  print(f"Total memory use (bytes): {total_memory}")

  peak_name, peak_live_memory = getPeakLivenessStep(report)
  print(f"Program step consuming peak memory: {peak_name} {peak_live_memory}")

  size, bs, radix, cycles, flops = getFFTInfoFromLog(args.log_file)
  flops_per_cycle = flops/cycles if flops else None
  gflops_per_second = flops_per_cycle * args.clock_speed_ghz if flops_per_cycle else None
  print(f"Input size: {size}")
  print(f"Batch size: {bs}")
  print(f"Radix size: {radix}")
  print(f"FFT Cycles: {cycles}")
  print(f"Estimated FLOP count: {flops}")
  print(f"FLOPS per cycle: {flops_per_cycle}")
  print(f"GFLOPS/sec: {gflops_per_second}")

  # Collate everything into one line of CSV and append to file if specififed:
  if args.csv_out:
    with open(args.csv_out, "a") as f:
      f.write(f"{size},{bs},{cycles},{flops_per_cycle},{gflops_per_second},{total_memory},{peak_name},{peak_live_memory}\n")
