#!/bin/bash

# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

TYPE="1d"
RUN_DIR="profiles_${TYPE}"
mkdir -p $RUN_DIR
CSV_FILE=${RUN_DIR}/"csv_results.txt"

for SIZE in 64 128 256 512 1024 2048 4096 8192
do
  for RADIX in 16 32 64 128 256 512 1024 2048
  do
    for BS in 1 2 4 8 16 32 64 128 512 1024
    do
      RUN_NAME="fft_1d_${SIZE}_bs${BS}_radix${RADIX}"
      export POPLAR_ENGINE_OPTIONS="{\"autoReport.all\":\"true\", \"autoReport.directory\":\"${RUN_DIR}/${RUN_NAME}\", \"profiler.includeFlopEstimates\":\"true\"}"
      echo "Running size: ${SIZE} batch-size: ${BS} radix: ${RADIX}"
      mkdir -p ${RUN_DIR}/${RUN_NAME}
      ./multi-tool FourierTransform --fft-type ${TYPE} --fft-size ${SIZE} --batch-size ${BS} --radix-size ${RADIX} > ${RUN_DIR}/${RUN_NAME}/run_log.txt &
    done
    wait
  done
done

# Overwrite the CSV file writing new headers:
python ../python/fft/perf_analysis.py --report-file fake --log-file fake --csv-out ${CSV_FILE} --csv-write-headers

# Append all run results to CSV file:
for DIR in $RUN_DIR/*
do
  LOG_FILE=$DIR/run_log.txt
  REPORT_FILE=$DIR/ipu_utils_engine/profile.pop
  python ../python/fft/perf_analysis.py --report-file ${REPORT_FILE} --log-file ${LOG_FILE} --csv-out ${CSV_FILE}
done
