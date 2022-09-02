#!/bin/bash

# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

RUN_DIR="2d_profiles"
mkdir -p $RUN_DIR
CSV_FILE=${RUN_DIR}/"2d_csv_results_2.txt"

for SIZE in 256 512 1024 2048
do
  for BS in 1 4 8 16 32 64
  do
    for RADIX in 2 4 8 16 32 64 128 
    do
      RUN_NAME="fft_2d_${SIZE}_bs${BS}_radix${RADIX}"
      export POPLAR_ENGINE_OPTIONS="{\"autoReport.all\":\"true\", \"autoReport.directory\":\"${RUN_DIR}/${RUN_NAME}\", \"profiler.includeFlopEstimates\":\"true\"}"
      echo "Running size: ${SIZE} batch-size: ${BS} radix: ${RADIX}"
      mkdir -p ${RUN_DIR}/${RUN_NAME}
      ./multi-tool FourierTransform2D --fft-size ${SIZE} --batch-size ${BS} --radix-size ${RADIX} > ${RUN_DIR}/${RUN_NAME}/run_log.txt &
    done
    wait
  done
done

# Overwrite the CSV file writing new headers:
python3 ../python/fft/perf_analysis.py --report-file fake --log-file fake --csv-out ${CSV_FILE} --csv-write-headers

# Append all run results to CSV file:
for DIR in $RUN_DIR/*
do
  echo Processing path $DIR
  LOG_FILE=$DIR/run_log.txt
  REPORT_FILE=$DIR/ipu_utils_engine/profile.pop
  python3 ../python/fft/perf_analysis.py --report-file ${REPORT_FILE} --log-file ${LOG_FILE} --csv-out ${CSV_FILE}
done
