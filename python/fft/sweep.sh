#!/bin/bash

# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

RUN_DIR="profiles"
mkdir -p $RUN_DIR
CSV_FILE=${RUN_DIR}/"csv_results.txt"

for SIZE in 128 256 512 1024 2048 4096 8192 16384 32768
do
  for BS in 1 2 4 8 16 32 64 128 256
  do
    for RADIX in 32 64 128 256 512 1024 2048
    do
      RUN_NAME="fft_1d_${SIZE}_bs${BS}_radix${RADIX}"
      export POPLAR_ENGINE_OPTIONS="{\"autoReport.all\":\"true\", \"autoReport.directory\":\"${RUN_DIR}/${RUN_NAME}\", \"profiler.includeFlopEstimates\":\"true\"}"
      echo "Running size: ${SIZE} batch-size: ${BS} radix: ${RADIX}"
      mkdir -p ${RUN_DIR}/${RUN_NAME}
      ./multi-tool FourierTransform --fft-size ${SIZE} --batch-size ${BS} --radix-size ${RADIX} > ${RUN_DIR}/${RUN_NAME}/run_log.txt
    done
  done
done

for DIR in $RUN_DIR/*
do
  LOG_FILE=$DIR/run_log.txt
  REPORT_FILE=$DIR/ipu_utils_engine/profile.pop
  python ../python/fft/perf_analysis.py --report-file ${REPORT_FILE} --log-file ${LOG_FILE} --csv-out ${CSV_FILE}
done
