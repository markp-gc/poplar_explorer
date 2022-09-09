#!/bin/bash

# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

TYPE="2d"
RUN_DIR="profiles_${TYPE}"
mkdir -p $RUN_DIR
CSV_FILE=${RUN_DIR}/"csv_results.txt"
RUN_PREFIX="fft_${TYPE}"

# Overwrite the CSV file writing new headers:
python ../python/fft/perf_analysis.py --report-file fake --log-file fake --csv-out ${CSV_FILE} --csv-write-headers

for SIZE in 32 64 128 256 512 1024 2048 4096 8192
do
  for SF in 1 2 4 8 16 32 64
  do
    for RADIX in 2 4 8 16 32 64 128 256
    do
      RUN_NAME="${RUN_PREFIX}_${SIZE}_sf${SF}_radix${RADIX}"
      export POPLAR_ENGINE_OPTIONS="{\"autoReport.all\":\"true\", \"autoReport.outputExecutionProfile\": \"false\", \"autoReport.directory\":\"${RUN_DIR}/${RUN_NAME}\", \"profiler.includeFlopEstimates\":\"true\"}"
      echo "Running size: ${SIZE} serial-steps: ${SF} radix: ${RADIX}"
      mkdir -p ${RUN_DIR}/${RUN_NAME}
      ./multi-tool FourierTransform --fft-type ${TYPE} --fft-size ${SIZE} --batch-size ${SIZE} --serialisation-factor ${SF} --radix-size ${RADIX} > ${RUN_DIR}/${RUN_NAME}/run_log.txt &
    done
    wait

    # Append run results to CSV file:
    for RADIX in 2 4 8 16 32 64 128 256
    do
      RUN_NAME="${RUN_PREFIX}_${SIZE}_sf${SF}_radix${RADIX}"
      DIR=${RUN_DIR}/${RUN_NAME}
      LOG_FILE=$DIR/run_log.txt
      REPORT_FILE=$DIR/ipu_utils_engine/profile.pop
      python ../python/fft/perf_analysis.py --report-file ${REPORT_FILE} --log-file ${LOG_FILE} --csv-out ${CSV_FILE}
      # Delete profiles when done as they are large:
      rm -rf ${RUN_DIR}/${RUN_NAME}/ipu_utils_engine
    done

  done
done
