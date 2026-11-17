#!/bin/sh
# Stage 2 remainder after dropping the 30-gen extension (user decision 2026-11-14; deviation logged).
cd "<LOCAL_REPOSITORY>"
export EVO_THREADS=16 PYTHONUNBUFFERED=1
while tasklist //FI "PID eq 33584" | grep -qi python; do sleep 20; done   # wait for running DFE (PID 33584)
E="python -m experiments evo"
$E crossover || exit 1
$E es --seeds 3 4 5 || exit 1          # addition beyond frozen protocol
python -m experiments.evolution.analyze || exit 1
echo STAGE2_DONE
