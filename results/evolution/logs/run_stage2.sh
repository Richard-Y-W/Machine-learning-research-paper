#!/bin/sh
# Stage 2 (sequential; RAM-limited). Starts after Stage 1 API arms finish.
cd "<LOCAL_REPOSITORY>"
export EVO_THREADS=16 PYTHONUNBUFFERED=1
until grep -q STAGE1_API_DONE results/evolution/logs/stage1_api.log; do sleep 30; done
while tasklist | grep -qi "python.exe"; do sleep 20; done
E="python -m experiments evo"
$E calibrate || exit 1
$E lineage --arm E-RAND --proposer random --lineages $(seq 1 16) --generations 10 --test-at 5 10 || exit 1
$E dfe || exit 1
$E crossover || exit 1
# extension: 2 best E-LLM lineages by gen-10 DEV fitness (E-LLM_3 -1.0956, E-LLM_8 -1.1041)
$E lineage --arm E-LLM --lineages 3 8 --generations 30 --test-at 15 20 25 30 || exit 1
$E es --seeds 3 4 5 || exit 1          # addition beyond frozen protocol
python -m experiments.evolution.analyze || exit 1
echo STAGE2_DONE
