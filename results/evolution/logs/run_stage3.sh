#!/bin/sh
# v1 Stage 2 remainder + Evolution v2 (docs/evolution-v2-protocol.md). Sequential: RAM fits one model process.
cd "<LOCAL_REPOSITORY>"
export EVO_THREADS=16 PYTHONUNBUFFERED=1
while tasklist //FI "PID eq 33584" | grep -qi python; do sleep 20; done   # running v1 DFE
E="python -m experiments evo"; V="python -m experiments.evolution.v2"
TASKS="mmlu arc_challenge hellaswag commonsense_qa"
S1=""; S2=""; for t in $TASKS; do S1="$S1 1_$t"; S2="$S2 2_$t"; done
$E crossover || exit 1
$V smoke > results/evolution/logs/v2_smoke.log 2>&1; grep -q SMOKE_OK results/evolution/logs/v2_smoke.log || { echo SMOKE_FAILED; exit 1; }
echo V2_SMOKE_OK
# protocol cut order: G-LLM gens 11-15 are cut first, then specialist set 2 -> run them last
$V lineage --arm G-LLM --units 1 2 --generations 10 --test-at 5 10 || exit 1
$V lineage --arm S-LLM --units $S1 --generations 10 --test-at 10 || exit 1
$V lineage --arm S-RAND --units $S1 --generations 10 --test-at 10 || exit 1
$V breed --set 1 --source LLM || exit 1
$V breed --set 1 --source RAND || exit 1
$V lineage --arm S-LLM --units $S2 --generations 10 --test-at 10 || exit 1
$V lineage --arm S-RAND --units $S2 --generations 10 --test-at 10 || exit 1
$V breed --set 2 --source LLM || exit 1
$V breed --set 2 --source RAND || exit 1
$V lineage --arm G-RAND --units 1 2 --generations 15 --test-at 5 10 15 || exit 1
$V lineage --arm G-LLM --units 1 2 --generations 15 --test-at 5 10 15 || exit 1
echo V2_DONE
$E es --seeds 3 4 5 || exit 1          # v1 addition beyond frozen protocol
python -m experiments.evolution.analyze || exit 1
echo STAGE3_DONE
