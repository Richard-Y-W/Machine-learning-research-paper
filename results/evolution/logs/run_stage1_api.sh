#!/bin/sh
# Sequential (one process: RAM-limited) Stage 1 API arms, protocol priority order.
cd "<LOCAL_REPOSITORY>"
export EVO_THREADS=16 PYTHONUNBUFFERED=1
while tasklist | grep -qi "python.exe"; do sleep 20; done   # wait for grad reference to finish
P="python -m experiments evo lineage --generations 10"
$P --arm E-LLM --lineages 3 4 5 6 7 8 --test-at 1 2 3 5 7 10 || exit 1
$P --arm R-LLM --lineages 8 --no-inherit --test-at || exit 1
$P --arm E-LLM-nohist --lineages 1 2 3 4 --no-history --test-at 5 10 || exit 1
$P --arm E-LLM-free --lineages 1 2 3 4 --unconstrained --test-at 5 10 || exit 1
echo STAGE1_API_DONE
