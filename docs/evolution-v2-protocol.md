# Evolution v2 protocol: rotating selection and gene-level breeding

Status: FROZEN on 2026-11-14, before any v2 API call. Amendment A1 (below) was also applied before any v2 API call. Stage 1 of v1 is reused unchanged: the organism, the edit language, the editor, the
test sets and the $20 cap.

## Why v2

v1 showed that single-lineage evolution plateaus by generation 5. All methods overfit the fixed
160-item dev set, and Opus's per-edit advantage (66% vs 51% improving offspring) does not reach
held-out accuracy. A v1-style lineage is mostly repeated single-session WSP.

v2 asks two questions that only an evolutionary process can answer:

- **Q1, compounding.** When selection data rotates every generation, so memorizing a sample cannot
  pay, do LLM lineages keep gaining on held-out data? Do they beat v1 lineages run with the same
  number of generations?
- **Q2, genetics.** Can traits evolved in separate specialist lineages be recombined, gene by gene,
  into a generalist? Is LLM-authored breeding better than human-designed recombination and merging
  operators?

## Data

- **Held-out test.** The v1 test sets, unchanged: target 4 × 250 items, protected 5 × 100 items,
  WikiText 32 × 128. v2 numbers are therefore directly comparable to v1. The editor never sees test
  data.
- **Selection pool.** For each target task, every item in the v1 sources that is not in any v1
  split (dev, test or calib), using the same normalization, deduplication and ≤ 200-token filter.
  The pool is written to `results/evolution/v2/pool.json` with hashes.
- **Rotating sample.** Every generation draws a fresh sample from the pool, deterministically from
  the lineage name and generation, and re-scores the parent on it before scoring offspring. An
  offspring is accepted if it beats the parent on that same sample and is admissible.
  - Generalist sample: 40 items per target task.
  - Specialist sample: 160 items of one task.
- **Constraints.** The fixed v1 protected dev set and WikiText dev, with the same Δ = 0.02 vs the
  original model.

## Arms

Each arm uses 6 offspring per generation, and the editor uses the v1 system prompt and edit
language.

| Arm | Units | Generations | Fitness | Proposer | API calls |
|---|---|---|---|---|---|
| G-LLM (rotating generalist) | 2 | 15 | 4-task rotating sample | Opus, history on | 30 |
| G-RAND | 2 | 15 | same | magnitude-matched random (v1 calibration) | 0 |
| S-LLM (specialists) | 8 = 2 sets × 4 tasks | 10 | 1-task rotating sample | Opus | 80 |
| S-RAND | 8 | 10 | same | random | 0 |

The fair v1 comparator for G-LLM is v1 E-LLM at generation 10, plus G-LLM's own generation-10 test.

## Breeding (Q2)

Each specialist set holds one specialist per target task. Its genes are the accepted mutations of
its 4 specialists, and each gene carries its ops, rationale, birth generation and birth Δfitness.

1. **Gene assay.** For every gene, apply that gene alone to the original model and evaluate it on a
   fresh 4 × 40 breeding sample. This gives per-task Δ log-probability and admissibility. The assay
   is shared by all operators and is not charged to any operator's budget.
2. **Operators.** Each operator gets 24 evaluations on the breeding sample. The best admissible
   recombinant is chosen on that sample and tested once.
   - **LLM-breed.** 4 rounds × 6 recombinants. Opus sees the gene table (ops summary, rationale,
     source task, assay effects) and the results of earlier rounds. It returns gene subsets.
     2 replicates per set and gene source; 4 calls per replicate. Reduced from 3 before freezing,
     to fit the cap.
   - **RAND-breed.** 24 random gene subsets, with subset sizes drawn from LLM-breed's subset sizes
     for that set. 3 seeds per set.
   - **GREEDY-breed.** A human-designed heuristic. Rank genes by assay Δ on the 4-task sample. Try
     prefixes of that ranking, plus greedy forward additions, until 24 evaluations are used.
     Deterministic.
   - **UNION.** All genes of the set (1 evaluation).
   - **TIES.** TIES-merge of the 4 specialist task vectors (keep 0.2) with scaling λ ∈ {0.5, 0.75,
     1.0, 1.25, 1.5}, chosen on the breeding sample.
   - **CMA-merge.** Sakana-style evolutionary merging: sep-CMA-ES over 4 per-specialist task-vector
     coefficients in [0, 1.5], 24 evaluations, 3 seeds per set.
3. **Gene source.** Breeding runs both on S-LLM sets (Opus-authored genes) and on S-RAND sets
   (random genes). Every operator is applied to both, giving a gene source × operator design.

## Outcomes and analysis

**Primary outcome:** test target accuracy, with protected accuracy and NLL reported and test-constraint
pass flagged as in v1.

- **Q1.**
  - G-LLM test trajectory at generations 5, 10 and 15.
  - G-LLM at generation 10 vs v1 E-LLM at generation 10.
  - G-LLM vs G-RAND.
- **Q2.**
  - LLM-breed vs each other operator, on S-LLM genes.
  - The interaction between gene source and operator.
  - The bred generalist vs its best specialist on the 4-task test, and vs G-LLM.
  - Gene transplantability: the fraction of genes whose assay Δ on non-source tasks is ≥ 0.
- **Statistics.** Hierarchical bootstrap over units and paired test items, as in v1. The sample
  sizes are small (2 G lineages; 2 sets × 3 replicates). v2 is reported as a pre-registered pilot
  with effect sizes and CIs, not as confirmatory.

## Budget

- Estimated cost:
  - Evolution: 30 G-LLM + 80 S-LLM calls at about $0.055.
  - Breeding: 2 gene sources × 2 sets × 2 replicates × 4 calls (32 calls) at about $0.06.
  - Total about $8.0. Projected committed total about $18.8 of the $19.50 stop.
- The v1 ledger and cap stay as they are: stop at $19.50 committed.
- If money runs short, cut in this order: G-LLM generations 11–15 first, then specialist set 2.
- The driver runs those last (`results/evolution/logs/run_stage3.sh`):
  - G-LLM 1–10 → S-LLM/S-RAND set 1 → breed set 1 → set 2 → breed set 2 → G-RAND → G-LLM 11–15.
- Extra editor-prompt text in v2:
  - A "v2 conditions" note saying that dev items are re-sampled each generation.
  - For specialists, a note naming the single target task.
- The v1 30-generation extension was dropped by user decision on 2026-11-14 to fund v2. This is a
  documented deviation.

## Amendment A1 (2026-11-14, before any v2 API call or v2 run)

A quantitative-genetics analysis of v1 found that selection there is noise-limited:
- Realized heritability, test gain over dev gain, is 0.27 for E-LLM and 0.51 for E-RAND.
- The mean accepted dev step (0.0123) is about the expected best of 6 draws of pure noise at
  160 items (0.0131).
- At 160 items per evaluation, v2 would be underpowered.

Changes:
- Generalist sample: 160 items per task, 640 total.
- Specialist sample: 640 items of the task. The CSQA (907) and ARC-C (1,152) pools mean successive
  samples overlap more, so rotation is weaker for those specialists.
- Breeding sample: 160 items per task.
- Dev cache: levels [12, 16, 19, 21, 23, 25, 26, 27], down from 12..27, to bound RAM. This is exact
  either way, only slower for edits at uncached layers.

API cost is unchanged, because each generation is still one call. CPU time rises by about 2–3×.
Arms, generations, operators, outcomes and analysis are unchanged.
