# Frozen protocol: model evolution by weight-space programming (Stage 1–2)

Frozen 2026-11-13, before any Stage 1 run. Results are reported against this file; any later
change is listed in the living plan's status log as a deviation.

## Organism, data, fitness

- **Target model:** `Qwen/Qwen3-0.6B-Base`, FP32, CPU, forward passes only. Chosen by the Stage 0
  rule as the largest candidate that runs on this machine within a practical time per evaluation.
- **Format:** zero-shot lettered multiple choice. Each item is scored by the next-token distribution
  over the valid letters after `Answer:`.
- **Benchmarks:**
  - Target: MMLU, ARC-Challenge, HellaSwag, CommonsenseQA.
  - Protected: ARC-Easy, OpenBookQA, PIQA, Winogrande, SciQ, plus WikiText-103 NLL.
- **Splits:** frozen in `results/evolution/splits.json`, hashed by underlying question. Sizes:

  | Split | Size |
  |---|---|
  | dev target | 4×40 |
  | dev protected | 5×12 |
  | test target | 4×250 |
  | test protected | 5×100 |
  | calibration target | 4×24 |
  | WikiText dev | 4×128 tokens |
  | WikiText test | 32×128 tokens |

  Split hashes are recorded in every run log. Test data never reaches the editor and is never used for
  selection.
- **Fitness (selection):** mean over target tasks of the mean dev log-probability of the correct letter.
- **Admissibility:** protected-dev fitness ≥ original − 0.02, and dev WikiText NLL ≤ original + 0.02.
  Both bounds are relative to the ORIGINAL checkpoint (stabilizing selection).
- **Primary outcome:** held-out TEST target accuracy (macro mean over the 4 tasks) minus the original
  model's.
- **Secondary outcomes:** test target log-probability, protected test accuracy, test NLL, per-task
  accuracy.

## Mutation operator

- **Editor:** `claude-opus-5-5`, adaptive thinking, effort `medium`, `max_tokens` 12,000, JSON schema
  output. The system prompt is fixed in `experiments/evolution/editor.py`.
- **Each call:**
  - Input: a forward-only dossier of the parent (dev fitness, per-matrix σ1, direct logit attribution
    for heads and neurons, error directions with logit lens, confidently wrong dev items), plus the
    lineage history.
  - Output: 6 mutations in the edit language (`scale`, `ablate`, `rank1` with named reference vectors,
    `set`, `revert`), each with ≤8 operations and ≤50% relative Frobenius change per touched tensor.
- Invalid mutations count as failed offspring. Evaluated calls are never retried.

## Selection

(1+λ) elitist selection with λ = 6. The best admissible offspring replaces the parent if it raises
dev fitness. Accepted mutations are inherited as an ordered, exactly replayable genome. Offspring
are evaluated by exact partial re-execution from the parent's cached layer inputs; this was verified
bitwise equal to full evaluation (`results/evolution/verify_fast.json`).

## Arms (G = 10 generations unless noted)

| Arm | Description | Lineages | Calls |
|---|---|---|---|
| E-LLM | LLM mutation, heredity, history shown, admissibility enforced | 8 | 80 |
| R-LLM | Same calls on the ORIGINAL each generation (no heredity); result = best admissible offspring seen | 8 | 80 |
| E-LLM-nohist | Heredity of weights, but the editor sees no lineage history | 4 | 40 |
| E-LLM-free | Heredity and history, but selection ignores admissibility (ratchet test) | 4 | 40 |
| E-RAND | Same loop, random mutations magnitude-matched to the LLM's proposal distribution (see `baselines.py`), calibrated on all Stage 1 LLM proposals | 16 | 0 |
| ES-56 | sep-CMA-ES over 56 per-layer output gates (attn.o and mlp.down scales); penalized fitness | 2 seeds × 300 evaluations | 0 |

Equal-budget comparisons use 60 offspring evaluations per lineage. That is the E-LLM lineage budget
of 10 generations × 6.

**Test evaluation points:**
- E-LLM: generations 1, 2, 3, 5, 7, 10.
- Other lineage arms: generations 5 and 10.
- R-LLM: final best only.
- The original model is evaluated on test once.

## Stage 2 (after Stage 1)

- **DFE:** 30 random mutations at generations 0, 5 and 10 of every E-LLM and E-LLM-free lineage.
  Measured: fraction beneficial, fraction admissible-beneficial, and the mean deleterious effect.
- **Genetics on E-LLM finals:**
  - transplant and additivity for all pairs (A∪B in both orders);
  - TIES merging of the genome deltas;
  - LLM-mediated crossover (one call per pair, for 4 disjoint pairs; the result is the best admissible
    of its 6 offspring);
  - genome overlap: cosine similarity of deltas and shared touched components.
- **Long-run extension:** the 2 E-LLM lineages with the highest generation-10 dev fitness continue to
  generation 30. Ties are broken by lineage id. This lineage selection uses dev data only.
- **Reference (not a competitor):** gradient fine-tuning of layers ≥ 20 on the same dev data, to mark
  the available headroom.

## Analysis

**Primary contrasts on the primary outcome at generation 10:**
- C1: E-LLM − R-LLM.
- C2: E-LLM − E-RAND.
- C3: E-LLM − ES-56 (ES at 60 evaluations; evaluations-to-match from its curve).

**Secondary contrasts:**
- C4: E-LLM − E-LLM-nohist.
- C5: E-LLM-free vs E-LLM on protected accuracy and NLL across generations.
- H4: DFE trend.
- H5: interaction term Δ(A∪B) − Δ(A) − Δ(B).
- H6: overlap.

Uncertainty uses a hierarchical bootstrap: resample lineages within each arm, then test items;
10,000 replicates; 95% percentile intervals. With 4–16 lineages per arm this is an exploratory
pilot. Every lineage is reported. Failed calls and invalid offspring stay in all denominators.

## Budget

Hard cap $20.00 of editor API usage. The ledger reserves the worst case before each call and stops
at $19.50 committed. Priority order if the money runs short:
1. E-LLM
2. R-LLM
3. E-LLM-nohist
4. E-LLM-free
5. crossover
6. long-run extension

Stage 0 spent $0.123 (3 calls) and is excluded from all study results.
