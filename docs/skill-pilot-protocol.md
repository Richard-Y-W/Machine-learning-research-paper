# Pilot: programming a new skill into a pretrained model by iterated WSP

Status: FROZEN on 2026-11-14, before any pilot API call or result.
Budget: user-authorized $33.24 of further API spend (total cap $45.08, stop at $44.50).

## Question

Can an LLM editor write a **new, compact skill** into a pretrained language model's weights,
directly and without gradients or training data, so that the skill **generalizes to inputs the
editor never saw**? A skill that only works on seen inputs is fact insertion, which existing
editors such as ROME already do. Generalization is what would make it a skill.

The prior claim being tested is MIT's "Sparks of WSP" statement that editing "could not add an
ability the model lacked". The closest prior method is Tracr-Injection (arXiv 2505.10719), which
inserts algorithms into pretrained models by gradient distillation.

## Organism and skill

- **Model:** Qwen3-0.6B-Base, FP32, CPU. The same organism as in the evolution study.
- **Skill, ENC:** after the trigger `ENC`, output the next letter of the alphabet.
  - `ENC: c ->` → ` d`.
  - Letters a–y.
  - The base model does not know this convention; that is measured.
- **Inputs.** Held-out input letters are fixed by seed `skill-pilot-v1`.
  - Held-out letters: **a d h o q t u v**.
  - Practice letters: the other 17.
- **Templates:**
  - Practice: `ENC: {x} ->` and `ENC({x}) =`.
  - Held-out: `Input: ENC {x}\nOutput:`.
- **Control prompt:** `The letter after {x} is`. It measures whether the latent successor knowledge
  already exists. It is shown on practice letters only.

The editor sees only practice letters with practice templates, plus the control prompt on practice
letters. Activation references can be built only from templates instantiated over practice
letters. Literal held-out letters are rejected.

## Editor loop (WSP arm)

- Iterated WSP chain, reusing the evolution study's edit language: scale, ablate, rank1, set,
  revert.
- New activation references:
  - `act:<site>:L<i>:<template>`, where site is resid, attn or mlp: the mean activation at the
    last position over practice letters.
  - Differences written `A - B`.
- Per generation: 1 Opus 5.5 call (effort medium) proposing 6 candidate edits.
  - Each candidate is scored on the practice set.
  - Fitness: mean log-probability of the correct token over the full vocabulary.
  - The best candidate is accepted if it improves fitness and is admissible.
  - Admissible means: protected-dev fitness ≥ original − 0.02 and WikiText-dev NLL ≤ original +
    0.02, the same constraints as v1.
- **2 chains × 15 generations**, at most 30 calls, about $3.

## Baselines (the same practice data unless noted)

| Baseline | What it is |
|---|---|
| Base | Zero-shot |
| Few-shot prompting | 3 practice examples in the prompt at inference: the "package" alternative |
| Activation steering (ActAdd) | Mean residual difference between few-shot and zero-shot practice prompts, added at layer L with coefficient c at inference. L ∈ {4, 8, 12, 16, 20} and c ∈ {1, 2, 4, 8} are chosen on practice accuracy. Note that it is applied only to trigger prompts. |
| Fine-tuning | Adam on layers ≥ 14, cross-entropy on the answer token. Train on 13 practice letters × 2 templates; validate on 4 practice letters to choose lr ∈ {1e-5, 3e-5, 1e-4} and the step count (≤ 40). |

## Outcomes, measured once at the end

1. **Held-out letters**: exact-match accuracy on practice templates. **Primary.**
2. **Held-out template**: accuracy on practice letters.
3. **Both held out**: held-out letters on the held-out template.
4. Practice accuracy.
5. Regression: test protected accuracy and fitness (500 items) and WikiText-test NLL, compared
   with the original.

Exact match means the argmax over the full vocabulary is the correct token.

## Pre-registered verdicts for the WSP arm, on its best chain

| Verdict | Criterion |
|---|---|
| **WORKS** | Held-out-letter accuracy ≥ 80%, no test regression beyond 0.02 |
| **PARTIAL (fact insertion)** | Practice ≥ 80%, held-out letters < 50% |
| **FAILS** | Held-out letters < 30%, and not PARTIAL |

Anything else is reported as "intermediate". The comparison with steering and fine-tuning is
reported as is. If WSP WORKS, the pilot justifies a full study. If it is PARTIAL or FAILS, the
negative result is reported.

## Deviation D1 (2026-11-14, about 02:40)

- **Harness bug.** The system prompt advertised v1 reference vectors (err:/mean:/attnerr:/mlperr:)
  that do not exist in this task. As a result, 5 of 6 round-1 edits were invalid.
- **Fix.** The prompt now lists only the available references.
- **Restart.** Chain A was restarted from scratch. The broken round 1 ($0.06) is kept in
  `results/skill/discarded_A_bad_refs/` and excluded from all results.
