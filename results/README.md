# Results index

This repository preserves both the routing-edit study and the earlier experiments that led to it. Original paths are retained because scripts and archived records refer to them. The [current paper](../paper/main.pdf) is the authoritative account of the reported claims and limitations.

## Evidence for the routing-edit paper

| Result location | Contents |
| :--- | :--- |
| [`compose/lookup3.json`](compose/lookup3.json) | 0.6B routing fit, gain sweep and initial controls |
| [`compose/lookup3_confirm.json`](compose/lookup3_confirm.json) | Fresh-prompt confirmation, paired wins/losses and output interventions |
| [`compose/lookup3_controls.json`](compose/lookup3_controls.json) | Host specificity, shared keys, counterfactual tables and structural nulls |
| [`compose_0p6b_check/find_head.json`](compose_0p6b_check/find_head.json) | 0.6B head-selection rule |
| [`compose_1p7b/`](compose_1p7b/) | Larger-model fit, confirmation, head selection and related measurements |
| [`compose/increment.json`](compose/increment.json) | Code-like variable lookup and digit increment |
| [`compose/learning_cmp.json`](compose/learning_cmp.json) | Gradient-trained routing and fine-tuning comparisons |
| [`compose/hand_ref.json`](compose/hand_ref.json) | Hand-designed edit measured through the designer harness |
| [`compose/opus_program/`](compose/opus_program/) | First 0.6B designer run and programme decomposition |
| [`compose_run2/opus_program/`](compose_run2/opus_program/) | Second 0.6B designer run |
| [`compose_1p7b/opus_program/`](compose_1p7b/opus_program/) | 1.7B designer run |
| [`evolution/splits.json`](evolution/splits.json) | Frozen data arrays, including the WikiText probe reused by routing experiments |

The `.pt` files in the composition directories contain fitted routing matrices and related edit artifacts. They are not full pretrained model checkpoints. The table and figure scripts in [`paper/`](../paper/README.md) show precisely which JSON fields feed each display.

## Exploratory and earlier work

| Directories | Study |
| :--- | :--- |
| `compose/bridge*`, `compose/feedback*`, `compose/reembed*`, `compose/lookup*` | Earlier composition constructions and controls; `lookup3*` is the successful routing construction |
| `compose_op2_*`, `compose_opgate_*` | Competence screening and oracle interventions for other operations |
| [`amortize/`](amortize/) | Attempts to improve reasoning accuracy within token budgets |
| [`evolution/`](evolution/) | Iterative weight-editing lineages, random controls, evolution strategies and budget records |
| [`skill/`](skill/) | Attempted installation of a letter-shift skill |
| `main/`, `dense-*`, `deployment-*`, `cycle3-*`, `cycle4-*`, `reproduction/` | Earlier synthetic-dynamics experiments and reproduction checks |
| `wsp-*` | Earlier weight-space-programming pilots and evaluations |

Directories and files named `*dry*`, `*smoke*`, `local_aborted/` or `*broken*` (for example `amortize/edit_dry/`, `amortize/edit_smoke/`, `compose/gate_v0_brokenlayout.json` and `evolution/smoke_dossier.txt`) are debugging runs or aborted attempts kept for provenance; they carry no results.

Negative or exploratory outcomes remain part of the record. A filename or an early report should not be read as a claim that its approach succeeded. See the paper's appendix and the [historical documentation](../docs/README.md).
