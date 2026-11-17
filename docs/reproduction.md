# Reproduction guide

Run commands from the repository root. The manuscript is [`paper/main.tex`](../paper/main.tex), and its compiled PDF is [`paper/main.pdf`](../paper/main.pdf).

## 1. Regenerate displays from saved measurements

This path needs Python and Matplotlib. It makes no model or LLM API calls.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install matplotlib
python paper/make_tables.py
python paper/make_figures.py
```

Tables read the archived JSON values directly. Figures use the same saved measurements. See the [result-to-experiment map](../results/README.md) for the underlying files. Rendering with a different Matplotlib version may change typography without changing the measurements.

To regenerate the README's result image from the figure PDF, install `pymupdf` and run:

```bash
python -m pip install pymupdf
python paper/make_readme_graphics.py
```

## 2. Compile and package the paper

With [Tectonic](https://tectonic-typesetting.github.io/) installed:

```bash
tectonic -X compile paper/main.tex
python paper/package_preprint.py
```

Alternatively, run `pdflatex main.tex` twice from `paper/`. References are embedded in the source; BibTeX is not required. The package script copies the PDF into `output/pdf/`, creates a source archive under `output/preprint/`, and writes submission text and SHA-256 hashes. Compile before packaging so the PDF reflects the latest source.

The PDF, source archive and submission text prepared for this version are in [`output/`](../output/). The source archive includes only `main.tex`, the six figure PDFs and nine table files.

## 3. Run the routing experiments

This path downloads a Qwen checkpoint when it is not cached and performs model inference. Install an appropriate PyTorch build for your CPU or GPU, then the experiment dependencies:

```bash
python -m pip install -r requirements-compose.txt
```

The [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B-Base#requirements) specifies Transformers 4.51.0 or newer for Qwen3 support. This repository uses the `dtype` loading argument introduced in [Transformers 4.56](https://github.com/huggingface/transformers/releases/tag/v4.56.0), so `requirements-compose.txt` sets 4.56 as its lower bound. This is a dependency specification, not a lockfile for the original runs. [`requirements-reproduced.txt`](../requirements-reproduced.txt) records an earlier synthetic-dynamics environment and should not be treated as the routing study's environment.

The backend uses float32 and selects CUDA when available; otherwise it uses CPU. Set `EVO_MODEL`, `WSP_OUT` and, optionally, `EVO_THREADS`. Keep new measurements in a separate output directory to preserve the archived evidence.

Linux/macOS:

```bash
export EVO_MODEL=Qwen/Qwen3-0.6B-Base
export WSP_OUT=results/local-routing-0p6b
export EVO_THREADS=6
python -m experiments.compose.find_head
python -m experiments.compose.lookup3
python -m experiments.compose.lookup3_confirm
```

Windows PowerShell:

```powershell
$env:EVO_MODEL = "Qwen/Qwen3-0.6B-Base"
$env:WSP_OUT = "results/local-routing-0p6b"
$env:EVO_THREADS = "6"
python -m experiments.compose.find_head
python -m experiments.compose.lookup3
python -m experiments.compose.lookup3_confirm
```

The locating script writes its selected head to `find_head.json`; for both reported models this is layer 21, head 6. The routing scripts default to `WSP_HEAD=21,6`. If exploring a model with a different selected head, set that variable explicitly; the locating script does not automatically configure later commands.

`lookup3` fits the routing matrices and writes `lookup3_attention_programme.pt`. Confirmation loads that tensor file, so the fit must run first. Repeat the sequence with `EVO_MODEL=Qwen/Qwen3-1.7B-Base` and a separate `WSP_OUT` to reproduce the larger-model procedure.

The short WikiText probe is read from the checked-in `wiki_dev` arrays in [`results/evolution/splits.json`](../results/evolution/splits.json). It is four 128-token chunks, scored on 508 within-chunk next-token predictions. Reusing those arrays reproduces the paper's measurement; it does not create an independent preservation set.

## Additional experiments

These are separate model runs. They are not needed to regenerate the paper from saved results.

| Module | Purpose | Prerequisite |
| :--- | :--- | :--- |
| `experiments.compose.lookup3_controls` | Host, shared-key, distractor and structural controls | Saved routing fit in `WSP_OUT` |
| `experiments.compose.increment` | Word-fitted routing on digit assignments | Model and archived WikiText arrays |
| `experiments.compose.learning_cmp` | Gradient-trained routing and fine-tuning comparisons | Saved routing fit in `WSP_OUT` from `experiments.compose.lookup3`; model and archived WikiText arrays; includes gradient training |
| `experiments.compose.hand_ref` | Hand-designed edit evaluated through the primitive harness | Model and archived WikiText arrays |
| `experiments.compose.opus_controls` | Decomposition of the first designer programme | Completed `opus_program/log.json` in `WSP_OUT` |

For example, `python -m experiments.compose.lookup3_controls` runs the corresponding controls with the current environment settings. Figure/table generators intentionally continue to read the archived result directories; they do not automatically switch to `WSP_OUT`.

## API-based designer

The recorded runs can be inspected in [`results/compose/opus_program/`](../results/compose/opus_program/), [`results/compose_run2/opus_program/`](../results/compose_run2/opus_program/) and [`results/compose_1p7b/opus_program/`](../results/compose_1p7b/opus_program/) without an API key.

To execute a new designer run, set `ANTHROPIC_API_KEY` in your environment and run `python -m experiments.compose.opus_program` with a fresh `WSP_OUT` and a unique `WSP_LABEL` (for example `local-routing-20261116_`). Spending and the per-run cap are grouped by this label prefix in the global ledger, so a fresh `WSP_OUT` alone does not separate accounting; with the default label a new run inherits the archived first run's spending. This makes paid API calls. The archived harness uses model identifier `claude-opus-5-5`, five rounds by default, and a $4 per-run cap. Model access, rates and accounting assumptions must be checked for a new run; the original ledger and rate constants describe the recorded study.

The harness also uses the global budget ledger at `results/evolution/ledger.json`. It can stop a run when historical commitments exceed its configured limit; a new `WSP_OUT` alone does not create a fresh global budget ledger. Inspect the budget configuration in [`experiments/evolution/editor.py`](../experiments/evolution/editor.py) before a new paid run. Do not modify the archived ledger to make it appear that historical calls were never made.

## Reproduction limits

The original loader did not pin model revisions or record a complete routing-study environment lockfile. CPU/GPU refits can differ slightly. Prompt seeds, generators, fitted matrices and saved measurements are provided, but exact numerical identity across fresh environments is not guaranteed. The paper distinguishes fitting, adaptive selection and confirmation, and documents the small preservation samples.

Existing software tests are in [`tests/`](../tests/). For a configured experiment environment, `python -m pytest tests/test_evolution.py` checks invariants of the earlier editing harness; it does not validate the routing paper's scientific claims. No full experiment suite is needed for manuscript-only updates.
