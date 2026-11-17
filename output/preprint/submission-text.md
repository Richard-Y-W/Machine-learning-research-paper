# Preprint submission text

Title: Routing Edits: Reusing a Successor Circuit through Attention Weight Changes

Author: Yucheng Wang

Affiliation: University of Wisconsin-Madison

Author description: Undergraduate, Mathematics and Computer Science

## Abstract

A small language model can continue c d e with f, and it can retrieve c from a table. Can the two computations be connected, so that a lookup returns the successor of the retrieved value, without training on the combined task? We test whether the model's own successor computation can be reused by changing only where it reads its input. In Qwen3-0.6B-Base and Qwen3-1.7B-Base we locate an attention head involved in sequence succession and rewrite its query and key weights so that it attends to the retrieved value, leaving its value and output weights unchanged apart from a gain. A table containing pine -> c, queried with pine ->, now often elicits d: on fresh prompts, successor rates rise from 1-12% to 46-69% (0.6B) and from 6-25% to 32-52% (1.7B). Controls and output swaps show that the effect depends on the head's existing computation, not on the new attention alone. We then give a second language model circuit-level editing primitives; in all three runs it builds working edits, and in each it proposes a design we had not built, copying the successor circuit into another head. The direct edit has a measurable cost (14-17% higher WikiText loss), and fine-tuning on composed examples works better. The result is a specific, causally tested case of reusing learned computation through a permanent weight change.

## Files

- PDF: `output/pdf/routing-edits-preprint.pdf`
- LaTeX source bundle: `output/preprint/routing-edits-source.tar.gz`
- Main source file within the bundle: `main.tex`
- References are embedded in `main.tex`; no BibTeX run is needed.
- The source bundle contains the manuscript, six figure PDFs and nine table files.

## Status

Prepared locally; not submitted or published. No experiments or model API runs were executed during this revision. Tables were regenerated from saved results, figures were regenerated to repair labels, and the PDF was compiled and visually reviewed.

Public access to the code depends on the repository's GitHub visibility setting; this script does not change that setting. Make the repository public before sharing the preprint's code link with readers. The manuscript discloses that original run-specific checkpoint revisions were not recorded by the loader; no provenance was invented after the fact.

## Editorial changes

Added author affiliation and PDF metadata; simplified the title and abstract; focused the introduction on circuit reuse; moved earlier exploratory studies to an appendix; separated fitting, gain selection and confirmation; clarified RMSNorm gains and rotary position effects; narrowed preservation and LLM-search claims; expanded the WikiText evaluation specification; checked and completed the flagged references against primary sources; added QK Product Steering and clarified the Function Vectors comparison. Quantitative experimental results are unchanged.
