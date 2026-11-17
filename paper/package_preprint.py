"""Package the current manuscript for upload; never runs experiments."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
OUT = ROOT / 'output/preprint'
PDF_OUT = ROOT / 'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)
PDF_OUT.mkdir(parents=True, exist_ok=True)

source = PAPER / 'main.tex'
tex = source.read_text(encoding='utf-8')
abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S).group(1).strip()
abstract = re.sub(r'\\texttt\{([^}]*)\}', r'\1', abstract)
abstract = abstract.replace(r'\%', '%').replace('``', '"').replace("''", '"').replace('--', '-')
abstract = re.sub(r'\s+', ' ', abstract)

files = [source, *sorted((PAPER/'figures').glob('*.pdf')), *sorted((PAPER/'tables').glob('*.tex'))]
for relative in re.findall(r'\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}', tex):
    target = PAPER / relative
    if not target.suffix:
        target = target.with_suffix('.tex')
    if not target.is_file():
        raise FileNotFoundError(target)

archive = OUT / 'routing-edits-source.tar.gz'
with tarfile.open(archive, 'w:gz') as bundle:
    for path in files:
        bundle.add(path, arcname=path.relative_to(PAPER).as_posix(), recursive=False)

pdf = PDF_OUT / 'routing-edits-preprint.pdf'
shutil.copy2(PAPER/'main.pdf', pdf)

metadata = f'''# Preprint submission text

Title: Routing Edits: Reusing a Successor Circuit through Attention Weight Changes

Author: Yucheng Wang

Affiliation: University of Wisconsin-Madison

Author description: Undergraduate, Mathematics and Computer Science

## Abstract

{abstract}

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
'''
(OUT/'submission-text.md').write_text(metadata, encoding='utf-8', newline='\n')
manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*files, pdf, archive]}
(OUT/'sha256.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
print(f'Packaged {len(files)} source files. Abstract: {len(abstract.split())} words.')
print(pdf)
print(archive)
