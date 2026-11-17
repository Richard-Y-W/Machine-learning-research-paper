# Manuscript and build files

The current paper is **Routing Edits: Reusing a Successor Circuit through Attention Weight Changes**, by **Yucheng Wang**, University of Wisconsin–Madison.

| File | Purpose |
| :--- | :--- |
| [`main.pdf`](main.pdf) | Compiled current preprint |
| [`main.tex`](main.tex) | Authoritative manuscript source, including references |
| [`make_tables.py`](make_tables.py) | Regenerate tables from saved JSON results |
| [`make_figures.py`](make_figures.py) | Regenerate figures from saved JSON results |
| [`make_readme_graphics.py`](make_readme_graphics.py) | Rebuild the README diagram and result-image export |
| [`package_preprint.py`](package_preprint.py) | Create a PDF copy, source bundle, submission text and checksums |
| [`figures/`](figures/) and [`tables/`](tables/) | Generated manuscript displays |

Build instructions and dependency notes are in the [reproduction guide](../docs/reproduction.md).

Files named `draft*`, `v4.template.md`, `wsp.template.md`, `numerical-baselines.md` and `references*` belong to earlier studies. They are preserved at their original paths because archived build scripts refer to them; they are not alternative versions of the current preprint. The current manuscript uses its own inline bibliography in `main.tex`.
