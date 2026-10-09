# paper/ — arXiv-ready manuscript (NeurIPS style, preprint mode)

Source for *Evaluating Adaptive Region Acquisition for Spatial Transcriptomics Reconstruction*
(sole author). `manuscript.tex` is the source of record for the prose.

## Files needed to compile (upload all of these to Overleaf / include in the arXiv tarball)
- `manuscript.tex` — the paper (main text + Appendices A–F).
- `references.bib` — bibliography. SOFisher and S2-omics checked against Crossref; SCR2-ST against arXiv
  (2026-10-09; arXiv lists no venue, so it is cited as a preprint).
- `neurips_2023.sty` — NeurIPS style file (must be bundled).
- `generated/*.tex` — every table, written by `../scripts/15_paper_assets.py`. Keep the folder name;
  the paper `\input`s `generated/tab_*.tex`. (`generated/facts.json` is not needed to compile.)
- `fig_controls.pdf`, `fig_main.pdf`, `fig_sofisher_native.pdf` — figures, written by the same script.

`figure1_adaptivity_gap.pdf` / `.zip` belong to the earlier 7-slide draft and are no longer used.

## Regenerating numbers (never edit tables by hand)
```bash
python -u scripts/14_power_summary.py    # Holm / study-cluster statistics -> results/power_summary.json
python -u scripts/15_paper_assets.py     # tables + figures -> paper/generated/, paper/fig_*.pdf
```
Both read only the committed `results/*.json`; no data download is needed.

## Compile
natbib numeric citations with `plainnat`. Overleaf: upload the files above (with the `generated/`
folder) and recompile. Locally: `pdflatex manuscript && bibtex manuscript && pdflatex manuscript &&
pdflatex manuscript`, or `tectonic -X compile manuscript.tex`. Verified on 2026-10-09 with Tectonic
0.17: 11 pages (main text ends on page 7), no overfull boxes, no undefined references.

## arXiv tarball
After one clean compile (which produces `manuscript.bbl`):
```bash
tar czf arxiv_submission.tar.gz manuscript.tex manuscript.bbl neurips_2023.sty generated/*.tex fig_controls.pdf fig_main.pdf fig_sofisher_native.pdf
```
Suggested categories: cs.LG, cross-list q-bio.GN and q-bio.QM.

## Pre-submission checklist
- [ ] Compiles on Overleaf with the `generated/` folder present.
- [ ] Figures legible in grayscale (series also differ by marker shape).
- [ ] Tables regenerated after any change to `results/*.json`.
