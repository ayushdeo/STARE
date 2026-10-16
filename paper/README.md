# paper/ — arXiv-ready manuscript (NeurIPS style, preprint mode)

Source for *Where to Measure Next? Evaluating Feedback-Driven Region Selection for Spatial Transcriptomics*
(sole author). `manuscript.tex` is the source of record for the prose.

## Files needed to compile (upload all of these to Overleaf / include in the arXiv tarball)
- `manuscript.tex` — the paper (main text + Appendices A–L).
- `references.bib` — bibliography (49 entries, all cited). Each entry added on 2026-10-16 carries its verification
  source (Crossref, arXiv, JMLR, or NeurIPS proceedings) in its `note`; notes that say "to be checked by the author"
  mark details that could not be confirmed online. SOFisher and S2-omics checked against Crossref; SCR2-ST against arXiv.
- `neurips_2023.sty` — NeurIPS style file (must be bundled).
- `generated/*.tex` — every table, written by `../scripts/15_paper_assets.py`. Keep the folder name;
  the paper `\input`s `generated/tab_*.tex`. (`generated/facts.json` is not needed to compile.)
- `fig_controls.pdf`, `fig_r2.pdf`, `fig_s3.pdf`, `fig_sofisher_native.pdf` — figures (`fig_r2.pdf` from
  `scripts/19_paper_assets_round2.py`, `fig_s3.pdf` from `scripts/23_paper_assets_stage3.py`, the others from
  `scripts/15_paper_assets.py`). `generated/r2_*.tex` come from script 19; `generated/s3_*.tex` and
  `generated/s2_sensitivity.tex` from script 23.

`figure1_adaptivity_gap.pdf` / `.zip` belong to the earlier 7-slide draft and are no longer used.

## Regenerating numbers (never edit tables by hand)
```bash
python -u scripts/14_power_summary.py    # Holm / study-cluster statistics -> results/power_summary.json
python -u scripts/15_paper_assets.py     # stage-1 tables + figures -> paper/generated/, paper/fig_*.pdf
python -u scripts/18_revision_analysis.py      # stage-2 registered analysis -> results/revision_summary.json
python -u scripts/19_paper_assets_round2.py    # stage-2 tables + main figure -> paper/generated/r2_*.tex, paper/fig_r2.pdf
python -u scripts/20_stage2_sensitivity.py     # post hoc stage-2 inference checks -> results/stage2_sensitivity.json
python -u scripts/22_stage3_analysis.py        # stage-3 registered analysis -> results/stage3_summary.json
python -u scripts/23_paper_assets_stage3.py    # stage-3 tables + figure -> paper/generated/s3_*.tex, paper/fig_s3.pdf
```
All read only the committed `results/*.json`; no data download is needed.

## Compile
natbib numeric citations with `plainnat`. Overleaf: upload the files above (with the `generated/`
folder) and recompile. Locally: `pdflatex manuscript && bibtex manuscript && pdflatex manuscript &&
pdflatex manuscript`, or `tectonic -X compile manuscript.tex`. Verified on 2026-10-16 with Tectonic
0.17: 20 pages (main text ends on page 10), no overfull boxes, no undefined references.

## arXiv tarball
After one clean compile (which produces `manuscript.bbl`):
```bash
tar czf arxiv_submission.tar.gz manuscript.tex manuscript.bbl neurips_2023.sty generated/*.tex fig_controls.pdf fig_r2.pdf fig_s3.pdf fig_sofisher_native.pdf
```
Suggested categories: cs.LG, cross-list q-bio.GN and q-bio.QM.

## Pre-submission checklist
- [ ] Compiles on Overleaf with the `generated/` folder present.
- [ ] Figures legible in grayscale (series also differ by marker shape).
- [ ] Tables regenerated after any change to `results/*.json`.
