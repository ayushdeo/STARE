# paper/ — manuscript (TMLR format)

Source for *Where to Measure Next? Evaluating Feedback-Driven Region Selection for Spatial Transcriptomics*
(sole author), formatted for Transactions on Machine Learning Research. `manuscript.tex` is the source of record.
The submission workflow, form fields, and checklist are in [`TMLR_SUBMISSION.md`](TMLR_SUBMISSION.md).

## Files needed to compile
- `manuscript.tex` — the paper (main text, references, Appendices A–L).
- `references.bib` — bibliography (50 entries, all cited; metadata checked against Crossref, arXiv, JMLR, PMLR,
  and NeurIPS proceedings on 2026-10-26).
- `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` — TMLR style files, unmodified, from
  https://github.com/JmlrOrg/tmlr-style-file.
- `generated/*.tex` — every table, written by the scripts below; never edit by hand.
- `fig_controls.pdf`, `fig_r2.pdf`, `fig_s3.pdf`, `fig_sofisher_native.pdf` — figures.

`\usepackage{tmlr}` gives the anonymous review version; `[accepted]` the camera-ready (set `\month`, `\year`, and
`\openreview`, and replace `CONTACT-EMAIL`); `[preprint]` a de-anonymized preprint.

## Regenerating tables and figures (no data download)
```bash
python -u scripts/14_power_summary.py          # stage-1 statistics -> results/power_summary.json
python -u scripts/15_paper_assets.py           # stage-1 tables + figures
python -u scripts/18_revision_analysis.py      # stage-2 registered analysis -> results/revision_summary.json
python -u scripts/20_stage2_sensitivity.py     # post hoc stage-2 inference checks -> results/stage2_sensitivity.json
python -u scripts/19_paper_assets_round2.py    # stage-2 tables + figure
python -u scripts/22_stage3_analysis.py        # stage-3 registered analysis -> results/stage3_summary.json
python -u scripts/23_paper_assets_stage3.py    # stage-3 tables + figure
```

## Compile
`pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript`, or
`tectonic --keep-intermediates manuscript.tex`. Verified on 2026-10-26 with Tectonic 0.17: 21 pages, main body
ends on page 11 (TMLR's longer review timeline applies only above 12 pages), no overfull boxes, no undefined
references.

## Submission materials
- `python scripts/25_tmlr_supplement.py` writes `tmlr_supplementary.zip`: the anonymized code, results, and
  registered plans, plus `REGISTRATION_LOG.md` (commit dates of each plan and its results) and checks the archive
  for identifying strings. Tables regenerate byte-for-byte from the archive.
- `python scripts/24_arxiv_package.py` builds a de-anonymized arXiv package for later use; see
  [`ARXIV_SUBMISSION.md`](ARXIV_SUBMISSION.md).
