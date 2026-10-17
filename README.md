# STARE: evaluating feedback-driven region selection for spatial transcriptomics

Code, results, and registered analysis plans for the preprint

> **Where to Measure Next? Evaluating Feedback-Driven Region Selection for Spatial Transcriptomics**
> Ayush Deo, University of Southern California.

The study asks whether choosing each next region of a tissue section from the expression already measured improves
on a fixed design, for two objectives: reconstructing expression across the section, and capturing molecularly
heterogeneous regions. It uses a retrospective acquisition simulator built from 22 human HEST-1k slides (18 studies;
prostate, kidney, lung, colon), a fixed image encoder (UNI2-h) and imputer, feedback ablations (permuted and constant
values), objective-matched non-adaptive comparators chosen on held-out studies, and study-level inference. The
paper source is in [`paper/`](paper/).

## What is in this repository

| Path | Contents |
|---|---|
| `src/stare/` | Simulator, data loading, imputers, policies, metrics, statistics |
| `scripts/` | Numbered experiment, analysis, and paper-asset scripts (below) |
| `results/*.json` | Every result the paper reports; the tables and figures regenerate from these alone |
| `paper/` | Manuscript, bibliography, generated tables, figures, arXiv packager output |
| `docs/ITERATION_LOG.md` | Dated record of every iteration, including the three registered analysis plans |
| `artifacts/sofish_seed222_5m/`, `sofisher_resume/` | Native SOFisher run (weights, TensorBoard logs) and the resume patch used for it |
| `tests/` | Unit tests for the statistics, policies, and data pipeline |

The other files in `docs/` are working notes kept as a record of how the project developed. They describe earlier
framings that the final results did not support; the paper and `docs/ITERATION_LOG.md` are authoritative.

## Registered analyses

Each stage's analysis plan was committed to `docs/ITERATION_LOG.md` before its experiments were run; the commit
history records the order.

| Stage | Plan | Experiments | Analysis | Paper assets |
|---|---|---|---|---|
| 1 | iteration 7 | `05`, `09`, `10`, `12`, `13` | `14_power_summary.py` | `15_paper_assets.py` |
| 2 | iteration 9 | `17_revision_experiments.py` | `18_revision_analysis.py`, `20_stage2_sensitivity.py` (post hoc) | `19_paper_assets_round2.py` |
| 3 | iteration 11 | `21_stage3_experiments.py` | `22_stage3_analysis.py` | `23_paper_assets_stage3.py` |

Synthetic positive controls: `07_positive_control.py`, `08_detection_control.py`. Native SOFisher curve:
`11_sofisher_curve.py`. Slide selection: `00_select_samples.py`; acquisition-unit audit: `00b_scale_audit.py`.
Scripts numbered `01`–`04`, `06`, `16`, and the unnumbered helpers belong to earlier development stages and are not
needed to reproduce the paper.

## Regenerating the paper from committed results (no data needed)

```bash
pip install -e .
python -u scripts/14_power_summary.py
python -u scripts/15_paper_assets.py
python -u scripts/18_revision_analysis.py
python -u scripts/20_stage2_sensitivity.py
python -u scripts/19_paper_assets_round2.py
python -u scripts/22_stage3_analysis.py
python -u scripts/23_paper_assets_stage3.py
```

Then compile `paper/manuscript.tex` (see [`paper/README.md`](paper/README.md)). `scripts/24_arxiv_package.py` builds
the arXiv source package and test-compiles it in a clean folder.

## Rerunning the experiments

Rerunning requires access to two gated resources on Hugging Face: the HEST-1k dataset (`MahmoodLab/hest`, including
the v1.3 metadata) and the UNI2-h encoder (`MahmoodLab/UNI2-h`). Install PyTorch for your CUDA version first, then:

```bash
pip install -r requirements.txt
pip install -e .
python -u scripts/00_select_samples.py           # reproduces the 22-slide selection
python -u scripts/05_multislide_ci.py            # stage 1 (also 09, 10, 12, 13)
python -u scripts/17_revision_experiments.py --part A   # stage 2 (then --part B)
python -u scripts/21_stage3_experiments.py --part H     # stage 3 (then --part S, --part R)
```

The experiments were run on a laptop with an NVIDIA RTX 4070. Slide embeddings and placements are cached under
`cache/` (not committed); experiment scripts checkpoint and resume. All hyperparameters and seeds are fixed in
`src/stare/config.py`.

## Citation

If you use this code, please cite the preprint (citation details will be added once it is posted).
