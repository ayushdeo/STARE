# Competitor reproduction — 4070 runbook

Two levels, both queued for the 4070. Level 1 (harmonized) confirms the reconciliation inside our
simulator under matched objectives; Level 2 (native) reproduces the authors' own reported numbers.

Verified targets (all real):
- **SOFisher** — Ng et al., Nat. Commun. 2026 (bioRxiv 2024.07.05.602236). Repo: `Heathcliff-Ng/SOFisher`.
  DQN over FOVs; reward = capture more **regions of interest** (detection objective).
- **SCR2-ST** — Zhu et al., MIDL 2026 / arXiv 2512.13635. Repo: `hrlblab/SCR2ST`. RL sampler reward =
  single-cell manifold coverage + cell-type diversity + spatial uniformity (**diversity/coverage**
  objective); prediction head = regression + retrieval.

---

## Level 1 — harmonized head-to-head (fast, in our simulator) — DO THIS FIRST

Ports of each method's stated acquisition rule live in `src/stare/policies/competitors.py` (policies
`scr2st`, `sofisher`). Run them against our baselines on BOTH objectives:

```bash
git pull && python -u scripts/10_competitors.py
git add results/competitors.json && git commit -m "4070: competitor head-to-head" && git push
```

Uses the cached slides from the `05`/`09` runs (no HEST re-download). Expected result (confirming the
reconciliation): the adaptive detection-seekers (`sofisher`, `value_greedy`) show
DETECT margin > 0 but RECON margin <= 0 vs the best coverage baseline; `scr2st` (diversity coverage)
does not beat strong placement on reconstruction. Reported per budget with 95% CIs.

Honesty flag for the paper: these are **harmonized reimplementations of the published acquisition
objective**, not the authors' trained models. State this explicitly; use Level 2 for the authors' own
numbers.

---

## Level 2 — native reproduction (the authors' own code + data)

Run each repo AS THE AUTHORS RUN IT — their scripts, their data, their metric, their default
hyperparameters. Do not adapt their code except to point at data or fix an install error; log any change.
Use two separate conda envs so their deps do not collide with STARE or each other. Verified entry points
and arguments below (from the repos, 2026-09).

### SOFisher (self-contained; run this one first)
Faithful metric = **episode reward = regions-of-interest captured** (a DETECTION objective), saved to
`data/{agent}/seed_{seed}/eval/evaluate_episode_rewards.npy`. Data: 48 bundled mouse-brain slices via
`pysodb`; pretrained models in `/model`. This is a DQN, so a full retrain is optional — the pretrained
models + eval reproduce their curves.
```bash
git clone https://github.com/Heathcliff-Ng/SOFisher && cd SOFisher
conda env create -f environment.yml -n sofisher-env && conda activate sofisher-env
pip install pysodb
ls -R data model src                     # INSPECT: confirm bundled data + pretrained checkpoints exist
python main_cortex.py                    # then main_aging.py / main_disease.py / main_CRC.py
jupyter nbconvert --to notebook --execute visualize_cortex.ipynb   # reproduce their figures
```
Hardcoded in main_cortex.py: `args.rs=100`, `args.cell_num=21`; obs = grid_x*grid_y*cell_num+2+17,
24 discrete actions. RECORD: the eval episode-reward curve (ROI capture) vs the paper; confirm the
optimized quantity is ROI capture, not whole-slide expression reconstruction.

### SCR2-ST
Faithful args (defaults): `--dataset {HER2,BC,Kidney}` (default HER2), `--total_ratio {0.15,0.3,0.5,0.75,1.0}`
(default 0.5), reward weights `--w_sc 20 --w_type 5 --w_spatial 0.05` (single-cell manifold >> cell-type >>
spatial), `--max_epochs 100 --batch_size 128 --base_lr 5e-5 --seed 42`. Metric = **MSE / PCC / MAE per
sampling ratio**, written to `slidewise_metrics/epoch_{N}.csv` and `merged_metrics/epoch_{N}.xlsx`.
```bash
git clone https://github.com/hrlblab/SCR2ST && cd SCR2ST
conda create -n scr2st-env python=3.10 -y && conda activate scr2st-env
pip install -r requirements.txt
cat README* ; ls -R ; grep -rn "download\|gdown\|zenodo\|figshare\|drive.google" .   # INSPECT: find how to get data
# DATA CAVEAT: default paths are the authors' cluster (/data/zhuj28/...). You must obtain their processed
# HER2/BC/Kidney data in "our_format_final" NPY (position, expression, cell type, embeddings) + patches +
# sc embeddings. If the repo has no download, HER2 == HER2ST (public) but their preprocessing must be
# reproduced, OR request the processed data from the authors. Do NOT fake it.
python main.py --dataset HER2 --total_ratio 0.15 --gpu 0   # then 0.3, 0.5; sweep the ratio
```
RECORD: their PCC-vs-sampling-ratio curve, AND — critically — which baselines they compare against
(inspect the repo/paper: random / uniform / grid vs a strong MI-greedy placement). Our claim is that
their gains are vs weak baselines and/or on their manifold-coverage objective, not vs strong placement.

### STATUS (2026-09-26)
- **SOFisher native: DONE (5M steps, seed 222, cortex).** Ran to completion on the 4070 (resumable via
  `sofisher_resume/apply_resume.py`). The ROI-capture eval reward rises from a median of -238 (untrained)
  to a +147 median late-training plateau (best 514.2), crossing zero by ~0.3M steps. This natively
  confirms SOFisher optimizes and succeeds at a DETECTION objective (regions of interest captured), not
  whole-slide reconstruction. Evidence: `results/sofisher_native.json`, `docs/figures/figure4_sofisher_native.png`
  (from `scripts/11_sofisher_curve.py` over the committed TB logs in `artifacts/sofish_seed222_5m/`).
  Reward is noisy with rare catastrophic episodes (~-1.7e5); medians are the robust trend.
- **SCR2-ST native: NOT REPRODUCED (data unavailable).** The repo's default paths are the authors'
  cluster (`/data/zhuj28/...`); no public download of their processed HER2/BC/Kidney data
  (`our_format_final` NPY + single-cell embeddings) was found, and reproducing their exact preprocessing
  faithfully is not feasible without it. We do NOT claim a native reproduction. State this plainly in the
  paper's limitations.

### How we position SCR2-ST without a native run (honest)
The reconciliation for SCR2-ST rests on evidence we DO have, none of it requiring their data:
1. Their sampler's stated reward is diversity/coverage with `w_sc=20 >> w_type=5 >> w_spatial=0.05`
   (verified from `main.py` defaults): single-cell-manifold coverage dominates. Not reconstruction error.
2. Level-1 harmonized port (`10_competitors.py`, real HEST): the diversity objective does not beat strong
   placement on reconstruction and is coverage-like on detection.
3. Our oracle result (`07`): reconstruction has ~no adaptivity headroom, so any reported gain cannot be a
   reconstruction-vs-strong-placement gain.
4. Note (from their code) that SCR2-ST reports PCC vs a *sampling ratio*; the open question a reader will
   have is their BASELINE — check the paper text for whether they compare vs random/uniform/grid rather
   than vs MI-greedy placement. Cite that comparison honestly; do not assert beyond what their paper shows.

### Reconciliation to write up
- SOFisher optimizes ROI capture (detection) — confirmed natively via its eval reward.
- SCR2-ST's reward is manifold-coverage-dominant (w_sc=20) and reports PCC vs a sampling ratio; record its
  baselines to show the comparison is not against strong placement.
- Our oracle result (`07`) shows reconstruction has ~no adaptivity headroom; `09`/`10` show the split on
  HEST. Together: prior positive results live on detection/diversity or vs weak baselines; our negative
  result is vs strong placement on reconstruction. Both correct under their own objective.

---

## Ownership / caveats
- Level 1 is `scripts/10_competitors.py` (ours). Level 2 runs the authors' repos unmodified where
  possible; document any adaptation.
- Keep their code/data OUT of this repo (licenses, size); commit only the recorded numbers + notes.
