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

## Level 2 — native reproduction (heavier; the authors' own code + data)

Reproduces each paper's reported metric to confirm the objective they optimize. Run in separate conda
envs so their deps do not collide with STARE.

### SOFisher
```bash
git clone https://github.com/Heathcliff-Ng/SOFisher && cd SOFisher
conda env create -f environment.yml -n sofisher-env && conda activate sofisher-env
pip install pysodb                      # downloads processed spatial-omics h5ad
python main_cortex.py                   # (also main_aging.py / main_disease.py / main_CRC.py)
# validate with visualize_cortex.ipynb ; record their ROI-capture metric
```
What to record: their reported **ROI-capture / information-gain** curve vs #FOVs. Note that the metric is
detection (regions of interest captured), not whole-slide expression reconstruction. If feasible, add a
reconstruction-PCC readout on the unprofiled FOVs and a strong non-adaptive placement (MI-greedy) as a
matched baseline, to show the objective split in their own setting.

### SCR2-ST
```bash
git clone https://github.com/hrlblab/SCR2ST && cd SCR2ST
pip install -r requirements.txt
python main.py --dataset HER2 --root_path <st> --patch_root <patches> --sc_root <sc_embeddings> \
               --total_ratio 0.5 --max_epochs 100 --batch_size 128 --gpu 0
```
Needs three inputs: ST data (`--root_path`), histology patches (`--patch_root`), single-cell embeddings
(`--sc_root`). What to record: their reported **prediction accuracy vs sampling ratio** on HER2 (and the
other two datasets), and the baselines they compare against (check whether they compare vs random/grid or
vs strong MI-greedy placement — our claim is the former).

### Reconciliation to write up
- Both methods optimize a **detection/diversity** objective, not whole-slide reconstruction against
  strong placement (their stated rewards; Level-1 confirms in our simulator; Level-2 confirms in theirs).
- Our oracle result (`07_positive_control.py`) shows reconstruction has ~no adaptivity headroom, so their
  gains cannot be reconstruction gains.
- The `09`/`10` real-data results show the same split on HEST.

---

## Ownership / caveats
- Level 1 is `scripts/10_competitors.py` (ours). Level 2 runs the authors' repos unmodified where
  possible; document any adaptation.
- Keep their code/data OUT of this repo (licenses, size); commit only the recorded numbers + notes.
