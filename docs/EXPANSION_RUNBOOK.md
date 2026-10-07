# Expanded slide set — 4070 runbook (reviewer point 5)

The analysis plan is pre-registered in `docs/ITERATION_LOG.md` (iteration 7), committed before any
expanded result. Do not change the primary family or decision rules after seeing results.

## 0. Pull and authenticate
```bash
git pull
huggingface-cli login        # gated MahmoodLab/hest; needed for the v1.3 metadata + new slides
```

## 1. Select slides (minutes, metadata only)
```bash
python -u scripts/00_select_samples.py --apply
git diff src/stare/config.py     # review the picks; the printout lists exclusions and specimen keys
```
Curated: human, whole transcriptome, exact organ, >=1000 spots, WSI <= 0.6 um/px, one slide per
specimen, <= 2 per study. The original 7 stay unless the v1.3 metadata excludes one; the printout
says why (e.g. if one of them turns out to be non-human, report that in the paper).

## 2. Scale audit (minutes; h5ad only, no embedding)
```bash
python -u scripts/00b_scale_audit.py
```
Reports the true pixel size, spot pitch, and whether the legacy grid was one spot per region.

## 3. Reruns (CPU-heavy; run overnight, in this order)
Each new slide downloads once (~0.5-1 GB h5ad + patches) and is UNI2-h-embedded once (minutes);
all scripts share the per-slide cache in `cache/slides/` and k-means orders in `cache/placements/`.
```bash
python -u scripts/13_batched_contiguous.py   # the power-critical question: batched detection
python -u scripts/09_realdata_detection.py
python -u scripts/05_multislide_ci.py        # graphflow only by default (primary)
python -u scripts/10_competitors.py
python -u scripts/12_hvg_sensitivity.py
python -u scripts/14_power_summary.py        # local; Holm + clusters + n=7 comparison
```

## 4. Push
```bash
git add src/stare/config.py results/*.json docs/POWER_SUMMARY.md
git commit -m "4070: expanded slide set reruns"
git push
```
Do not commit `cache/` (gitignored). If any slide fails to build, keep going and note the error;
no slide is dropped after results are seen.

## Notes
- Unit: `config.REGION_MODE = "spot"` (one region = one native spot, true-micron coordinates). All
  slides, including the original 7, are rebuilt in this mode; the n=7 JSONs are archived in
  `results/n7_legacy/` for comparison.
- Policies, imputer and metrics were optimized for 4-9k-spot slides (sparse graph diffusion,
  incremental kernel sums, memoized placements). On a 132-run equivalence check (11 policies x 3
  imputers x 2 slide sizes x 2 budgets) the selections and scores are identical to the old code.
- knn/kriging imputers still use dense blocks; run them on the expanded set only if memory allows
  (`05_multislide_ci.py --imputers knn kriging graphflow`).
