# Stage 4: registered plan

Written before any stage-4 experiment is run. Experiments: `scripts/27_stage4_experiments.py`; analysis:
`scripts/28_stage4_analysis.py` (both committed with this plan). Results: `results/stage4_runs.json`,
`results/stage4_summary.json`.

## Purpose

Stage 4 tests whether the stage-1 to stage-3 conclusions survive five weaknesses of those designs:

| Weakness | Part |
|---|---|
| Reconstruction was scored on each policy's own unprofiled regions, so policies were compared on different region sets, and PCC falls on low-variance sets | E: common held-out set, PCC and MSE |
| Permuted feedback can mislead a policy, so "real beat permuted" does not show that the values added information | E, T: constant-feedback controls |
| No estimate of how much any selection could gain in reconstruction | U: ground-truth greedy upper bound |
| Single spots are not an acquisition unit any platform offers, and rounds of 5 at 10-20% budgets are nearly sequential | T: 300 um tiles, 12-48 tiles, 2 and 4 rounds |
| The local-variance policy was used with fixed, untuned settings | P: settings chosen per held-out study |

## Experiments

**Part E (22 slides, budgets 0.5, 1, 2, 5, 10, 20%).** Per slide, a fixed random 20% of regions (seed: global seed
and slide position) is held out: no policy may select it, and every policy is scored on it. Policies select from
the remaining 80%; the budget is round(b x N) of all N regions. Runs: grid, k-means, variance-greedy, MI-greedy,
one-shot; residual with real, permuted (seed 1), and constant feedback; half-coverage reconstruction hybrid
(stage-3 fixed designs) with real, permuted, and constant feedback; at 2-20%, local-variance and the half-coverage
detection hybrid, each with real, permuted, and constant feedback. Scores: held-out PCC over the top-50 genes,
held-out MSE over the top-50 genes, and ROI recall among selectable regions (stage-2 ROI definition).

**Part T (22 slides).** Spots pooled into 300 um square tiles with at least 3 spots (tile expression and
embedding = means over spots). Budgets 12, 24, 48 tiles. ROIs = the 20% of tiles with the highest within-tile
heterogeneity (mean variance of the top-50 genes across the tile's spots). 20% of tiles held out for reconstruction
as in Part E. Runs: the five non-adaptive designs; residual and local-variance sequentially, in 4 rounds, and in 2
rounds; their permuted and constant versions sequentially.

**Part P (22 slides, budgets 2, 5, 10, 20%, stage-2 protocol).** Local-variance with beta in {0.25, 0.5, 1, 2} and
kernel width in {0.05, 0.1, 0.2} of the slide extent (12 settings). Recall only.

**Part U (4 slides: TENX171 lung, TENX155 colon, TENX174 prostate, TENX173 kidney; budgets 0.5, 1, 2%).** Each
step adds, among 48 randomly drawn selectable candidates, the region that most increases held-out PCC computed
with the true expression. Same hold-out as Part E. Not deployable, and optimistic by design: it selects on the same
held-out score it is evaluated on, so it overstates what any real policy could reach.

## Analysis

Margins are per-slide differences; inference is at the level of the 18 studies: exact two-sided sign-flip test,
percentile bootstrap interval (5,000 resamples of studies) and sign-flip-inversion interval, both 95% and
unadjusted; Holm within each family; minimum detectable effect at 80% power and alpha divided by family size.
"Feasible" means the non-adaptive design with the best mean on the other studies, chosen per held-out study and per
budget, within the same part and on the same score.

**E primary (12):** residual - feasible (held-out PCC), 6 budgets; reconstruction hybrid - the same fixed design
(held-out PCC), 6 budgets.
**E secondary (30):** feasible - residual (held-out MSE; positive favours residual), 6; residual - constant feedback
(PCC), 6; reconstruction hybrid - constant feedback (PCC), 6; local-variance - constant feedback (recall), 4;
detection hybrid - fixed design (recall), 4; detection hybrid - constant feedback (recall), 4.
**T primary (9):** at 12, 24, 48 tiles: residual - feasible (held-out tile PCC); local-variance - feasible
(tile-ROI recall); local-variance in 2 rounds - feasible (recall).
**T secondary (9):** at each tile budget: residual - constant feedback (PCC); local-variance - constant feedback
(recall); local-variance in 4 rounds - feasible (recall).
**P primary (4):** tuned local-variance - feasible detection design (recall; stage-2 runs), 2, 5, 10, 20%, with the
setting chosen per held-out study by mean recall on the other studies.
**Descriptive:** residual - permuted feedback (E); local-variance - feasible in held-out PCC (E), to be compared with
the stage-2 deficit; recall above the random expectation (T); greedy upper bound - best non-adaptive design and -
residual in held-out PCC (U).

## Decision rules

- A benefit of feedback is claimed only for a positive margin with Holm-adjusted p < 0.05. Otherwise the margin is
  "not established" and reported with both intervals and the family-wise MDE.
- RQ1 and RQ3 (reconstruction): if every E primary margin is non-positive or not established, the paper states that
  the reconstruction conclusions hold on a common evaluation set. If a residual or hybrid margin is positive with
  p_H < 0.05, the paper's RQ1 or RQ3 conclusion is revised accordingly.
- Evaluation-set artefact: the local-variance PCC deficit on the common set (E descriptive) is reported next to its
  stage-2 deficit; the paper states the size of any change and does not attribute it causally beyond that.
- Value of feedback: "the values added information" is claimed only where real minus constant feedback is positive
  with p_H < 0.05 (E or T secondary).
- Headroom: the U gaps are reported descriptively; the paper uses them only to state how much room a better
  selection had under this imputer on these four slides.
- ROI scale (T): the T primary family answers RQ1, RQ2, and RQ4 at ROI scale; its outcome is reported alongside, not
  merged with, the spot-level families.
- Tuned policy (P): if tuned local-variance beats the feasible design with p_H < 0.05, the paper reports that tuning
  changes the detection conclusion.
- No slide is dropped after results are seen; a tile budget larger than half a slide's selectable tiles is skipped
  for that slide and reported.

## Run order on the GPU machine

```bash
git pull
python -u scripts/27_stage4_experiments.py --part E
python -u scripts/27_stage4_experiments.py --part T
python -u scripts/27_stage4_experiments.py --part P
python -u scripts/27_stage4_experiments.py --part U
git add results/stage4_runs.json && git commit -m "stage 4 runs" && git push
```

Parts are independent and checkpoint after every slide and budget, so any part can be interrupted and resumed, and
results can be pushed after each part. Part U prints the time of its first step; it is the slowest per slide.
