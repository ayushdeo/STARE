# Deviations log (REPRODUCTION_PLAN §6)

Every departure from a competitor's native setup or from the pinned config goes here:
**what changed · why · why it does not unfairly advantage us.** Empty until runs begin.

| Date | Component | Deviation | Justification | Unfair-advantage check |
|---|---|---|---|---|
| 2026-07-05 | SpatialFinder (Tier-2 port, `policies/oneshot.py`) | Replaced native cell-composition Shannon-entropy term (StarDist + PLIP + XGBoost cell classifier) with an image-embedding diversity proxy: k-means the frozen UNI embeddings into C clusters, reward rare-cluster membership + distance-to-centroid. | The cell-classifier pipeline is heavy and its gist is incomplete; the proxy preserves SpatialFinder's DIVERSITY intent and — critically — keeps it ONE-SHOT, image-only, and measured-expression-blind (the exact properties our closed-loop method is designed to beat). | Does NOT advantage us: the proxy is a diversity heuristic (favors SpatialFinder's own objective), still ignores measured expression, and is applied only to the competitor. **TODO before trusting head-to-head:** Tier-1 native sanity repro of SpatialFinder's ρ on ≥1 native tissue to confirm the port is faithful. |
| (template) | Component | what changed | why | why not unfair |
