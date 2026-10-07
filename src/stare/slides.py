"""
Shared slide loading for the cross-slide experiments (05/09/10/12/13) with a PER-SLIDE cache.

Replaces the per-tissue `slides_{tissue}_...pkl` cache, which ignored which sample IDs it held: after
expanding config.HEST_SAMPLE_IDS the scripts would silently reload the old 7-slide pickle. Cache keys
are now (sample_id, region mode, encoder), so adding or removing an ID is always reflected.

Also returns the grouping needed by the revised statistics: `strata` (tissue) and `clusters` (HEST
study; serial sections / same-study slides are not independent, so the bootstrap and sign-flip test
resample whole studies -- see stats.paired_margin_summary).
"""
from __future__ import annotations

import pickle
from pathlib import Path

from . import config, hest_meta


def _cache_path(sample_id: str, mode: str) -> Path:
    return Path(config.CACHE_ROOT) / "slides" / f"{sample_id}_{mode}_{config.IMAGE_ENCODER}.pkl"


def _legacy_tissue_pickle(tissue: str) -> Path:
    return Path(config.CACHE_ROOT) / f"slides_{tissue}_{int(config.REGION_UM)}um_{config.IMAGE_ENCODER}.pkl"


def _migrate_legacy(tissue: str, log=print) -> None:
    """Split an old per-tissue pickle into per-slide 'legacy' caches (no recompute)."""
    p = _legacy_tissue_pickle(tissue)
    if not p.exists():
        return
    for s in pickle.load(open(p, "rb")):
        q = _cache_path(str(s.patient_id), "legacy")
        if not q.exists():
            q.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(s, open(q, "wb")); log(f"[migrate] {p.name} -> {q.name}")


def load_one(sample_id: str, tissue: str, mode: str | None = None, log=print):
    mode = mode or config.REGION_MODE
    q = _cache_path(sample_id, mode)
    if mode == "legacy" and not q.exists():
        _migrate_legacy(tissue, log)
    if q.exists():
        log(f"[cache] {q}")
        return pickle.load(open(q, "rb"))
    from . import data   # lazy: needs h5py/anndata/torch, only where a build is required (4070)
    log(f"[build] {sample_id} ({tissue}, mode={mode}) ...")
    s = data.load_slide(sample_id, tissue, mode=mode)
    q.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(s, open(q, "wb"))
    m = s.meta or {}
    log(f"[built] {sample_id}: {s.n_regions} regions, px={m.get('px_um')} ({m.get('px_source')}), "
        f"spots/region max={m.get('spots_per_region_max')}, {m.get('species')} {m.get('st_technology')}")
    return s


def load_ci_slides(tissues=None, mode: str | None = None, log=print):
    """All slides in config.HEST_SAMPLE_IDS for `tissues` (default config.CI_TISSUES).

    Returns (slides, labels, grouping) with grouping = {"strata": [tissue...], "clusters": [study...]}.
    """
    tissues = tissues or config.CI_TISSUES
    meta = hest_meta.annotate(hest_meta.load_metadata())
    study_of = dict(zip(meta["id"], meta["study"]))
    specimen_of = dict(zip(meta["id"], meta["specimen"]))
    slides, labels, strata, clusters, specimens = [], [], [], [], []
    for tissue in tissues:
        for sid in config.HEST_SAMPLE_IDS.get(tissue) or []:
            slides.append(load_one(sid, tissue, mode, log))
            labels.append(f"{tissue}:{sid}")
            strata.append(tissue)
            # unknown to the (possibly older public) metadata -> its own cluster (stated in the output)
            clusters.append(study_of.get(sid, f"unlisted:{sid}"))
            specimens.append(specimen_of.get(sid, f"unlisted:{sid}"))
    dup = {k for k in specimens if specimens.count(k) > 1 and not k.startswith("unlisted:")}
    if dup:
        log(f"[warn] same specimen appears more than once (not independent): {sorted(dup)}")
    log(f"[slides] n={len(slides)} across {len(set(strata))} tissues, {len(set(clusters))} study clusters "
        f"(metadata: {meta.attrs.get('source')})")
    return slides, labels, {"strata": strata, "clusters": clusters, "specimens": specimens}
