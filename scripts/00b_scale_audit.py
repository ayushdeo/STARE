"""
Acquisition-unit scale audit (revision; RUN ON 4070 -- reads h5ad only, no patches, no embedding).

The legacy loader converted full-res pixels to microns with a fixed 0.5 um/px, but HEST WSIs are
0.26-1.0 um/px (the Visium HD slides are ~0.274). So the legacy "112 um" grid was 112 * px / 0.5 um in
reality. For each slide in config.HEST_SAMPLE_IDS this reports, from the raw AnnData:

  * true px size (HEST estimate) and the median nearest-neighbour spot pitch in TRUE microns
    (Visium = 100 um; an independent check that the px size is right)
  * the legacy grid's true side length, its region count, and spots per region (mean / max)
  * whether legacy == spot mode (max one spot per legacy region): if so, the n=7 results were in
    effect computed on native spots and carry over to config.REGION_MODE = "spot" unchanged in unit.

Writes results/scale_audit.json.   Run:  python -u scripts/00b_scale_audit.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import numpy as np
from anndata import read_h5ad
from huggingface_hub import hf_hub_download
from stare import config, hest_meta
from stare.data import build_region_grid


def nn_pitch(coords):
    rng = np.random.default_rng(0)
    sub = coords[rng.choice(len(coords), size=min(2000, len(coords)), replace=False)]
    d = np.linalg.norm(sub[:, None] - sub[None], axis=2); np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(1)))


def main():
    rows = {}
    for tissue, ids in config.HEST_SAMPLE_IDS.items():
        for sid in ids:
            ad = read_h5ad(hf_hub_download(config.HEST_DATASET_REPO, f"st/{sid}.h5ad", repo_type="dataset"))
            px = np.asarray(ad.obsm["spatial"], dtype=np.float64) if "spatial" in ad.obsm else \
                ad.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(np.float64)
            if "in_tissue" in ad.obs.columns:
                px = px[ad.obs["in_tissue"].to_numpy(dtype=bool)]
            px_um, src = hest_meta.pixel_size_um(sid)
            row = hest_meta.sample_row(sid) or {}
            legacy_um = px * hest_meta.LEGACY_PX_UM
            rid, cent = build_region_grid(legacy_um, config.REGION_UM)
            spr = np.bincount(rid)
            r = {
                "tissue": tissue, "species": row.get("species"), "st_technology": row.get("st_technology"),
                "n_spots_in_tissue": int(len(px)),
                "px_um": px_um, "px_source": src,
                "nn_pitch_true_um": round(nn_pitch(px * px_um), 1),
                "legacy_region_true_side_um": round(config.REGION_UM * px_um / hest_meta.LEGACY_PX_UM, 1),
                "legacy_n_regions": int(len(cent)),
                "legacy_spots_per_region_mean": round(float(spr.mean()), 3),
                "legacy_spots_per_region_max": int(spr.max()),
            }
            r["legacy_equals_spot_mode"] = r["legacy_spots_per_region_max"] == 1
            rows[sid] = r
            print(f"{tissue:8s} {sid:9s} {str(r['species'])[:12]:12s} {str(r['st_technology'])[:13]:13s} "
                  f"spots={r['n_spots_in_tissue']:5d} px={px_um:.4f} pitch={r['nn_pitch_true_um']:6.1f}um "
                  f"legacy_side={r['legacy_region_true_side_um']:6.1f}um regions={r['legacy_n_regions']:5d} "
                  f"spots/region max={r['legacy_spots_per_region_max']} -> legacy==spot: {r['legacy_equals_spot_mode']}",
                  flush=True)
    non_human = [k for k, v in rows.items() if v["species"] not in ("Homo sapiens", None)]
    out = {"metadata_source": hest_meta.load_metadata().attrs.get("source"), "rows": rows,
           "all_legacy_equals_spot": all(v["legacy_equals_spot_mode"] for v in rows.values()),
           "non_human": non_human}
    p = Path(config.RESULTS_ROOT) / "scale_audit.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[write] {p}\nall legacy==spot: {out['all_legacy_equals_spot']} | non-human slides: {non_human or 'none'}")


if __name__ == "__main__":
    main()
