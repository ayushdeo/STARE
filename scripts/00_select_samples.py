"""
Expand the slide set for statistical power (revision, reviewer point 5; RUN ON 4070).

HEST-1k has only ~10 Visium HD samples total, so more HD is not possible. But it has ~547 standard
Visium samples, and our loader is platform-agnostic (spots are pooled into 112 um regions regardless of
platform), so we expand with standard Visium across the same four tissues. This queries the live HEST
metadata, lists whole-transcriptome (Visium / Visium HD) candidates per tissue ranked by spot count, and
writes a ready-to-paste expanded HEST_SAMPLE_IDS for config.py.

Run:  python -u scripts/00_select_samples.py           # prints candidates + suggested dict
      python -u scripts/00_select_samples.py --per 8   # up to 8 samples per tissue
Then review results/sample_candidates.json, paste the suggested dict into config.HEST_SAMPLE_IDS, and
re-run the experiments (05/09/10/12/13) at the larger n. Note the resolution-mixing caveat in the paper:
Visium (~100 um pitch) and Visium HD (8 um bins) are both pooled onto the shared 112 um region grid.

REQUIRES: HuggingFace authentication with access to the gated MahmoodLab/hest dataset (run
`huggingface-cli login` on the 4070; the metadata CSV is gated). COST: each new slide must be
downloaded (h5ad + patches, ~100s of MB to GB) and UNI2-h-embedded once; expanding to ~30 slides is
tens of GB + hours of GPU embedding. Build caches incrementally and commit only the results JSONs.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from stare import config

WHOLE_TX = ["Visium HD", "Visium"]          # whole-transcriptome platforms (exclude panel-limited Xenium)
SPOT_COLS = ["nb_spots", "n_spots", "spots", "num_spots", "nb_cells", "n_cells"]  # try these for ranking


def load_meta():
    path = hf_hub_download(repo_id=config.HEST_DATASET_REPO, repo_type="dataset",
                           filename=config.HEST_METADATA_FILE)
    return pd.read_csv(path)


def tissue_match(df, tissue):
    t = tissue.lower()
    org = df.get("organ", pd.Series([""] * len(df))).fillna("").str.lower()
    tis = df.get("tissue", pd.Series([""] * len(df))).fillna("").str.lower()
    return df[org.eq(t) | tis.eq(t) | org.str.contains(t) | tis.str.contains(t)].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=8, help="max samples per tissue in the suggested dict")
    args = ap.parse_args()

    df = load_meta()
    tech = df["st_technology"].astype(str)
    df = df[tech.str.contains("|".join(WHOLE_TX), case=False, na=False)].copy()
    spot_col = next((c for c in SPOT_COLS if c in df.columns), None)
    print(f"[meta] {len(df)} whole-transcriptome samples; ranking column = {spot_col or 'none (id order)'}")

    candidates, suggested = {}, {}
    for tissue in config.SCALE_TISSUES:
        sub = tissue_match(df, tissue)
        if spot_col:
            sub = sub.sort_values(spot_col, ascending=False)
        cols = [c for c in ["id", "st_technology", "organ", "tissue", spot_col] if c and c in sub.columns]
        rows = sub[cols].to_dict("records")
        candidates[tissue] = rows
        # suggested: keep up to --per; HD first, then Visium, preserving the spot-count ranking
        hd = [r["id"] for r in rows if "hd" in str(r["st_technology"]).lower()]
        vis = [r["id"] for r in rows if "hd" not in str(r["st_technology"]).lower()]
        suggested[tissue] = (hd + vis)[:args.per]
        print(f"\n== {tissue}: {len(rows)} candidates (HD={len(hd)}, Visium={len(vis)}) ==")
        for r in rows[:args.per]:
            print(f"   {r.get('id'):12s} {str(r.get('st_technology')):10s} "
                  f"{str(r.get('organ',''))[:18]:18s} spots={r.get(spot_col) if spot_col else '?'}")

    out = Path(config.RESULTS_ROOT) / "sample_candidates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ranking_col": spot_col, "per_tissue_cap": args.per,
                               "candidates": candidates, "suggested_HEST_SAMPLE_IDS": suggested}, indent=2),
                   encoding="utf-8")
    print(f"\n[write] {out}")
    print("\nPaste this into config.HEST_SAMPLE_IDS (review first), then rebuild caches + re-run:")
    print("HEST_SAMPLE_IDS = " + json.dumps(suggested, indent=4))
    total = sum(len(v) for v in suggested.values())
    print(f"\n-> {total} slides across {len(suggested)} tissues (was 7). "
          f"Re-run 05/09/10/12/13; mi_greedy's O(R^3) step scales with regions/slide, so expect longer runs.")


if __name__ == "__main__":
    main()
