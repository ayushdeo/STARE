"""
Curated expansion of the slide set for statistical power (revision, reviewer point 5; RUN ON 4070).

v1 of this script ranked raw metadata matches and suggested slides that must NOT be used: a mouse
kidney (TENX146), a specimen duplicate (TENX128 == TENX155, identical WSI dimensions), serial sections
of one patient, a targeted gene panel, and deliberately degraded-RNA studies. v2 curates:

  * human only; whole-transcriptome Visium / Visium HD (titles with "targeted"/"panel" excluded)
  * exact organ match (mixed "Kidney/Brain"-style organs excluded); degraded-RNA studies excluded
  * >= --min_spots spots under tissue (small slides make budget fractions meaningless)
  * source WSI <= --max_px um/px (HEST patches from coarser scans are upsampled; keeps image quality even)
  * one slide per SPECIMEN (hest_meta.specimen_key + identical WSI dimensions => same specimen)
  * at most --max_per_study slides per study, filled round-robin across studies (more independent
    clusters per slide = more power under the study-clustered statistics)
  * Visium HD first, then standard Visium; the original 7 slides are kept if they pass the filters

The acquisition unit is a native (pseudo-)spot with true-micron coordinates (config.REGION_MODE =
"spot"), so Visium and Visium HD slides are on the same ~55 um spot / 100 um pitch footing.

Run:  python -u scripts/00_select_samples.py                 # prints per-tissue picks + paste-ready dict
      python -u scripts/00_select_samples.py --per 6         # target slides per tissue (default 6)
      python -u scripts/00_select_samples.py --apply         # also writes them into config.py
Writes results/sample_candidates.json. REQUIRES `huggingface-cli login` with access to the gated
MahmoodLab/hest dataset for the current (v1.3) metadata; without it the public v1.1 CSV is used, which
lacks samples added later (several of our Visium HD slides) -- the script says which source it used.
COST of the expanded run: ~0.5-1 GB download per new Visium slide (h5ad + patches); UNI2-h embedding
of ~3-6k patches/slide is minutes on the 4070.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import pandas as pd
from stare import config, hest_meta

ORGANS = {   # tissue -> accepted (organ, tissue) lowercase values
    "prostate": {"organ": {"prostate"}, "tissue": {"prostate"}},
    "kidney":   {"organ": {"kidney"}, "tissue": {"kidney"}},
    "lung":     {"organ": {"lung"}, "tissue": {"lung", "parenchyma"}},   # airway (bronchi/trachea) excluded
    "colon":    {"organ": {"bowel", "colon"}, "tissue": {"colon", "colorectal", "rectum", "bowel"}},
}
EXCLUDE_TITLE = ["targeted", "gene signature panel", "panel", "degraded"]


def exclusion_reason(r, tissue, min_spots, max_px=0.6):
    if str(r.get("species")) != "Homo sapiens":
        return f"species={r.get('species')}"
    tech = str(r.get("st_technology"))
    if "visium" not in tech.lower():
        return f"technology={tech}"
    org, tis = str(r.get("organ") or "").lower(), str(r.get("tissue") or "").lower()
    if "/" in org or "/" in tis:
        return f"mixed organ ({r.get('organ')})"
    acc = ORGANS[tissue]
    if not (org in acc["organ"] and (tis in acc["tissue"] or tis in acc["organ"] or not tis)):
        return f"organ/tissue mismatch ({r.get('organ')}/{r.get('tissue')})"
    title = str(r.get("dataset_title") or "").lower()
    for w in EXCLUDE_TITLE:
        if w in title:
            return f"title contains '{w}'"
    px = r.get("pixel_size_um_estimated")
    if px is not None and pd.notna(px) and float(px) > max_px:
        return f"low-res WSI px={float(px):.2f}um > {max_px}"   # HEST patches would be upsampled
    sp = r.get("spots_under_tissue")
    if sp is not None and pd.notna(sp) and float(sp) < min_spots:
        return f"spots_under_tissue={int(sp)} < {min_spots}"
    return None


def is_hd(r):
    return "hd" in str(r.get("st_technology")).lower()


def pick(rows, current, per, max_per_study):
    """rows: eligible dicts for one tissue. Returns ordered picks (dicts)."""
    picks, seen_spec, seen_dims, per_study = [], set(), set(), {}

    def take(r, force=False):
        dims = (r.get("fullres_px_width"), r.get("fullres_px_height"))
        has_dims = all(d is not None and pd.notna(d) for d in dims)
        if r["specimen"] in seen_spec or (has_dims and dims in seen_dims):
            return False
        if not force and per_study.get(r["study"], 0) >= max_per_study:
            return False
        picks.append(r); seen_spec.add(r["specimen"]); per_study[r["study"]] = per_study.get(r["study"], 0) + 1
        if has_dims:
            seen_dims.add(dims)
        return True

    by_id = {r["id"]: r for r in rows}
    for sid in current:                                   # continuity: keep the original slides
        if sid in by_id:
            take(by_id[sid], force=True)
    key = lambda r: -(float(r["spots_under_tissue"]) if pd.notna(r.get("spots_under_tissue")) else 0)
    for r in sorted([r for r in rows if is_hd(r)], key=key):   # all remaining HD next
        if len(picks) >= per:
            break
        take(r)
    vis = [r for r in rows if not is_hd(r)]
    studies = {}
    for r in sorted(vis, key=key):
        studies.setdefault(r["study"], []).append(r)
    while len(picks) < per and any(studies.values()):    # round-robin across studies
        for st in sorted(studies, key=lambda s: key(studies[s][0]) if studies[s] else 0):
            if len(picks) >= per:
                break
            while studies[st]:
                if take(studies[st].pop(0)):
                    break
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=6, help="target slides per tissue")
    ap.add_argument("--max_per_study", type=int, default=2)
    ap.add_argument("--min_spots", type=int, default=1000)
    ap.add_argument("--max_px", type=float, default=0.6, help="max um/px of the source WSI (image quality)")
    ap.add_argument("--apply", action="store_true", help="write the suggested IDs into src/stare/config.py")
    args = ap.parse_args()

    df = hest_meta.annotate(hest_meta.load_metadata())
    src = hest_meta.load_metadata().attrs.get("source")
    print(f"[meta] {len(df)} samples from {src}")
    cols = ["id", "st_technology", "species", "organ", "tissue", "disease_state", "patient", "subseries",
            "study", "specimen", "spots_under_tissue", "pixel_size_um_embedded", "pixel_size_um_estimated",
            "fullres_px_width", "fullres_px_height", "preservation_method"]
    out = {"metadata_source": src, "params": vars(args), "current": {}, "picks": {}, "excluded_counts": {},
           "suggested_HEST_SAMPLE_IDS": {}}

    for tissue in config.SCALE_TISSUES:
        current = list(config.HEST_SAMPLE_IDS.get(tissue) or [])
        loose = df[df["organ"].fillna("").str.lower().str.contains("|".join(ORGANS[tissue]["organ"]))
                   | df["tissue"].fillna("").str.lower().str.contains(tissue)]
        rows, reasons = [], {}
        for r in loose[cols + ["dataset_title"]].to_dict("records"):
            why = exclusion_reason(r, tissue, args.min_spots, args.max_px)
            if why:
                reasons[why.split("=")[0].split(" (")[0]] = reasons.get(why.split("=")[0].split(" (")[0], 0) + 1
            else:
                rows.append(r)
        # status of the slides already in use (incl. ones the metadata source does not list)
        cur_status = {}
        for sid in current:
            hit = df[df["id"] == sid]
            if hit.empty:
                cur_status[sid] = "NOT IN METADATA SOURCE (verify on the 4070 with the gated v1.3 CSV)"
            else:
                why = exclusion_reason(hit.iloc[0].to_dict(), tissue, args.min_spots, args.max_px)
                cur_status[sid] = f"EXCLUDE: {why}" if why else "keep"
        keep_current = [s for s in current if cur_status[s] == "keep" or cur_status[s].startswith("NOT IN")]
        unlisted = [s for s in keep_current if cur_status[s].startswith("NOT IN")]
        picks = pick(rows, keep_current, args.per - len(unlisted), args.max_per_study)
        ids = unlisted + [p["id"] for p in picks]  # unlisted current slides stay (cannot be checked here)
        out["current"][tissue] = cur_status
        out["excluded_counts"][tissue] = reasons
        out["picks"][tissue] = [{k: (None if (isinstance(v, float) and pd.isna(v)) else v)
                                 for k, v in p.items() if k != "dataset_title"} for p in picks]
        out["suggested_HEST_SAMPLE_IDS"][tissue] = ids
        n_st = len({p["study"] for p in picks})
        print(f"\n== {tissue}: {len(rows)} eligible after filters; excluded {reasons}")
        for sid, st in cur_status.items():
            print(f"   current {sid:9s} -> {st}")
        print(f"   picks ({len(ids)} slides, {n_st} studies):")
        for p in picks:
            print(f"     {p['id']:9s} {str(p['st_technology']):13s} spots={p.get('spots_under_tissue')!s:6s} "
                  f"px={p.get('pixel_size_um_estimated')!s:6.6s} {str(p.get('disease_state'))[:10]:10s} "
                  f"specimen={str(p['specimen'])[:70]}")

    path = Path(config.RESULTS_ROOT) / "sample_candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    sug = out["suggested_HEST_SAMPLE_IDS"]
    print(f"\n[write] {path}")
    print("\nPaste into src/stare/config.py (review the picks above first):")
    print("HEST_SAMPLE_IDS: dict[str, list[str]] = " + json.dumps(sug, indent=4))
    print(f"\n-> {sum(len(v) for v in sug.values())} slides (was 7). Next: scripts/00b_scale_audit.py, then the reruns.")
    if args.apply:
        import re
        cfg = Path(__file__).resolve().parents[1] / "src" / "stare" / "config.py"
        txt = cfg.read_text(encoding="utf-8")
        block = "HEST_SAMPLE_IDS: dict[str, list[str]] = {\n" + "".join(
            f'    "{t}": {json.dumps(v)},\n' for t, v in sug.items()) + "}\n"
        new, k = re.subn(r"HEST_SAMPLE_IDS: dict\[str, list\[str\]\] = \{\n(?:.*\n)*?\}\n", lambda _: block, txt, count=1)
        if k != 1:
            raise SystemExit("[apply] could not find the HEST_SAMPLE_IDS block in config.py")
        cfg.write_text(new, encoding="utf-8")
        print(f"[apply] wrote {sum(len(v) for v in sug.values())} IDs into {cfg} (review with git diff)")


if __name__ == "__main__":
    main()
