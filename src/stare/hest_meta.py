"""
HEST-1k sample metadata: pixel size, species, specimen identity (revision point 5).

Two facts from the metadata drive the expanded slide set:

1. PIXEL SIZE VARIES PER SLIDE (0.26-1.0 um/px). The legacy loader assumed a fixed 0.5 um/px
   (MICRONS_PER_PATCH / PATCH_PX), so its "112 um" grid was 112 * px/0.5 um in reality (~59 um on
   the 0.2646 um/px Visium HD WSIs). `pixel_size_um` gives the true per-slide value.
2. HEST's `patient` column is unreliable for clustering (e.g. the 10x CRC Visium HD study lists
   "Patient 1" for patients P1/P2/P3/P5; the subseries holds the real ID), and some samples are the
   same physical specimen (TENX128 == TENX155, identical WSI dimensions). `specimen_key` returns a
   conservative specimen/patient identity used for de-duplication and cluster-level statistics.

Metadata source: the gated v1.3 CSV on HuggingFace (needs `huggingface-cli login`); falls back to
the public v1.1 CSV on GitHub (older; lacks samples added after v1.1) so logic can be tested offline.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config

PUBLIC_META_URL = "https://raw.githubusercontent.com/mahmoodlab/HEST/main/assets/HEST_v1_1_0.csv"
LEGACY_PX_UM = config.MICRONS_PER_PATCH / config.PATCH_PX      # 0.5 um/px (the old fixed assumption)

# Known specimen identities the metadata cannot express (verified from subseries / WSI dimensions).
SPECIMEN_OVERRIDES = {
    "TENX156": "CRC-HD-study:P1",
    "TENX155": "CRC-HD-study:P2",
    "TENX154": "CRC-HD-study:P5",
    "TENX153": "CRC-HD-study:P3",
    "TENX152": "CRC-HD-study:P2",   # Visium CytAssist section of patient P2 (serial to TENX155)
    "TENX128": "CRC-HD-study:P2",   # 10x public HD CRC demo; same WSI dims as TENX155 -> same specimen
}

_SECTION_SUFFIX = re.compile(r"([\s_,\-]*(replicate|rep|section|slice|serial)?[\s_\-]*\d+)$", re.I)


@lru_cache(maxsize=1)
def load_metadata() -> pd.DataFrame:
    """HEST metadata as a DataFrame (gated v1.3 if authenticated, else public v1.1)."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=config.HEST_DATASET_REPO, repo_type="dataset",
                               filename=config.HEST_METADATA_FILE)
        df = pd.read_csv(path)
        df.attrs["source"] = config.HEST_METADATA_FILE
    except Exception as e:  # gated / offline -> public older release
        local = Path(config.CACHE_ROOT) / "HEST_v1_1_0.csv"
        if local.exists():
            df = pd.read_csv(local)
        else:
            df = pd.read_csv(PUBLIC_META_URL)
            try:
                local.parent.mkdir(parents=True, exist_ok=True); df.to_csv(local, index=False)
            except OSError:
                pass
        df.attrs["source"] = f"HEST_v1_1_0.csv (public fallback; gated v1.3 unavailable: {type(e).__name__})"
    df["id"] = df["id"].astype(str)
    return df


def sample_row(sample_id: str) -> dict | None:
    df = load_metadata()
    hit = df[df["id"] == sample_id]
    return None if hit.empty else hit.iloc[0].to_dict()


def pixel_size_um(sample_id: str) -> tuple[float, str]:
    """(um per full-res pixel, provenance). HEST's spot-spacing estimate first: it is present for every
    Visium sample, whereas the embedded value is missing for ~half and off by 30-2000x for some studies
    (e.g. 31.75 vs 0.455 for the cardiac-niche study). Embedded is used only if plausible."""
    row = sample_row(sample_id)
    if row is not None:
        for col in ("pixel_size_um_estimated", "pixel_size_um_embedded"):
            v = row.get(col)
            if v is not None and pd.notna(v) and 0.1 <= float(v) <= 4.0:
                return float(v), col
    return LEGACY_PX_UM, "legacy_assumption_0.5"


def _strip_section(s: str) -> str:
    return _SECTION_SUFFIX.sub("", s).strip() or s


def specimen_key(row: dict, study_patient_varies: bool) -> str:
    """Conservative specimen/patient identity for de-duplication and cluster statistics.

    Priority: manual override > `patient` (only when it actually varies within the study) >
    subseries with trailing section/replicate numbers stripped > the whole study (most conservative).
    """
    sid = str(row["id"])
    if sid in SPECIMEN_OVERRIDES:
        return SPECIMEN_OVERRIDES[sid]
    study = str(row.get("dataset_title") or "unknown-study")[:80]
    patient = row.get("patient")
    if study_patient_varies and patient is not None and pd.notna(patient):
        return f"{study}:{patient}"
    sub = row.get("subseries")
    if sub is not None and pd.notna(sub):
        return f"{study}:{_strip_section(str(sub))}"
    if sid.startswith("TENX"):        # 10x single-dataset releases: one specimen per dataset title
        return f"{study}"
    return study                      # unknown -> whole study is one cluster


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add `specimen` (dedup key) and `study` (cluster key) columns."""
    df = df.copy()
    df["study"] = df["dataset_title"].fillna("unknown-study").astype(str).str[:80]
    varies = df.groupby("study")["patient"].transform(lambda x: x.dropna().nunique() > 1)
    df["specimen"] = [specimen_key(r, bool(v)) for r, v in zip(df.to_dict("records"), varies)]
    return df
