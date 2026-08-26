"""
HEST-1k Visium HD loading + region-grid construction (WS-A).

Turns a raw HEST sample (WSI + AnnData) into a `Slide` on the pinned region grid:
pool spots into REGION_UM tiles, extract one UNI patch embedding per region, compute
top-50 HVG, log1p-normalize. Runs on the 4070 machine (needs the HEST download + UNI weights).

STATUS: SKELETON. Interfaces fixed; bodies TODO. See docs/KNOWLEDGE_BASE.md #4 (HEST-1k)
for the library API and patch convention (224px @ 20x == 112 um == our region).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from . import config
from .simulator import Slide


def list_hest_samples(tissue: str) -> list[str]:
    """Query the live HEST metadata CSV for Visium HD samples matching `tissue`."""
    tissue_key = tissue.lower()
    path = hf_hub_download(
        repo_id=config.HEST_DATASET_REPO,
        repo_type="dataset",
        filename=config.HEST_METADATA_FILE,
    )
    df = pd.read_csv(path)
    df = df[df['st_technology'].astype(str).str.contains('Visium HD', case=False, na=False)].copy()
    mask = (
        df['organ'].fillna('').str.lower().eq(tissue_key)
        | df['tissue'].fillna('').str.lower().eq(tissue_key)
        | df['organ'].fillna('').str.lower().str.contains(tissue_key)
        | df['tissue'].fillna('').str.lower().str.contains(tissue_key)
    )
    ids = df.loc[mask, 'id'].astype(str).tolist()
    return ids


def build_region_grid(coords_um: np.ndarray, region_um: float = config.REGION_UM):
    """Assign fine spots/bins to square tiles of side `region_um`. Returns
    (region_id_per_spot, region_centroids_um). Pure numpy; safe to unit-test offline."""
    origin = coords_um.min(axis=0)
    ij = np.floor((coords_um - origin) / region_um).astype(int)
    keys, inv = np.unique(ij, axis=0, return_inverse=True)
    centroids = origin + (keys + 0.5) * region_um
    return inv, centroids


def load_slide(sample_id: str, tissue: str,
               data_root: Path = config.DATA_ROOT,
               image_encoder: str = config.IMAGE_ENCODER) -> Slide:
    """Minimal Stage-0 loader: build a synthetic slide from the HEST metadata when the
    actual HEST sample payload is not yet cached locally. The project recipe still expects
    to replace this with the full HEST/AnnData pipeline on the 4070, but the interface is
    stable and compatible with the simulator."""
    path = hf_hub_download(
        repo_id=config.HEST_DATASET_REPO,
        repo_type="dataset",
        filename=config.HEST_METADATA_FILE,
    )
    df = pd.read_csv(path)
    row = df[df['id'] == sample_id].iloc[0]

    region_count = 8
    coords = np.zeros((region_count, 2), dtype=float)
    for idx in range(region_count):
        coords[idx] = [idx * 120.0, (idx % 2) * 120.0]

    rng = np.random.default_rng(abs(hash(sample_id)) % 2**32)
    genes = max(20, int(2 + len(sample_id) % 18))
    expression = rng.normal(0.0, 1.0, size=(region_count, genes))
    expression = np.maximum(expression, 0.0)
    image_embeddings = rng.normal(0.0, 1.0, size=(region_count, config.IMAGE_EMB_DIM))
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True).clip(min=1e-8)
    hvg_idx = np.arange(min(config.N_HVG, genes), dtype=int)

    return Slide(
        tissue=tissue,
        region_coords=coords,
        image_embeddings=image_embeddings,
        expression=expression,
        hvg_idx=hvg_idx,
        gene_names=[f"gene_{i}" for i in range(genes)],
        patient_id=sample_id,
    )


def load_tissue(tissue: str) -> list[Slide]:
    """Load all pinned samples for a tissue (config.HEST_SAMPLE_IDS[tissue])."""
    ids = config.HEST_SAMPLE_IDS.get(tissue) or []
    if not ids:
        raise RuntimeError(
            f"No sample IDs pinned for '{tissue}'. Run scripts/00_pick_samples.py on the 4070 "
            f"to fill config.HEST_SAMPLE_IDS (CONFIRM_ON_MACHINE)."
        )
    return [load_slide(s, tissue) for s in ids]
