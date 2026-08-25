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

from . import config
from .simulator import Slide


def list_hest_samples(tissue: str) -> list[str]:
    """Query HEST metadata for Visium HD samples of `tissue` (oncotree filter).
    TODO(4070): use `datasets`/`hest` to read metadata CSV, filter
    st_technology == 'Visium HD' and organ/oncotree == tissue. Fill config.HEST_SAMPLE_IDS."""
    raise NotImplementedError("Stage-0 TODO: implement via hest metadata; see scripts/00_pick_samples.py")


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
    """Load one HEST sample into a `Slide` on the pinned region grid.
    Steps (TODO on 4070):
      1. read AnnData (raw counts) + WSI via hest;
      2. build_region_grid() to pool bins -> regions;
      3. sum counts per region, log1p-normalize, drop genes <MIN_SPOT_FRACTION;
      4. compute top-N_HVG per slide;
      5. extract one PATCH_PX@TARGET_MAG patch per region centroid -> UNI embedding (cached);
      6. return Slide(...).
    """
    raise NotImplementedError("Stage-0 TODO: implement HEST->Slide; cache embeddings under config.CACHE_ROOT")


def load_tissue(tissue: str) -> list[Slide]:
    """Load all pinned samples for a tissue (config.HEST_SAMPLE_IDS[tissue])."""
    ids = config.HEST_SAMPLE_IDS.get(tissue) or []
    if not ids:
        raise RuntimeError(
            f"No sample IDs pinned for '{tissue}'. Run scripts/00_pick_samples.py on the 4070 "
            f"to fill config.HEST_SAMPLE_IDS (CONFIRM_ON_MACHINE)."
        )
    return [load_slide(s, tissue) for s in ids]
