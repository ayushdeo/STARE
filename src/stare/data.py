"""
HEST-1k Visium HD loading + region-grid construction (WS-A).

Turns a raw HEST sample (WSI + AnnData) into a `Slide` on the pinned region grid:
pool spots into REGION_UM tiles, extract one UNI patch embedding per region, compute
top-50 HVG, log1p-normalize. Runs on the 4070 machine (needs the HEST download + UNI weights).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata import read_h5ad
from huggingface_hub import hf_hub_download

from . import config
from .encoders import embed_patches
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


def _sample_hf_path(sample_id: str, filename: str) -> str:
    return hf_hub_download(
        repo_id=config.HEST_DATASET_REPO,
        repo_type="dataset",
        filename=filename.format(sample_id=sample_id),
    )


def _normalize_spatial_coords(coords_px: np.ndarray) -> np.ndarray:
    scales = config.MICRONS_PER_PATCH / config.PATCH_PX
    return coords_px.astype(np.float32) * scales


def _pooled_expression_by_region(expr: np.ndarray, region_id: np.ndarray, n_regions: int | None = None) -> np.ndarray:
    expr = np.asarray(expr, dtype=np.float32)
    if expr.ndim != 2:
        raise ValueError(f"expression must be (n_spots, n_genes); got {expr.shape}")
    if n_regions is None:
        n_regions = int(region_id.max()) + 1
    pooled = np.zeros((n_regions, expr.shape[1]), dtype=np.float32)
    counts = np.zeros(n_regions, dtype=np.int32)
    for i in range(n_regions):
        mask = region_id == i
        if not np.any(mask):
            continue
        pooled[i] = expr[mask].mean(axis=0)
        counts[i] = int(mask.sum())
    return pooled


def load_slide(sample_id: str, tissue: str,
               data_root: Path = config.DATA_ROOT,
               image_encoder: str = config.IMAGE_ENCODER) -> Slide:
    """Load one live HEST sample into the common Slide region-grid interface."""
    adata_path = _sample_hf_path(sample_id, 'st/{sample_id}.h5ad')
    adata = read_h5ad(adata_path)

    patch_path = _sample_hf_path(sample_id, 'patches/{sample_id}.h5')
    with h5py.File(patch_path, 'r') as src:
        patch_barcodes = [b.decode('utf-8') for b in src['barcode'][:].reshape(-1).tolist()]
        patch_coords = np.asarray(src['coords'][:], dtype=np.float32)
        patches = np.asarray(src['img'][:], dtype=np.uint8)

    obs_index = [str(x) for x in adata.obs.index.tolist()]
    obs_map = {barcode: idx for idx, barcode in enumerate(obs_index)}
    patch_match = [obs_map[b] for b in patch_barcodes if b in obs_map]
    if len(patch_match) != len(patch_barcodes):
        missing = set(patch_barcodes) - set(obs_map)
        raise RuntimeError(f"Patch barcode mismatch for sample {sample_id}: {len(missing)} missing from AnnData; first={sorted(list(missing))[:10]}")

    expr_raw = adata.X
    if hasattr(expr_raw, "toarray"):
        expr = expr_raw.toarray().astype(np.float32)
    else:
        expr = np.asarray(expr_raw, dtype=np.float32)

    if 'spatial' in adata.obsm:
        coords_px = np.asarray(adata.obsm['spatial'], dtype=np.float32)
    else:
        coords_px = np.asarray(adata.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy(), dtype=np.float32)
    coords_um = _normalize_spatial_coords(coords_px)

    expr = expr[patch_match]
    coords_um = coords_um[patch_match]

    if 'in_tissue' in adata.obs.columns:
        keep = adata.obs['in_tissue'].to_numpy(dtype=bool)[patch_match]
        expr = expr[keep]
        coords_um = coords_um[keep]
        patches = patches[keep]

    region_id, region_centroids = build_region_grid(coords_um, region_um=config.REGION_UM)
    expr_region = _pooled_expression_by_region(expr, region_id, n_regions=region_centroids.shape[0])

    expr_region = expr_region.astype(np.float32)
    gene_mask = np.mean(expr_region > 0, axis=0) >= config.MIN_SPOT_FRACTION
    expr_filtered = expr_region[:, gene_mask]
    expr_filtered = np.log1p(expr_filtered)

    if expr_filtered.shape[1] == 0:
        raise RuntimeError(f"No genes survive the HEST expression filter for sample {sample_id}")

    if expr_filtered.shape[1] > config.N_HVG:
        gene_var = expr_filtered.var(axis=0)
        hvg_idx = np.argsort(gene_var)[::-1][: config.N_HVG]
        hvg_idx = np.asarray(hvg_idx, dtype=int)
    else:
        hvg_idx = np.arange(expr_filtered.shape[1], dtype=int)

    # HEST patch tiles are per-spot images; average their UNI embeddings by region.
    region_patches = []
    for region_idx in range(region_centroids.shape[0]):
        mask = region_id == region_idx
        if not np.any(mask):
            continue
        region_patches.append(patches[mask])

    region_embeddings = []
    for group in region_patches:
        if group.size == 0:
            region_embeddings.append(np.zeros((config.IMAGE_EMB_DIM,), dtype=np.float32))
            continue
        emb = embed_patches(group, batch_size=config.BATCH_SIZE_EMB)
        region_embeddings.append(emb.mean(axis=0))

    image_embeddings = np.stack(region_embeddings, axis=0).astype(np.float32)
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True).clip(min=1e-8)

    return Slide(
        tissue=tissue,
        region_coords=region_centroids,
        image_embeddings=image_embeddings,
        expression=expr_filtered,
        hvg_idx=hvg_idx,
        gene_names=list(adata.var_names[gene_mask]),
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
