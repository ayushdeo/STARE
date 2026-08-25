"""
Stage-0 step 2 (RUN ON 4070): build cached `Slide` objects for the feasibility tissue.
Loads HEST samples -> region grid -> UNI embeddings + HVG + log1p expression -> cache.

TODO(4070): implement stare.data.load_slide, then this just loops config.HEST_SAMPLE_IDS.
"""
from stare import config, data


def main():
    tissue = config.FEASIBILITY_TISSUE
    slides = data.load_tissue(tissue)
    print(f"Built {len(slides)} slides for {tissue}")
    for s in slides:
        print(f"  {s.patient_id}: {s.n_regions} regions, {s.expression.shape[1]} genes, "
              f"{len(s.hvg_idx)} HVG")


if __name__ == "__main__":
    main()
