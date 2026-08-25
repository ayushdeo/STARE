"""
Stage-0 step 1 (RUN ON 4070): pick the exact HEST-1k Visium HD sample IDs per tissue and
write them back into src/stare/config.py HEST_SAMPLE_IDS.

We could not verify exact sample IDs without HEST metadata, so this must run on the machine
that has the HEST download / HuggingFace access.

TODO(4070):
  1. pip install hest datasets  (see requirements.txt)
  2. read HEST metadata (HEST_v1_1_0.csv or `datasets` metadata), filter:
       st_technology == 'Visium HD'  AND  organ/oncotree matches tissue
     for tissue in config.SCALE_TISSUES (start with FEASIBILITY_TISSUE='prostate').
  3. print candidate sample IDs + spot counts; choose 1-2 per tissue for feasibility.
  4. paste them into config.HEST_SAMPLE_IDS and commit.
FALLBACK: if Visium HD samples are too few for a tissue, set config.PLATFORM='visium'
  and pick standard-Visium prostate/kidney samples (log the switch in experiments/deviations.md).
"""
from stare import config


def main():
    print("Feasibility tissue:", config.FEASIBILITY_TISSUE)
    print("Platform (pinned):", config.PLATFORM, "| fallback:", config.PLATFORM_FALLBACK)
    print("Region unit:", config.REGION_UM, "um  ==", config.PATCH_PX, "px @", config.TARGET_MAG, "x")
    print("\nTODO: implement HEST metadata filtering here (see module docstring).")
    print("Then fill config.HEST_SAMPLE_IDS and re-run scripts/01_build_slides.py")


if __name__ == "__main__":
    main()
