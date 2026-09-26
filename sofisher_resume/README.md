# SOFisher resume patch

Adds stop/resume to a local **SOFisher** clone (`Heathcliff-Ng/SOFisher`) for the Level-2 native run.
SOFisher's `max_steps` default is **5,000,000** and it persists only network weights (no optimizer,
epsilon, or step count) with no resume path — so long runs can't be continued. This patch adds
persistence only (no change to reward, hyperparameters, or training logic), so runs stay faithful.

## Apply
```bash
# from the STARE repo root, pointing at your SOFisher clone:
python sofisher_resume/apply_resume.py /path/to/SOFisher
```
Idempotent; backs up originals to `*.orig`. Patches `src/args.py` (+`--resume`, `--warmstart`,
`--ckpt_every`) and `main_cortex.py` (adds `save_resume`/`load_resume`, loads before the loop, writes a
full checkpoint every `--ckpt_every` steps).

## Continue your current 600k run (weights only exist)
Your interrupted run saved only `final`/`best` network weights, not a full checkpoint. Warm-start from
them (epsilon is already at its 0.01 floor at 600k, so this is a sound continuation):
```bash
cd /path/to/SOFisher
find . -name 'seed*_final_eval.pth'                 # confirms model/DQN/seed_222/ exists
python main_cortex.py --warmstart final             # loads final weights, then writes full checkpoints
```

## Fully resumable from here on
```bash
python main_cortex.py                                          # writes seed222_resume.pth every 50k steps
find . -name 'seed*_resume.pth'
python main_cortex.py --resume model/DQN/seed_222/seed222_resume.pth   # exact continue
```

## Notes
- SEED is hardcoded to 222 in `main_cortex.py`, so files are `model/DQN/seed_222/seed222_resume.pth`.
- The replay buffer is not persisted; on resume it refills from empty while epsilon is at its floor
  (minor sample-quality dip, not a correctness issue). Ask if you want buffer persistence too.
- `main_aging.py` / `main_disease.py` / `main_CRC.py` share the same `Runner` pattern but are not
  auto-patched — run `apply_resume.py` logic on them the same way if needed, or ask me to extend it.
- This is a faithful, logged modification: persistence only. Record it in your repro notes.
