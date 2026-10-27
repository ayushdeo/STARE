"""
Build the anonymized supplementary material for the TMLR submission (double-blind).

Writes paper/tmlr_supplementary.zip containing every tracked file of the repository except paper/, with:
  * author identity removed (LICENSE copyright line, README author line),
  * the machine hostname removed from TensorBoard log file names (their contents carry none),
  * commit hashes in the Markdown notes replaced by <commit>, so the public repository cannot be found from them,
  * REGISTRATION_LOG.md: for each registered stage, the commit date of the analysis plan and of the first committed
    results it governs (dates only), taken from the git history,
  * SUPPLEMENT_README.md: what the archive contains and how to regenerate the paper's tables and figures.
The script refuses to finish if any identifying string survives in the archive. TMLR limit: 100 MB.

Run from the repository root:  python scripts/25_tmlr_supplement.py
"""
import io, re, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tmlr_supplementary.zip"
TEXT = {".py", ".md", ".txt", ".toml", ".json", ".cfg", ".yml", ".yaml", ".csv", ".tex", ""}
FORBIDDEN = [r"ayush", r"\bdeo\b", r"abhijeet", r"southern california", r"\busc\b", r"laptop-7bvkv3gm",
             r"github\.com/ayushdeo", r"gmail"]
HASH = re.compile(r"(?<![0-9A-Za-z])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}(?![0-9A-Za-z])")

# (stage, registration heading in docs/ITERATION_LOG.md, result files the plan governs)
STAGES = [("Stage 1", "## Iteration 7", ["results/ci_multislide_graphflow.json", "results/realdata_detection.json"]),
          ("Stage 2", "## Iteration 9", ["results/revision_runs.json"]),
          ("Stage 3", "## Iteration 11", ["results/stage3_runs.json"])]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def first_commit_date(*args, after=None):
    """Author date of the first commit matching args; with `after` (a unix time), the first one strictly later."""
    out = git("log", "--reverse", "--format=%at %ad", "--date=iso-strict", *args)
    for line in out.splitlines():
        t, d = line.split(" ", 1)
        if after is None or int(t) > after:
            return d, int(t)
    return "not found", None


def registration_log():
    rows = ["# Registration log", "",
            "Commit dates from the project's version history (author dates, ISO 8601). For each stage: when its",
            "analysis plan was first committed, and when each file holding that stage's results was first committed",
            "after the plan (stage-1 files existed earlier with results from a 7-slide pilot, which the plan",
            "superseded). Commit identifiers are withheld for anonymity and will be published with the repository on",
            "acceptance.", "",
            "| Stage | Plan (docs/ITERATION_LOG.md) | Plan committed | Results file | Results committed after plan |",
            "|---|---|---|---|---|"]
    for stage, heading, results in STAGES:
        plan, t_plan = first_commit_date("-S", heading, "--", "docs/ITERATION_LOG.md")
        for r in results:
            res, t_res = first_commit_date("--", r, after=t_plan)
            assert t_res is not None and t_res > t_plan, (stage, r)
            rows.append(f"| {stage} | {heading.lstrip('# ')} | {plan} | `{r}` | {res} |")
    rows += ["", "The plans themselves are in `docs/ITERATION_LOG.md` (iterations 7, 9, and 11).", ""]
    return "\n".join(rows)


README = """# Supplementary material (anonymized)

Code, configuration, results, and registered analysis plans for the submission
"Where to Measure Next? Evaluating Feedback-Driven Region Selection for Spatial Transcriptomics".

- `src/stare/`: simulator, data loading, imputers, policies, metrics, statistics.
- `scripts/`: numbered experiment, analysis, and table/figure scripts (see `README.md` for the stage mapping).
- `results/*.json`: every result reported in the paper; tables and figures regenerate from these alone.
- `docs/ITERATION_LOG.md`: dated record of every iteration, including the three registered analysis plans
  (iterations 7, 9, 11). `REGISTRATION_LOG.md` lists when each plan and its results were committed.
- `artifacts/sofish_seed222_5m/`: weights and TensorBoard logs of the native SOFisher run (MIT-licensed code; see
  `THIRD_PARTY_NOTICES.md`).

Regenerate the paper's tables and figures without any data download:

    pip install -e .
    python -u scripts/14_power_summary.py
    python -u scripts/15_paper_assets.py
    python -u scripts/18_revision_analysis.py
    python -u scripts/20_stage2_sensitivity.py
    python -u scripts/19_paper_assets_round2.py
    python -u scripts/22_stage3_analysis.py
    python -u scripts/23_paper_assets_stage3.py

Outputs are written to `paper/generated/` and `paper/*.pdf`. Rerunning the experiments needs the gated HEST-1k
dataset and UNI2-h weights from Hugging Face (see `README.md`).
"""


def scrub_text(path: str, text: str) -> str:
    if path == "LICENSE":
        text = re.sub(r"Copyright \(c\) 2026 .*", "Copyright (c) 2026 Anonymous authors", text)
    if path == "README.md":
        text = re.sub(r"(?m)^> .*University of Southern California\.\s*$", "> Anonymous authors (under double-blind review).", text)
    if path.endswith(".md"):
        text = HASH.sub("<commit>", text)
    return text


def main():
    files = [f for f in git("ls-files").splitlines() if not f.startswith("paper/")]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            data = (ROOT / f).read_bytes()
            name = re.sub(r"\.LAPTOP-[A-Z0-9]+\.", ".host.", f)
            suffix = Path(f).suffix.lower()
            if suffix in TEXT or Path(f).name in ("LICENSE", ".gitignore"):
                try:
                    data = scrub_text(f, data.decode("utf-8")).encode("utf-8")
                except UnicodeDecodeError:
                    pass
            z.writestr("supplementary/" + name, data)
        z.writestr("supplementary/REGISTRATION_LOG.md", registration_log())
        z.writestr("supplementary/SUPPLEMENT_README.md", README)
    # verify anonymity of names and contents
    leaks = []
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as z:
        for info in z.infolist():
            raw = z.read(info)
            try:                                    # text: full pattern scan of name and contents
                blob = info.filename.lower() + "\n" + raw.decode("utf-8").lower()
                pats = FORBIDDEN
            except UnicodeDecodeError:              # binary: short patterns match random bytes, so scan the
                blob = info.filename.lower() + "\n" + raw.lower().decode("latin-1")    # name and long strings only
                pats = [r"ayush", r"abhijeet", r"laptop-7bvkv3gm", r"southern california", r"gmail\.com"]
            for pat in pats:
                if re.search(pat, blob):
                    leaks.append((info.filename, pat))
    if leaks:
        for f, p in leaks[:20]:
            print("LEAK", f, p)
        sys.exit("identifying strings remain; nothing written")
    OUT.write_bytes(buf.getvalue())
    print(f"[write] {OUT} ({len(files) + 2} files, {OUT.stat().st_size / 1e6:.1f} MB; TMLR limit 100 MB)")
    print(registration_log())


if __name__ == "__main__":
    main()
