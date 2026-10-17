"""
Build the arXiv source package for the paper.

Collects exactly the files manuscript.tex needs (the .tex, its .bbl, the style file, every \\input table, every
\\includegraphics figure) into paper/arxiv_submission.tar.gz, then extracts it into a fresh temporary folder and
compiles it there (tectonic, if found on PATH or via --tectonic) to confirm the package is self-contained.

arXiv does not run BibTeX, so paper/manuscript.bbl must be current: compile the paper once (e.g. `tectonic
--keep-intermediates manuscript.tex`) before running this script. The script refuses to package a .bbl older than
references.bib or manuscript.tex.

Run:  python scripts/24_arxiv_package.py [--tectonic PATH]
"""
import argparse, os, re, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"


def needed_files(tex: str):
    inputs = re.findall(r"\\input\{([^}]+)\}", tex)
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    out = ["manuscript.tex", "manuscript.bbl", "neurips_2023.sty"]
    out += [p if p.endswith(".tex") else p + ".tex" for p in inputs]
    out += figs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tectonic", default=shutil.which("tectonic"))
    args = ap.parse_args()
    tex = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    files = needed_files(tex)
    missing = [f for f in files if not (PAPER / f).exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    bbl = (PAPER / "manuscript.bbl").stat().st_mtime
    for src in ("references.bib", "manuscript.tex"):
        if (PAPER / src).stat().st_mtime > bbl:
            sys.exit(f"manuscript.bbl is older than {src}; recompile the paper first")
    out = PAPER / "arxiv_submission.tar.gz"
    with tarfile.open(out, "w:gz") as tf:
        for f in files:
            tf.add(PAPER / f, arcname=f)
    size = out.stat().st_size / 1024
    print(f"[write] {out} ({len(files)} files, {size:.0f} KiB)")
    for f in files:
        print("   ", f)
    if not args.tectonic:
        print("tectonic not found: skipped the clean-room compile")
        return
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(out) as tf:
            tf.extractall(d, filter="data")
        r = subprocess.run([args.tectonic, "manuscript.tex"], cwd=d, capture_output=True, text=True)
        log = r.stdout + r.stderr
        bad = [l for l in log.splitlines() if re.search(r"error|undefined|Overfull", l, re.I) and "Fontconfig" not in l]
        ok = r.returncode == 0 and (Path(d) / "manuscript.pdf").exists()
        # no references.bib is shipped (arXiv does not run BibTeX), so the packaged .bbl must survive and be used
        n_items = (Path(d) / "manuscript.bbl").read_text(encoding="utf-8").count("\\bibitem")
        n_src = (PAPER / "manuscript.bbl").read_text(encoding="utf-8").count("\\bibitem")
        bad = [l for l in bad if "BibTeX" not in l]
        ok = ok and n_items == n_src and n_src > 0
        print(f"clean-room compile: {'OK' if ok else 'FAILED'} ({n_items}/{n_src} bibliography entries)")
        for l in bad:
            print("   ", l)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
