"""
Build the arXiv source package for the paper.

Collects exactly the files manuscript.tex needs (the .tex, its .bbl, the style file, every \\input table, every
\\includegraphics figure) into paper/arxiv_submission.tar.gz, then extracts it into a fresh temporary folder and
compiles it there (tectonic, if found on PATH or via --tectonic) to confirm the package is self-contained.

arXiv does not run BibTeX, so the packaged manuscript.tex reads paper/manuscript.bbl directly (\\input instead of
\\bibliography) and the .bbl must be current: compile the paper once (e.g. `tectonic
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
    out = ["manuscript.tex", "manuscript.bbl", "tmlr.sty", "fancyhdr.sty"]
    out += [p if p.endswith(".tex") else p + ".tex" for p in inputs]
    out += figs
    return out


def strip_comments(text: str):
    """Remove TeX comments. A whole-line comment is dropped; an inline one keeps its '%' so that a line-end '%'
    still suppresses the space it suppressed before. An escaped '\\%' is not a comment."""
    out, n = [], 0
    for line in text.splitlines():
        m = re.search(r"(?<!\\)%", line)
        if m is None:
            out.append(line)
            continue
        n += 1
        if line[:m.start()].strip() == "":
            continue
        out.append(line[:m.start() + 1])
    return "\n".join(out) + "\n", n


REVIEW_ONLY = ("Under review as submission to TMLR", "Anonymous authors", "Paper under double-blind review")


def body_words(text: str, tex: str):
    """Words of the typeset paper without the lines that differ between review and preprint builds: the review
    header, the anonymous-author block, and the author block of the preprint."""
    m = re.search(r"\\author\{(.*?)\}\s*\n\s*\n", tex, re.S) or re.search(r"\\author\{(.*?)\n\\def", tex, re.S)
    author = re.sub(r"\\[a-zA-Z]+|[{}\\%]", " ", m.group(1)).split() if m else []
    lines = [l for l in text.splitlines() if l.strip() not in REVIEW_ONLY]
    words = " ".join(lines).split()
    return [w for w in words if w not in set(author)]


def pdf_text(path: Path) -> str:
    try:
        import pymupdf
    except ImportError:
        return ""
    with pymupdf.open(path) as d:
        return "\n".join(p.get_text() for p in d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tectonic", default=shutil.which("tectonic"))
    args = ap.parse_args()
    tex = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    if "CONTACT-EMAIL" in tex:              # the preprint shows the author block, so the placeholder would be published
        sys.exit("replace CONTACT-EMAIL in manuscript.tex's \\author block before building a preprint")
    files = needed_files(tex)
    missing = [f for f in files if not (PAPER / f).exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    bbl = (PAPER / "manuscript.bbl").stat().st_mtime
    for src in ("references.bib", "manuscript.tex"):
        if (PAPER / src).stat().st_mtime > bbl:
            sys.exit(f"manuscript.bbl is older than {src}; recompile the paper first")
    out = PAPER / "arxiv_submission.tar.gz"
    folder = PAPER / "arxiv_upload"                 # the same files unpacked, for a folder-based upload
    if folder.exists():
        shutil.rmtree(folder)
    removed = []
    with tarfile.open(out, "w:gz") as tf:
        for f in files:
            dst = folder / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            if f.endswith(".tex"):                  # arXiv archives the source: drop the author's own TeX comments
                text, n = strip_comments((PAPER / f).read_text(encoding="utf-8"))
                if f == "manuscript.tex":           # read the shipped .bbl directly, so no BibTeX run is ever needed
                    bib = "\\bibliography{references}"
                    assert text.count(bib) == 1, "expected one \\bibliography command"
                    text = text.replace(bib, "\\input{manuscript.bbl}")
                    # a preprint carries the author names and no TMLR review header
                    text = text.replace("\\usepackage{tmlr}", "\\usepackage[preprint]{tmlr}")
                dst.write_text(text, encoding="utf-8")
                if n:
                    removed.append((f, n))
            else:                                   # style file, .bbl, and figures are shipped unchanged
                shutil.copy2(PAPER / f, dst)
            tf.add(dst, arcname=f)
    size = out.stat().st_size / 1024
    print(f"[write] {out} ({len(files)} files, {size:.0f} KiB); unpacked copy in {folder}")
    for f in files:
        print("   ", f)
    for f, n in removed:
        print(f"    stripped {n} comment(s) from {f}")
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
        # stripping comments must not change the typeset paper
        a, b = pdf_text(Path(d) / "manuscript.pdf"), pdf_text(PAPER / "manuscript.pdf")
        if a and b:
            same = body_words(a, tex) == body_words(b, tex)
            ok = ok and same
            print(f"body text identical to paper/manuscript.pdf (author/review lines aside): {same}")
        for l in bad:
            print("   ", l)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
