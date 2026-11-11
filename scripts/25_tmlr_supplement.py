"""
Build the anonymized supplementary material for the TMLR submission.

The supplement is a curated reproduction package, not a copy of the repository. It contains, from an explicit
allow-list, only what the paper needs: the package code, the scripts that produce the paper's results, tables,
and figures, the result files they read, the native SOFisher training logs, and verbatim excerpts of the three
registered analysis plans. Files, policies, and result keys are renamed to the terms used in the paper, module and
script docstrings are replaced by short descriptions, and development comments are removed. Nothing else from the
repository (working notes, earlier development scripts, tests, model weights) is included.

Checks, all of which must pass before the archive is written:
  * every included Python file compiles, and the package's policies run on a synthetic slide;
  * renaming creates no duplicate keys in any result file;
  * no identifying string (name, affiliation, host, repository) and no development term remains;
  * the paper's 20 tables, regenerated from the extracted archive, are byte-identical to paper/generated/.

Run from the repository root:  python scripts/25_tmlr_supplement.py
"""
import ast, io, json, re, shutil, subprocess, sys, tempfile, tokenize, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tmlr_supplementary.zip"
TOP = "supplementary/"

# ------------------------------------------------------------------ allow-list: repository path -> supplement path
SCRIPTS = {
    "00_select_samples.py": "select_slides.py",
    "00b_scale_audit.py": "unit_scale_audit.py",
    "05_multislide_ci.py": "stage1_reconstruction.py",
    "09_realdata_detection.py": "stage1_detection.py",
    "10_competitors.py": "stage1_competitors.py",
    "12_hvg_sensitivity.py": "stage1_gene_sets.py",
    "13_batched_contiguous.py": "stage1_batched.py",
    "14_power_summary.py": "stage1_analysis.py",
    "15_paper_assets.py": "tables_stage1.py",
    "07_positive_control.py": "control_synthetic_reconstruction.py",
    "08_detection_control.py": "control_synthetic_detection.py",
    "11_sofisher_curve.py": "sofisher_native_curve.py",
    "17_revision_experiments.py": "stage2_experiments.py",
    "18_revision_analysis.py": "stage2_analysis.py",
    "20_stage2_sensitivity.py": "stage2_comparator_inference.py",
    "19_paper_assets_round2.py": "tables_stage2.py",
    "21_stage3_experiments.py": "stage3_experiments.py",
    "22_stage3_analysis.py": "stage3_analysis.py",
    "23_paper_assets_stage3.py": "tables_stage3.py",
    "26_posthoc_robustness.py": "posthoc_robustness.py",
}
MODULES = {
    "__init__.py": "__init__.py", "config.py": "config.py", "data.py": "data.py", "encoders.py": "encoders.py",
    "hest_meta.py": "hest_meta.py", "imputer.py": "imputer.py", "metrics.py": "metrics.py",
    "simulator.py": "simulator.py", "slides.py": "slides.py", "stats.py": "stats.py",
    "policies/__init__.py": "policies/__init__.py", "policies/base.py": "policies/base.py",
    "policies/naive.py": "policies/naive.py", "policies/oneshot.py": "policies/oneshot.py",
    "policies/placement.py": "policies/placement.py", "policies/greedy_ig.py": "policies/error_reduction.py",
    "policies/residual_guided.py": "policies/residual.py", "policies/value_greedy.py": "policies/local_variance.py",
    "policies/feedback.py": "policies/feedback.py", "policies/hybrid.py": "policies/hybrid.py",
    "policies/competitors.py": "policies/competitors.py",
}
RESULTS = {
    "ci_multislide_graphflow.json": "stage1_reconstruction_graphflow.json",
    "realdata_detection.json": "stage1_detection.json",
    "competitors.json": "stage1_competitors.json",
    "hvg_sensitivity.json": "stage1_gene_sets.json",
    "batched_contiguous.json": "stage1_batched.json",
    "power_summary.json": "stage1_summary.json",
    "positive_control.json": "control_synthetic_reconstruction.json",
    "detection_control.json": "control_synthetic_detection.json",
    "sofisher_native.json": "sofisher_native.json",
    "sample_candidates.json": "slide_selection.json",
    "scale_audit.json": "unit_scale_audit.json",
    "revision_runs.json": "stage2_runs.json",
    "revision_summary.json": "stage2_summary.json",
    "stage2_sensitivity.json": "stage2_comparator_inference.json",
    "stage3_runs.json": "stage3_runs.json",
    "stage3_summary.json": "stage3_summary.json",
    "posthoc_robustness.json": "posthoc_robustness.json",
}
LEGACY = ["batched_contiguous.json", "ci_multislide_graphflow.json", "ci_multislide_knn.json",
          "ci_multislide_kriging.json", "competitors.json", "hvg_sensitivity.json", "realdata_detection.json"]

# ------------------------------------------------------------------ renaming (applied to .py and .json, in order)
# file names and output names
NAME_MAP = [(o, n) for o, n in SCRIPTS.items()] + [(o, n) for o, n in RESULTS.items() if o != n] + [
    ("ci_multislide_", "stage1_reconstruction_"), ("n7_legacy", "earlier_slide_set"),
    ("facts_r2", "facts_stage2"), ("facts_s3", "facts_stage3"), ('"facts.json"', '"facts_stage1.json"'),
    ("'facts.json'", "'facts_stage1.json'"), ("fig_r2", "fig_stage2"), ("fig_s3", "fig_stage3"),
    ('"r2_', '"stage2_'), ("'r2_", "'stage2_"), ('"s3_', '"stage3_'), ("'s3_", "'stage3_"),
    ("figure2_positive_control", "control_synthetic_reconstruction"),
    ("figure3_detection_control", "control_synthetic_detection"),
    ("figure4_sofisher_native", "sofisher_native_curve"),
    ('ROOT / "docs" / "figures"', 'ROOT / "figures"'), ('"docs/figures/', '"figures/'),
    ('Path("docs") / "POWER_SUMMARY.md"', 'Path("results") / "stage1_summary.md"'),
    ("artifacts/sofish_seed222_5m", "sofisher_native_run"),
]
# identifiers and result keys -> paper terms (whole tokens; '_' counts as a boundary so compound keys follow)
TOKEN_MAP = [
    ("ours_closed_loop", "error_reduction"), ("ours_vs_best_nonadapt", "error_reduction_vs_best_nonadaptive"),
    ("ours_vs_mi", "error_reduction_vs_mi_greedy"), ("closed_vs_ablate", "error_reduction_vs_no_measurement_ablation"),
    ("ablate_no_measured", "no_measurement_ablation"), ("oneshot_spatialfinder", "oneshot"),
    ("value_greedy", "local_variance"), ("residual_guided", "residual"), ("greedy_ig", "error_reduction"),
    ("adaptive_uncertainty", "max_uncertainty"), ("scr2st", "diversity"),
    ("GreedyInfoGainPolicy", "ErrorReductionPolicy"), ("ValueGreedyPolicy", "LocalVariancePolicy"),
    ("ResidualGuidedPolicy", "ResidualPolicy"), ("SCR2STDiversityPolicy", "DiversityPolicy"),
    ("SOFisherROIPolicy", "ROISeekingPolicy"), ("OneShotSpatialFinderPolicy", "OneShotPolicy"),
    ("LOSO", "feasible"), ("loso", "feasible"), ("ours", "err_red"),
]
SOFISHER_TOKEN = (re.compile(r"(?<![A-Za-z0-9])sofisher(?!_native)(?![A-Za-z0-9])"), "roi_seeking")
QLABEL = (re.compile(r"(?<![A-Za-z0-9_])Q([123])(?![A-Za-z0-9])"), r"RQ\1")


def tok(word):
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])")


TOKEN_RES = [(tok(o), n) for o, n in TOKEN_MAP]
# exact strings that carried drafting-process wording into code or result values
STRING_MAP = [
    ("pre-registered round-2 analysis (ITERATION_LOG iteration 9)", "registered stage-2 analysis (registered_plans/stage2.md)"),
    ("stage-3 registered analysis (ITERATION_LOG iteration 11)", "registered stage-3 analysis (registered_plans/stage3.md)"),
    ("post hoc sensitivity of stage-2 inference (revision round 3, point 2)", "post hoc check of stage-2 comparator inference"),
    ("Run scripts/00_pick_samples.py on the 4070 ", "Run scripts/select_slides.py --apply "),
    ("wrapper is a further upgrade (WS-B scale step)", "imputer is not part of this package"),
    ("NOT IN METADATA SOURCE (verify on the 4070 with the gated v1.3 CSV)", "NOT IN METADATA SOURCE (check the gated v1.3 metadata)"),
]

# ------------------------------------------------------------------ docstrings for every included Python file
DOC = {
    "src/stare/__init__.py": "Simulator, imputers, acquisition policies, metrics, and study-level statistics for the evaluation.",
    "src/stare/config.py": "Fixed configuration: slide set, acquisition unit, budgets, imputer, policy and statistics settings, seeds.\n\nEvery script reads these values; none was tuned on the registered results.",
    "src/stare/data.py": "Loads a HEST-1k sample as a Slide: native spots as regions, coordinates in microns, UNI2-h patch\nembeddings, log1p expression with genes detected in fewer than 10% of regions removed, top-50 variable genes.",
    "src/stare/encoders.py": "Frozen UNI2-h image encoder used to embed region patches.",
    "src/stare/hest_meta.py": "HEST-1k metadata helpers: pixel size, species, and specimen and study identity, used for slide selection and\nfor grouping slides by study.",
    "src/stare/imputer.py": "Imputers.\n\nGraphFlowImputer is the paper's imputer: ridge regression on random Fourier features of image principal\ncomponents, followed by label propagation over a fixed image-and-space k-nearest-neighbour graph that holds\nprofiled regions at their measured values. Its uncertainty, one minus the graph influence reaching a region from\nprofiled nodes, depends only on which regions are profiled. KNNImputer and LearnedKrigingImputer are the\ncomparison imputers of the appendix.",
    "src/stare/metrics.py": "Metrics on unprofiled regions (per-gene Pearson correlation over the top-50 genes) and for detection (local\nmolecular heterogeneity and ROI recall).",
    "src/stare/simulator.py": "Retrospective acquisition simulator: a policy reveals regions up to a budget, one at a time or in rounds,\nand the imputer predicts the unprofiled regions from the revealed ones.",
    "src/stare/slides.py": "Loads the 22-slide set with a per-slide cache and returns the tissue and study grouping used for inference.",
    "src/stare/stats.py": "Study-level inference: bootstrap intervals that resample whole studies, exact sign-flip tests over study\nsign patterns, Holm adjustment, and paired margin summaries.",
    "src/stare/policies/__init__.py": "Acquisition policies and their registry. select(state) returns the next region to profile.\n\nNon-adaptive: random, grid, kmeans, oneshot, var_greedy, mi_greedy. Sequential fixed designs: error_reduction\n(and its maximum-uncertainty variant). Value-conditioned: residual, local_variance. Competitor objectives:\ndiversity, roi_seeking. Wrappers: FeedbackAblation, HybridPolicy.",
    "src/stare/policies/base.py": "Policy interface. closed_loop marks policies whose selection reads measured expression.",
    "src/stare/policies/naive.py": "Random, grid (farthest-point in space), and k-means (region nearest each centroid of the image embeddings).",
    "src/stare/policies/oneshot.py": "One-shot ranker after SpatialFinder: scores regions once from image embeddings by rare-cluster membership\nand distance to the cluster centroid, a simplification of SpatialFinder's diversity score without its cell\nclassifier or pathologist labelling.",
    "src/stare/policies/placement.py": "Variance-greedy and mutual-information-greedy placement with an RBF kernel on image and coordinate features\n(Krause et al., 2008). Non-adaptive: selections use only the kernel, never measured values.",
    "src/stare/policies/error_reduction.py": "Error-reduction policy: picks the region whose 15 nearest unprofiled neighbours carry the most imputer\nuncertainty; with criterion 'uncertainty' it picks the most uncertain region (maximum-uncertainty). With the\npaper's imputer the uncertainty does not depend on measured values, so both are sequential fixed designs.",
    "src/stare/policies/residual.py": "Residual policy: the error-reduction form with the uncertainty replaced by an estimate of the imputer's error,\nthe leave-one-out residual of the image ridge model averaged over each candidate's 10 nearest profiled regions,\ntimes one minus the graph influence.",
    "src/stare/policies/local_variance.py": "Local-variance policy: scores a candidate by the kernel-weighted variance of measured expression around it\nplus beta times its distance to the nearest profiled region (beta = 0.5).",
    "src/stare/policies/feedback.py": "Feedback ablations: run a policy unchanged while it sees permuted values (another region's expression) or\nconstant values (the slide mean). The imputer and the scores always use the true values.",
    "src/stare/policies/hybrid.py": "Coverage-plus-feedback hybrid: the first ceil(f x budget) regions come from a fixed coverage design and the\nrest from a feedback policy. The coverage phase equals the prefix of the design's own selection.",
    "src/stare/policies/competitors.py": "Competitor objectives, reimplemented (not the authors' trained models): a diversity sampler (farthest-point\nselection in joint image and spatial features) approximating the coverage and diversity terms of the SCR2-ST\nreward, and an ROI-seeking sampler following the SOFisher reward.",
    "scripts/select_slides.py": "Selects the 22 slides from HEST-1k metadata under the criteria in Section 3.2 and writes\nresults/slide_selection.json.  Run: python -u scripts/select_slides.py [--apply]",
    "scripts/unit_scale_audit.py": "Compares the acquisition unit (one native spot per region) with the square-grid unit of the earlier slide set\n(Appendix A); writes results/unit_scale_audit.json.  Run: python -u scripts/unit_scale_audit.py",
    "scripts/stage1_reconstruction.py": "Stage 1: reconstruction accuracy on the 22 slides for the non-adaptive and sequential fixed designs; writes\nresults/stage1_reconstruction_<imputer>.json.  Run: python -u scripts/stage1_reconstruction.py [--imputers graphflow]",
    "scripts/stage1_detection.py": "Stage 1: local-variance policy against coverage designs for reconstruction and ROI detection; writes\nresults/stage1_detection.json.  Run: python -u scripts/stage1_detection.py",
    "scripts/stage1_competitors.py": "Stage 1: competitor objectives (diversity and ROI-seeking samplers) and the ground-truth ordering for\ndetection; writes results/stage1_competitors.json.  Run: python -u scripts/stage1_competitors.py",
    "scripts/stage1_gene_sets.py": "Stage 1: sensitivity of the reconstruction margins to the evaluation gene set (Appendix J); writes\nresults/stage1_gene_sets.json.  Run: python -u scripts/stage1_gene_sets.py",
    "scripts/stage1_batched.py": "Stage 1: batched (rounds of 1, 5, 25, or one round) and contiguous-block acquisition at 10% budget; writes\nresults/stage1_batched.json.  Run: python -u scripts/stage1_batched.py",
    "scripts/stage1_analysis.py": "Stage-1 registered analysis: study-level intervals, sign-flip tests, and Holm over the 24 primary margins;\nwrites results/stage1_summary.json.  Run: python -u scripts/stage1_analysis.py",
    "scripts/tables_stage1.py": "Writes the stage-1 tables, Figure 1, the slide table, and the appendix tables (paper/generated/,\npaper/*.pdf) from the result files.  Run: python -u scripts/tables_stage1.py",
    "scripts/control_synthetic_reconstruction.py": "Synthetic positive control for reconstruction (Figure 1a): correlation length of hidden structure versus the\nmargin over coverage; writes results/control_synthetic_reconstruction.json.",
    "scripts/control_synthetic_detection.py": "Synthetic positive control for detection (Figure 1b): ROI amplitude versus the recall margin over coverage;\nwrites results/control_synthetic_detection.json.",
    "scripts/sofisher_native_curve.py": "Reads the TensorBoard logs of the native SOFisher run (sofisher_native_run/) and writes the evaluation-reward\ncurve (results/sofisher_native.json, Appendix L).  Run: python -u scripts/sofisher_native_curve.py",
    "scripts/stage2_experiments.py": "Stage-2 experiments (registered_plans/stage2.md). Part A: all policies, feedback ablations, and the diversity\nsampler on the 22 slides at six budgets. Part B: semi-synthetic positive control with injected hotspot genes.\nWrites results/stage2_runs.json, checkpointing every slide.  Run: python -u scripts/stage2_experiments.py --part A|B",
    "scripts/stage2_analysis.py": "Stage-2 registered analysis: feasible design chosen per held-out study, the 18 primary and the secondary\nmargins with Holm adjustment, and minimum detectable effects; writes results/stage2_summary.json.",
    "scripts/stage2_comparator_inference.py": "Post hoc checks of stage-2 inference (Appendix F): a bootstrap that repeats the feasible selection inside each\nresample, and a fixed MI-greedy comparator; writes results/stage2_comparator_inference.json.",
    "scripts/tables_stage2.py": "Writes the stage-2 tables and Figure 2 from the result files.  Run: python -u scripts/tables_stage2.py",
    "scripts/stage3_experiments.py": "Stage-3 experiments (registered_plans/stage3.md). Part H: coverage-plus-feedback hybrids with permuted controls.\nPart S: further permutation seeds for the stage-2 ablations. Part R: rounds of 5 regions with permuted controls.\nWrites results/stage3_runs.json.  Run: python -u scripts/stage3_experiments.py --part H|S|R",
    "scripts/stage3_analysis.py": "Stage-3 registered analysis: the 20 primary and 24 secondary margins with Holm adjustment, minimum detectable\neffects, and permutation-seed variation; writes results/stage3_summary.json.",
    "scripts/tables_stage3.py": "Writes the stage-3 tables, Figure 3, and the comparator-inference and seed tables from the result files.\nRun: python -u scripts/tables_stage3.py",
    "scripts/posthoc_robustness.py": "Post hoc robustness checks on existing results (Appendix M): recall above the expectation of random\nselection, intervals from inverting the exact study-level sign-flip test, and the primary analyses without the\nslide used to set the imputer's hyperparameters.  Run: python -u scripts/posthoc_robustness.py",
    "sofisher_resume_patch/apply_resume.py": "Adds checkpoint and resume support to a clone of the SOFisher repository, the only change made for the native\nrun (Appendix L). Training logic, reward, and hyperparameters are unchanged.\nRun: python sofisher_resume_patch/apply_resume.py /path/to/SOFisher",
}

# comments containing any of these are development notes and are removed
NOTE = re.compile(r"(?i)\b(ours|our (contribution|method|policy|closed)|source_of_truth|reproduction_plan|knowledge_base|"
                  r"checklist|deviations|ws-[a-e]|go/no-go|gate|revision|reviewer|round[ -]?[23]|4070|laptop|world[ -]?[12]|"
                  r"tier[ -]?[12]|must beat|kill|feasibility|iteration|confirm_on_machine|todo|stage[ -]?0|legacy|"
                  r"primary competitor|competitor|harmoniz|port\b|v[0-3]\b|honest|agentic|pitch|paper)\b")
FORBIDDEN_ID = [r"ayush", r"\bdeo\b", r"abhijeet", r"southern california", r"\busc\b", r"laptop-7bvkv3gm",
                r"github\.com/ayushdeo", r"gmail"]
FORBIDDEN_NOTE = [r"\breviewer", r"\brevision\b", r"round[ -]?2", r"round[ -]?3", r"\bours\b", r"source_of_truth",
                  r"knowledge_base", r"reproduction_plan", r"\bws-[a-e]\b", r"go/no-go", r"\b4070\b", r"iteration_log",
                  r"\btier[ -]?[12]\b", r"must beat", r"\bagentic\b"]
HASH = re.compile(r"(?<![0-9A-Za-z])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}(?![0-9A-Za-z])")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def rename(text, is_code):
    for o, n in NAME_MAP:
        text = text.replace(o, n)
    for o, n in STRING_MAP:
        text = text.replace(o, n)
    for rx, n in TOKEN_RES:
        text = rx.sub(n, text)
    text = SOFISHER_TOKEN[0].sub(SOFISHER_TOKEN[1], text)
    text = QLABEL[0].sub(QLABEL[1], text)
    return text


def clean_python(text, new_path):
    """Replace the module docstring, drop note-like comments and note-like function/class docstrings."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    drop, replace_at = set(), None
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        replace_at = (body[0].lineno - 1, body[0].end_lineno - 1)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str) and NOTE.search(first.value.value) and len(node.body) > 1:
                drop.update(range(first.lineno - 1, first.end_lineno))
    # all comments go: many were working notes, and removing only some leaves fragments of multi-line blocks
    comments = {}
    for t in tokenize.generate_tokens(io.StringIO(text).readline):
        if t.type == tokenize.COMMENT:
            comments[t.start[0] - 1] = t.start[1]
    out = []
    for i, line in enumerate(lines):
        if replace_at and i == replace_at[0]:
            out.append('"""\n' + DOC[new_path] + '\n"""\n')
            continue
        if (replace_at and replace_at[0] < i <= replace_at[1]) or i in drop:
            continue
        if i in comments:
            col = comments[i]
            if line[:col].strip() == "":
                continue
            line = line[:col].rstrip() + "\n"
        out.append(line)
    if replace_at is None:
        out.insert(0, '"""\n' + DOC[new_path] + '\n"""\n')
    text = "".join(out)
    text = re.sub(r"\n(?:[ \t]*\r?\n){3,}", "\n\n\n", text)       # no more than two blank lines in a row
    return text


def plan_excerpt(n, keep):
    """Verbatim lines of a registered plan section as first committed, with its commit date."""
    log = git("log", "--reverse", "--format=%h %at %ad", "--date=iso-strict", "-S", f"## Iteration {n} ", "--",
              "docs/ITERATION_LOG.md").split("\n")[0]
    commit, ts, date = log.split(" ", 2)
    section = re.search(rf"(?ms)^## Iteration {n} .*?(?=^## |\Z)", git("show", f"{commit}:docs/ITERATION_LOG.md")).group(0)
    lines = section.splitlines()
    picked = []
    for a, b in keep:
        picked += lines[a:b + 1] + [""]
    return date, int(ts), "\n".join(picked).rstrip() + "\n"


PLANS = {  # stage: (iteration, [(first, last) line ranges of the section to keep])
    "stage1": (7, [(23, 36)]),
    "stage2": (9, [(22, 29), (31, 44)]),
    "stage3": (11, [(20, 39)]),
}
PLAN_TITLES = {"stage1": "Stage 1: expanded 22-slide set", "stage2": "Stage 2: feasible design, residual policy, and feedback ablations",
               "stage3": "Stage 3: coverage-plus-feedback hybrids and batched acquisition"}


def plans():
    files, rows = {}, []
    for stage, (n, keep) in PLANS.items():
        date, ts, text = plan_excerpt(n, keep)
        files[f"registered_plans/{stage}.md"] = (f"# {PLAN_TITLES[stage]}: registered plan (excerpt)\n\nCommitted {date}. Verbatim "
                                                 "excerpt of the pre-specified experiments, comparison families, and decision rules; "
                                                 "motivation and interim notes in the same log entry are omitted. Names are those used "
                                                 "in the code at the time; see the name table in README.md.\n\n" + text)
        rows.append((stage, date, ts))
    return files, rows


def registration_log(plan_rows):
    governed = {"stage1": ["ci_multislide_graphflow.json", "realdata_detection.json"], "stage2": ["revision_runs.json"],
                "stage3": ["stage3_runs.json"]}
    out = ["# Registration log", "",
           "Commit dates from the project's version history (ISO 8601). Each plan was committed before the results it",
           "governs. Stage-1 result files existed earlier with results from a seven-slide pilot that the stage-1 plan",
           "superseded; the dates below are the first commits after the plan. Commit identifiers are withheld for",
           "anonymity.", "", "| Stage | Plan committed | Results file (this package) | Results committed |", "|---|---|---|---|"]
    for stage, date, t_plan in plan_rows:
        for f in governed[stage]:
            for line in git("log", "--reverse", "--format=%at %ad", "--date=iso-strict", "--", f"results/{f}").splitlines():
                t, d = line.split(" ", 1)
                if int(t) > t_plan:
                    out.append(f"| {stage.replace('stage', 'Stage ')} | {date} | `results/{RESULTS[f]}` | {d} |")
                    break
    return "\n".join(out) + "\n"


NAME_TABLE = """| Name in the paper | Name in this package | Name in the registered plans |
|---|---|---|
| local-variance policy | `local_variance` | `value_greedy` |
| residual policy | `residual` | `residual_guided` |
| error-reduction (sequential fixed design) | `error_reduction` | `greedy_ig`, `ours_*` keys |
| maximum-uncertainty | `max_uncertainty` | `adaptive_uncertainty` |
| diversity sampler (SCR2-ST objective) | `diversity` | `scr2st` |
| ROI-seeking sampler (SOFisher objective) | `roi_seeking` | `sofisher` |
| one-shot ranker | `oneshot` | `oneshot` |
| feasible design (leave-one-study-out) | `feasible` | `LOSO` |
| RQ1, RQ2, RQ4 | `RQ1`, `RQ2`, `RQ3` in keys | `Q1`, `Q2`, `Q3` |
| stage-1 experiments and analysis | `scripts/stage1_*.py` | scripts 05, 09, 10, 12, 13, 14 |
| stage-2 experiments and analysis | `scripts/stage2_*.py` | scripts 17, 18 |
| stage-3 experiments and analysis | `scripts/stage3_*.py` | scripts 21, 22 |

In result keys, RQ3 denotes the batched-acquisition question, which the paper numbers RQ4."""

README = """# Supplementary material

Code, results, and registered analysis plans for "Where to Measure Next? Evaluating Feedback-Driven Region
Selection for Spatial Transcriptomics".

## Contents

| Path | Contents |
|---|---|
| `src/stare/` | Simulator, imputers, policies, metrics, and study-level statistics |
| `scripts/` | Experiments, registered analyses, and the scripts that write every table and figure |
| `results/` | Every result reported in the paper (`earlier_slide_set/`: the seven-slide pilot used in Appendices A and C) |
| `registered_plans/` | Verbatim excerpts of the three registered analysis plans |
| `REGISTRATION_LOG.md` | When each plan and its results were committed |
| `sofisher_native_run/`, `sofisher_resume_patch/` | TensorBoard logs of the native SOFisher run and the resume patch used for it |
| `figures/` | Figure files the table scripts read or write |

## Paper to code

| Paper | Script | Result file |
|---|---|---|
| Slide set (Section 3.2, Appendix A) | `select_slides.py`, `unit_scale_audit.py` | `slide_selection.json`, `unit_scale_audit.json` |
| Synthetic controls (Figure 1) | `control_synthetic_reconstruction.py`, `control_synthetic_detection.py` | `control_synthetic_*.json` |
| Stage 1 (Appendix D, RQ4 batched) | `stage1_*.py`; analysis `stage1_analysis.py` | `stage1_*.json` |
| Stage 2, RQ1 and RQ2 (Tables 1-4, Figure 2) | `stage2_experiments.py`, `stage2_analysis.py` | `stage2_runs.json`, `stage2_summary.json` |
| Comparator inference (Appendix F) | `stage2_comparator_inference.py` | `stage2_comparator_inference.json` |
| Stage 3, RQ3 and RQ4 (Tables 5-6, Figure 3) | `stage3_experiments.py`, `stage3_analysis.py` | `stage3_runs.json`, `stage3_summary.json` |
| Native SOFisher run (Appendix L) | `sofisher_native_curve.py` | `sofisher_native.json` |
| Post hoc robustness checks (Appendix M) | `posthoc_robustness.py` | `posthoc_robustness.json` |
| All tables and figures | `tables_stage1.py`, `tables_stage2.py`, `tables_stage3.py` | `paper/generated/`, `paper/*.pdf` |

## Regenerating every table and figure (no data download)

    pip install -e .
    python -u scripts/tables_stage1.py
    python -u scripts/tables_stage2.py
    python -u scripts/tables_stage3.py
    python -u scripts/posthoc_robustness.py

The analysis scripts (`stage1_analysis.py`, `stage2_analysis.py`, `stage2_comparator_inference.py`,
`stage3_analysis.py`) recompute the summaries from the run files. Rerunning the experiments requires the gated
HEST-1k dataset (Hugging Face `MahmoodLab/hest`) and UNI2-h weights (`MahmoodLab/UNI2-h`); install PyTorch for
your CUDA version, then `pip install -r requirements.txt`. All settings and seeds are in `src/stare/config.py`.

## Names

""" + NAME_TABLE + "\n"

NOTICE = """# Third-party notices

`sofisher_native_run/` holds TensorBoard logs from a native training run of SOFisher, and
`sofisher_resume_patch/apply_resume.py` adds checkpoint and resume support to a SOFisher clone. SOFisher is
distributed under the MIT License (Copyright (c) 2024 Heathcliff Ng); the full license text accompanies the
SOFisher repository.

HEST-1k and UNI2-h are gated on Hugging Face under their own terms. This package contains no HEST data files or
UNI weights; `results/slide_selection.json` lists a few metadata fields per selected slide (identifier, organ,
platform, pixel size, spot count), as in the paper's slide table.
"""


def collect():
    files = {}

    def add(src, dst, text=None):
        files[TOP + dst] = text if text is not None else (ROOT / src).read_bytes()

    for o, n in MODULES.items():
        add(f"src/stare/{o}", f"src/stare/{n}")
    for o, n in SCRIPTS.items():
        add(f"scripts/{o}", f"scripts/{n}")
    for o, n in RESULTS.items():
        add(f"results/{o}", f"results/{n}")
    for f in LEGACY:
        add(f"results/n7_legacy/{f}", f"results/earlier_slide_set/{rename(f, False)}")
    for f in sorted((ROOT / "artifacts" / "sofish_seed222_5m").glob("events.out.tfevents.*")):
        add(str(f.relative_to(ROOT)), "sofisher_native_run/" + re.sub(r"\.LAPTOP-[A-Z0-9]+\.", ".host.", f.name))
    add("docs/figures/figure4_sofisher_native.pdf", "figures/sofisher_native_curve.pdf")
    add("sofisher_resume/apply_resume.py", "sofisher_resume_patch/apply_resume.py")
    add("pyproject.toml", "pyproject.toml")
    add("requirements.txt", "requirements.txt")
    lic = (ROOT / "LICENSE").read_text(encoding="utf-8")
    add(None, "LICENSE", re.sub(r"Copyright \(c\) 2026 .*", "Copyright (c) 2026 Anonymous authors", lic).encode())
    add(None, "THIRD_PARTY_NOTICES.md", NOTICE.encode())
    add(None, "README.md", README.encode())
    pfiles, prows = plans()
    for k, v in pfiles.items():
        add(None, k, v.encode())
    add(None, "REGISTRATION_LOG.md", registration_log(prows).encode())

    # transform code and results
    for name in list(files):
        rel = name[len(TOP):]
        if rel.endswith(".py"):
            text = rename(files[name].decode("utf-8"), True)
            text = clean_python(text, rel)
            if rel == "src/stare/config.py":    # settings no packaged script reads (earlier development stages)
                for pat in (r"(?m)^VALIDATION_TISSUE = .*\r?\n", r"(?m)^FEASIBILITY_TISSUE = .*\r?\n",
                            r"(?m)^FEASIBILITY_BUDGET_FRACTION = .*\r?\n", r"(?ms)^TIER1_TOLERANCE = \{.*?^\}\r?\n",
                            r"(?ms)^@dataclass\r?\nclass RunConfig:.*?(?=^\S|\Z)"):
                    text, k = re.subn(pat, "", text)
                    assert k == 1, pat
                text = text.replace("from dataclasses import dataclass, field\n", "")
                text = text.rstrip() + "\n"
            files[name] = text.encode("utf-8")
        elif rel.endswith(".json"):
            files[name] = rename(files[name].decode("utf-8"), False).encode("utf-8")
        elif rel == "requirements.txt":         # inline comments are setup notes
            files[name] = re.sub(r"(?m)[ \t]*#.*$", "", files[name].decode("utf-8")).encode("utf-8")
        elif rel == "pyproject.toml":           # whole-line comments only (TOML values may contain '#')
            files[name] = re.sub(r"(?m)^[ \t]*#.*\r?\n", "", files[name].decode("utf-8")).encode("utf-8")
    return files


def check(files):
    problems = []
    for name, data in files.items():
        rel = name[len(TOP):]
        if rel.endswith(".py"):
            try:
                compile(data, rel, "exec")
            except SyntaxError as e:
                problems.append(f"syntax {rel}: {e}")
        if rel.endswith(".json"):
            def hook(pairs):
                keys = [k for k, _ in pairs]
                if len(keys) != len(set(keys)):
                    problems.append(f"duplicate keys after renaming in {rel}")
                return dict(pairs)
            json.loads(data.decode("utf-8"), object_pairs_hook=hook)
        try:
            text = name.lower() + "\n" + data.decode("utf-8").lower()
            pats = FORBIDDEN_ID + ([] if rel.startswith("registered_plans/") else FORBIDDEN_NOTE)
        except UnicodeDecodeError:
            text = name.lower() + "\n" + data.lower().decode("latin-1")
            pats = [r"ayush", r"abhijeet", r"laptop-7bvkv3gm", r"southern california", r"gmail\.com"]
        for p in pats:
            m = re.search(p, text)
            if m:
                problems.append(f"term '{p}' in {rel}: ...{text[max(0, m.start() - 50):m.end() + 50]!r}...")
        if rel.endswith(".md") and HASH.search(data.decode("utf-8", "ignore")):
            problems.append(f"commit hash in {rel}")
    return problems


def reproduce(blob):
    """Regenerate the paper's tables from the extracted archive and compare with paper/generated."""
    back = {"stage2_": "r2_", "stage3_": "s3_"}
    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            z.extractall(d)
        pkg = Path(d) / "supplementary"
        smoke = ("import sys; sys.path.insert(0,'src'); import numpy as np\n"
                 "from stare.simulator import Slide, AcquisitionSimulator\n"
                 "from stare.imputer import build_imputer\nfrom stare.policies import REGISTRY, build_policy\n"
                 "rng=np.random.default_rng(0); R,G=150,60; lat=rng.normal(size=(R,8))\n"
                 "emb=np.concatenate([lat, rng.normal(scale=.1,size=(R,1528))],1); W=rng.normal(size=(8,G))\n"
                 "ex=np.maximum(lat@W+rng.normal(scale=.3,size=(R,G)),0); hv=np.argsort(-ex.var(0))[:50]\n"
                 "s=Slide(tissue='t',region_coords=rng.uniform(0,2000,(R,2)),image_embeddings=emb,expression=ex,hvg_idx=hv,patient_id='p')\n"
                 "for n in REGISTRY:\n"
                 "    imp=build_imputer('graphflow'); sim=AcquisitionSimulator(s,imp,0.05); st=sim.reset()\n"
                 "    pol=build_policy(n, imputer=imp) if n in ('error_reduction','residual') else build_policy(n)\n"
                 "    while not sim.done: st=sim.reveal(pol.select(st))\n"
                 "print('policies ok:', ', '.join(REGISTRY))\n")
        r = subprocess.run([sys.executable, "-c", smoke], cwd=pkg, capture_output=True, text=True)
        print("   ", (r.stdout or r.stderr).strip().splitlines()[-1])
        if r.returncode:
            return [f"policy smoke test failed: {r.stderr[-400:]}"]
        (pkg / "figures").mkdir(exist_ok=True)
        for s in ("tables_stage1.py", "tables_stage2.py", "tables_stage3.py", "posthoc_robustness.py"):
            r = subprocess.run([sys.executable, f"scripts/{s}"], cwd=pkg, capture_output=True, text=True)
            if r.returncode:
                return [f"{s} failed: {r.stderr[-600:]}"]
        bad, n = [], 0
        for f in sorted((pkg / "paper" / "generated").glob("*.tex")):
            name = f.name
            for new, old in back.items():
                if name.startswith(new):
                    name = old + name[len(new):]
            name = name.replace("feasible", "loso")     # the comparator table's name follows the renamed comparator
            ref = ROOT / "paper" / "generated" / name
            n += 1
            if not ref.exists() or ref.read_bytes() != f.read_bytes():
                bad.append(f"table differs from the paper: {f.name}")
        print(f"    tables regenerated from the package: {n}, identical to the paper: {n - len(bad)}")
        return bad


def main():
    files = collect()
    problems = check(files)
    if problems:
        for p in problems[:40]:
            print("FAIL", p)
        sys.exit(f"{len(problems)} problem(s); nothing written")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(TOP + "figures/", "")
        for name in sorted(files):
            z.writestr(name, files[name])
    problems = reproduce(buf.getvalue())
    if problems:
        for p in problems:
            print("FAIL", p)
        sys.exit("reproduction check failed; nothing written")
    OUT.write_bytes(buf.getvalue())
    print(f"[write] {OUT} ({len(files)} files, {OUT.stat().st_size / 1e6:.1f} MB; TMLR limit 100 MB)")


if __name__ == "__main__":
    main()
