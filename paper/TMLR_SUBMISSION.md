# TMLR submission sheet

Transactions on Machine Learning Research: rolling submissions on OpenReview, double-blind review, no fee,
published under CC BY 4.0 with copyright retained by the authors. Acceptance rests on two questions: are the claims
supported by accurate and convincing evidence, and would some of TMLR's audience be interested. Novelty is not a
criterion. Guides: https://jmlr.org/tmlr/author-guide.html, https://jmlr.org/tmlr/acceptance-criteria.html,
https://jmlr.org/tmlr/editorial-policies.html.

## Build the two files you upload

```bash
cd paper && tectonic --keep-intermediates manuscript.tex && cd ..   # paper/manuscript.pdf (anonymous)
python scripts/25_tmlr_supplement.py                                 # paper/tmlr_supplementary.zip
```

| Upload | File | Status |
|---|---|---|
| Paper | `paper/manuscript.pdf` | TMLR template, unmodified style files; anonymous; 25 pages, main body 12 pages |
| Supplementary material | `paper/tmlr_supplementary.zip` | 2.6 MB (limit 100 MB), 88 files; anonymized and scanned |

The supplement is a curated reproduction package, built from an allow-list: the package code, the scripts that
produce the paper's results, tables, and figures, the result files they read, the native SOFisher logs, verbatim
excerpts of the three registered plans, and `REGISTRATION_LOG.md` (commit dates; identifiers withheld). Files,
policies, and result keys use the paper's names (a table in its README maps them to the names in the plans);
working notes, earlier development scripts, tests, and model weights are excluded, and code comments are removed.
The builder refuses to write the archive unless every table regenerates from it byte-for-byte, all policies run,
renaming creates no duplicate result keys, and no identifying string or development term remains.

## Form fields

**Title:** Where to Measure Next? Evaluating Feedback-Driven Region Selection for Spatial Transcriptomics

**Abstract** (plain text, 1,927 characters, identical to the PDF):

Spatial transcriptomics is costly, so many studies profile only selected regions of a section, and recent methods choose each next region from expression already measured. A recent benchmark of active spot selection for training predictors across patients reported inconsistent gains over random sampling; we ask instead whether, within a slide, measured feedback improves selection over a fixed design matched to the objective. We built a retrospective simulator from 22 human HEST-1k slides (18 studies, four tissues), fixed the image encoder and imputer, registered three analysis stages in timestamped commits before running them, and drew inference at the study level. Each feedback policy was compared with a non-adaptive design chosen on held-out studies and with itself fed uninformative values. RQ1: for whole-section reconstruction, a policy targeting the imputer's error was below the fixed design at every budget (0.5–20%), significantly at four of six and by up to 0.092 Pearson correlation. RQ2: for capturing heterogeneous regions, informative feedback raised recall over permuted feedback by up to 0.063 without surviving multiplicity correction, and a diversity sampler that uses no feedback did about as well. RQ3: spending half or more of the budget on a fixed design and the rest on feedback never beat that design; from 2% upward every hybrid was within 0.007 correlation and 0.006 recall of it, a null with at least 80% power to detect gains of 0.02 and 0.03 at 2–10%. RQ4 (exploratory): batches of five neither eroded nor established a detection advantage. No feedback policy we tested outperformed a well-chosen fixed design on either objective, within simulated spot-level budgets, one imputer, and computational endpoints. Adaptive acquisition methods should be benchmarked against objective-matched fixed designs and against themselves without informative feedback, not only against random selection.

**Authors:** your OpenReview profile. TMLR asks for a complete, active profile with affiliation, conflicts of
interest, and publication history; these are hidden from reviewers until a decision.

**Suggested action editors** (from TMLR's editorial board, with the expertise the board lists; check each for
conflicts before suggesting):

| Action editor | Affiliation | Listed expertise |
|---|---|---|
| Chao Chen | Stony Brook University | spatial transcriptomics, digital pathology, medical image analysis |
| Arash Mehrjou | GlaxoSmithKline | active learning, experiment design, computational biology |
| Ge Liu | University of Illinois Urbana-Champaign | computational biology, active learning |
| Roman Garnett | Washington University in St. Louis | active learning, Bayesian optimization |
| Jana Doppa | Washington State University | adaptive experimental design, sequential decision-making |
| Ilan Shomorony | University of Illinois Urbana-Champaign | computational biology, bioinformatics, genomics |

The paper cites Garnett et al. (2012) and Sener & Savarese (2018); citing an editor's work is not a conflict.

**Human subjects:** the study collects no new data from people; it analyses only publicly released, de-identified
HEST-1k data under its access terms. Answer the form's IRB question accordingly (confirm with USC policy if unsure).

**Funding and competing interests:** fill in your own (none are stated in the paper).

**Broader impact statement:** not required; TMLR asks for one only when the work carries a significant risk of
harm.

## Policies to keep in mind

- No concurrent submission to another archival, peer-reviewed venue while under review. Non-archival workshops are
  allowed.
- Preprints are allowed at any time, but the submission must not link to a named version. Build one later with
  `scripts/24_arxiv_package.py` (it uses the template's `[preprint]` option and refuses to run until
  `CONTACT-EMAIL` is replaced).
- LLMs may be used as assistive tools; the authors are fully responsible for the content.
- Timeline: an action editor within a week, then reviews, a discussion and revision period, and a decision (accept
  as is, accept with minor revisions, or reject), typically about 9 weeks after submission for papers whose main
  body is at most 12 pages.

## Checklist

- [x] TMLR style files, unmodified; anonymous build (`\usepackage{tmlr}`).
- [x] No author names, affiliations, or repository links in the PDF; code is referenced only as supplementary
      material.
- [x] Main body 12 pages (the limit for TMLR's normal review timeline; add to it only with cuts); references before the appendix; table titles above tables, figure captions below.
- [x] All 50 references checked against Crossref, arXiv, JMLR, PMLR, or NeurIPS proceedings, and each cited claim
      checked against its source.
- [x] Supplement anonymized, under 100 MB, and reproduces every table.
- [ ] OpenReview profile complete (affiliation, conflicts, publication history).
- [ ] Funding and competing-interest fields filled in.
- [ ] Action editors chosen and checked for conflicts.
- [ ] At camera-ready: `\usepackage[accepted]{tmlr}`, set `\month`, `\year`, `\openreview`, replace `CONTACT-EMAIL`,
      add acknowledgements, and link the public repository.
