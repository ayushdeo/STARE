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
| Paper | `paper/manuscript.pdf` | TMLR template, unmodified style files; anonymous; 21 pages, main body 11 pages |
| Supplementary material | `paper/tmlr_supplementary.zip` | 12.4 MB (limit 100 MB); anonymized and scanned for identifying strings |

The supplement contains the code, results, registered plans, and `REGISTRATION_LOG.md`, which lists when each
analysis plan and its results were committed (commit identifiers withheld). All 20 tables regenerate byte-for-byte
from it.

## Form fields

**Title:** Where to Measure Next? Evaluating Feedback-Driven Region Selection for Spatial Transcriptomics

**Abstract** (plain text, 1,915 characters, identical to the PDF):

Spatial transcriptomics is costly, so many studies profile only selected regions of a section, and recent methods choose each next region from expression already measured. A recent benchmark of active spot selection for training predictors across patients reported inconsistent gains over random sampling; we ask instead whether, within a slide, measured feedback improves selection over a fixed design matched to the objective. We built a retrospective simulator from 22 human HEST-1k slides (18 studies, four tissues), fixed the image encoder and imputer, registered three analysis stages in advance, and drew inference at the study level. RQ1: for whole-section reconstruction, a feedback policy targeting the imputer's error was worse than a non-adaptive design chosen on held-out studies at every budget (0.5-20%), by up to 0.092 Pearson correlation; a semi-synthetic control did not establish sensitivity to reconstruction gains. RQ2: for capturing heterogeneous regions, informative feedback raised recall over permuted feedback by up to 0.063, but no margin survived multiplicity correction, and a diversity sampler did about as well. RQ3: when a coverage design spent half or more of the budget and feedback the rest, no hybrid beat the coverage design; from 2% upward all were within 0.007 correlation and 0.006 recall, although gains near 0.02 and 0.03 would probably have been detected at 2-10%. Feedback was informative (permuted values made the same policies worse) but mainly offset the cost of leaving coverage. RQ4 (exploratory): batches of five neither eroded nor established a detection advantage. Method developers should compare adaptive selection with objective-matched fixed designs and with itself fed uninformative values. For experimentalists, within simulated spot-level budgets, one imputer, and computational endpoints, no tested feedback policy outperformed a well-chosen fixed design.

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
- [x] Main body 11 pages; references before the appendix; table titles above tables, figure captions below.
- [x] All 50 references checked against Crossref, arXiv, JMLR, PMLR, or NeurIPS proceedings, and each cited claim
      checked against its source.
- [x] Supplement anonymized, under 100 MB, and reproduces every table.
- [ ] OpenReview profile complete (affiliation, conflicts, publication history).
- [ ] Funding and competing-interest fields filled in.
- [ ] Action editors chosen and checked for conflicts.
- [ ] At camera-ready: `\usepackage[accepted]{tmlr}`, set `\month`, `\year`, `\openreview`, replace `CONTACT-EMAIL`,
      add acknowledgements, and link the public repository.
