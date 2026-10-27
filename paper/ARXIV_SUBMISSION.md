# arXiv submission sheet

Everything needed for the arXiv submission form. Rebuild the package after any change to the paper:

```bash
cd paper && tectonic --keep-intermediates manuscript.tex && cd ..
python scripts/24_arxiv_package.py
```

The script writes `paper/arxiv_submission.tar.gz` and the same files unpacked in `paper/arxiv_upload/` (28 files:
`manuscript.tex`, `manuscript.bbl`, `tmlr.sty`, `fancyhdr.sty`, 20 generated tables, 4 figure PDFs); upload either one.
In the uploaded copy only: TeX comments are removed (arXiv archives the source permanently), and
`\bibliography{references}` is replaced by `\input{manuscript.bbl}`, so arXiv never needs BibTeX or the `.bib`.
The script then compiles the package in an empty folder and checks that its text is identical to
`paper/manuscript.pdf`, citations included. The figure PDFs carry no author or path metadata.

## Form fields

**Title**

Where to Measure Next? Evaluating Feedback-Driven Region Selection for Spatial Transcriptomics

**Authors**

Ayush Deo

**Abstract** (1,912 characters; arXiv limit 1,920; plain text, identical in content to the PDF)

Spatial transcriptomics is costly, so many studies profile only selected regions of a section, and recent methods choose each next region from expression already measured. Benchmarks of active spot selection for training predictors across patients report inconsistent gains over random sampling; we ask instead whether, within a slide, measured feedback improves selection over a fixed design matched to the objective. We built a retrospective simulator from 22 human HEST-1k slides (18 studies, four tissues), fixed the image encoder and imputer, registered three analysis stages in advance, and drew inference at the study level. RQ1: for whole-section reconstruction, a feedback policy targeting the imputer's error was worse than a non-adaptive design chosen on held-out studies at every budget from 0.5% to 20%, by up to 0.092 Pearson correlation; a semi-synthetic control did not establish sensitivity to reconstruction gains. RQ2: for capturing heterogeneous regions, informative feedback raised recall over permuted feedback by up to 0.063, but no margin survived multiplicity correction, and a diversity sampler did about as well. RQ3: when a coverage design spent half or more of the budget and feedback the rest, no hybrid beat the coverage design; from 2% upward all were within 0.007 correlation and 0.006 recall, although gains near 0.02 and 0.03 would probably have been detected at 2-10%. Feedback was informative (permuted values made the same policies worse) but mainly offset the cost of leaving coverage. RQ4 (exploratory): batches of five neither eroded nor established a detection advantage. Method developers should compare adaptive selection with objective-matched fixed designs and with itself fed uninformative values. For experimentalists, within simulated spot-level budgets, one imputer, and computational endpoints, no tested feedback policy outperformed a well-chosen fixed design.

**Comments**

20 pages, 4 figures, 20 tables. Code, results, and registered analysis plans: https://github.com/ayushdeo/STARE

**Categories**: primary cs.LG; cross-lists q-bio.QM, q-bio.GN.

**License**: arXiv.org perpetual, non-exclusive license (the default; compatible with later journal submission).

## Before submitting

- [x] Repository is public, MIT-licensed, and its README matches the paper (the paper's Reproducibility paragraph links it).
- [ ] Category endorsement: first-time submitters may need an endorser for the primary category.
- [ ] The arXiv preview PDF matches `paper/manuscript.pdf` (20 pages; references 1-50 resolved).
- [ ] After posting: add the arXiv identifier to the repository README's Citation section.
