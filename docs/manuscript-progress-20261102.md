# Standalone interim manuscript — 2 November 2026

The parallel manuscript task produced `paper/interim-20261102.tex` and
`paper/interim-20261102.pdf`, an English working report explicitly marked
incomplete with author details pending. It is separate from `paper/main.tex` and
the existing generated manuscript files. No submission or publication occurred.

The report covers the completed 7B experiment, paired text interventions,
truncation accuracy, all 64 steering/control comparisons, the timestamped 32B
baseline snapshot, and the evaluator audit. It includes the statistical agent's
text-contrast and activation-control figures. The 42 flagged cases and 30 sampled
unflagged cases are described as an exploratory assistant review, not human gold
labels or validated whole-run error rates. The original evaluator judgments are
preserved.

Build from the repository root:

```sh
env/bin/python scripts/build_interim_figures.py
cd paper
../.tools/tectonic --keep-logs --keep-intermediates interim-20261102.tex
```

The figure inputs are the completed 7B records and existing analyses under
`runs/study-20260925/distill-7b`. The baseline snapshot is
`artifacts/diagnostics/20261102/interim-evidence.json`; review evidence is
`artifacts/diagnostics/20261102/judge-review.json`. Figure files live under
`artifacts/diagnostics/20261102/figures/`. All paths in the TeX file are relative
to its `paper/` build directory.

Validation: Tectonic compilation, `pdfinfo` page count, TeX log inspection for
overflow/undefined references, and rasterized page inspection with `pdftoppm`.
The final report has six pages. Numeric tables were checked against 7B
`summary.json` and `robustness.json`; audit counts were checked against
`judge-review.json`. Figure calculations were independently verified from raw
records by the statistical task.

Still required for the primary manuscript:

- Complete Distill-32B generation, audit, assessment, and analysis.
- Run and audit QwQ-32B, then perform the paired checkpoint comparison.
- Obtain independent human assessment of reasoning labels; the assistant audit
  provides review candidates and failure examples, not a replacement gold set.
- Integrate the completed results and audited literature into the main manuscript.
- Supply authors, affiliations, and contact information.
- Pass the release-readiness audit before producing submission-ready files.

The main campaign and its frozen scientific files were not modified by this task.
