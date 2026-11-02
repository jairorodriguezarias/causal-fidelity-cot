# Completion plan — reviewed 2026-11-02

Target: execute a reproducible study of CoT sensitivity in distilled and RL-post-trained
models and prepare an evidence-based arXiv manuscript and reproduction bundle.
Publication itself remains the author's action; authorship details are pending.

## Required evidence

- [x] Correct extraction, complete raw generations, validated labels and missing-data handling.
- [x] At least 500 contrast pairs, premise-disjoint construction/calibration/evaluation sets,
      pinned dataset/model revisions and deterministic recorded seeds.
- [x] Verified hooks, including the prefill position predicting the first generated token;
      equal-norm orthogonal and Gaussian controls and per-example unsteered baselines.
- [x] Executed 1.5B/7B pilots and the held-out 7B generation, audit and analysis.
- [x] Completed 7B layer/intensity sweep, CoT truncation and direct text interventions.
- [x] Example-level 7B uncertainty, paired controls, figures and machine-readable tables.
- [ ] Complete Distill-32B generation, audit, assessment and analysis (generation active).
- [ ] Execute QwQ-32B and the matched primary checkpoint comparison.
- [ ] Independently validate reasoning labels on real traces. Synthetic validation and
      retests are available, but the exploratory audit found evaluator errors.
- [ ] Complete uncertainty, controls and figures for the primary 32B comparison.
- [x] Verified bibliography, accurate model-training descriptions and limitations.
- [x] Standalone interim manuscript and figures, explicitly marked incomplete.
- [x] Implementation tests and reproduction instructions.
- [ ] Final manuscript, author metadata, reproduction bundle and publication-readiness audit.

The [interim findings](interim-findings-20261102.md) and
[evaluator audit](judge-audit-20261102.md) describe the current evidence and limits.
Preparing source for GitHub does not complete the experiment or authorize publication
of a final scientific result. Raw runs and downloaded weights remain outside Git.

## Initial audit

The August v3 files are historical debugging outputs, not research evidence: all 64
CoT fields are empty, each model tests one context, and the local dataset has only
10 pairs with train/test overlap under the previous slicing. Preserve these files;
write all corrected runs to a new directory. The previous judge-model setting was
a local uncommitted edit; retain it as a backwards-compatible environment default
when replacing the evaluator.

The official QwQ-32B card lists both supervised fine-tuning and reinforcement
learning. Use the released QwQ-32B, record this choice, and do not describe the
comparison as a controlled SFT-versus-pure-RL experiment. Activation steering is a
causal intervention on hidden states, not identification of textual mediation.
Truncation insensitivity alone is not proof of post-hoc reasoning.
