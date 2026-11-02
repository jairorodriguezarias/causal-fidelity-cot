# Supplementary analysis plan — 2026-09-25

Prepared while the frozen campaign was loading the first held-out checkpoint, before
its outcome tables were available. This is a local, exploratory analysis plan, not an
external preregistration. It does not alter model generation, example selection,
intervention settings or the primary analysis script frozen with the campaign.

Run `env/bin/python robustness.py RUN_DIR` after a run completes. The output extends
the primary descriptive analysis with:

- Baseline accuracy by class, plus a balanced-accuracy bootstrap resampling the two
  hypotheses together at the premise level.
- Treatment-minus-control contradiction-rate differences paired by example. For
  missing labels, sharp binary-outcome bounds use the information still observed
  in each pair. These bounds are separate from bootstrap sampling intervals.
- Exact paired tests based on discordant outcomes, with Holm adjustment for the full
  activation/control family within each model. Unassessable planned contrasts remain
  in that family size. Cross-model and text contrasts remain exploratory.
- Truncation, substitution and shuffle comparisons against the regenerated answer
  after the full original CoT prefix. Report both all eligible examples and the
  subset with originally correct entailment baselines. Also report the change caused
  by full-prefix answer regeneration itself.
- Missing-outcome heatmaps alongside answer-shift plots, so a high-intensity loss of
  answer completion cannot be mistaken for absence of an effect.

Complete-case estimates can still be selected by treatment-dependent missingness.
Multiplicity adjustment does not establish external validity, causal mediation or
causal effects of the models' training methods. The final manuscript must report
these limitations and all prespecified conditions, including null and failed cases.
