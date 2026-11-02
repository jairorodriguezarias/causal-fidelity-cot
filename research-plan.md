# Research plan for an arXiv preprint

## Question and scope

How do hidden-state steering and direct interventions on generated reasoning affect
answers in distilled and RL-post-trained reasoning models? The primary comparison is
DeepSeek-R1-Distill-Qwen-32B versus the released QwQ-32B. The 1.5B and 7B distilled
models provide implementation and scale pilots.

The comparison matches parameter scale and the Qwen2.5 base family. It does not
isolate a training-method effect: the models differ in training data and post-training
recipes, and QwQ includes both supervised fine-tuning and reinforcement learning.
No claim about an irrevocable decision layer follows merely from an answer flip.

## Data

`artifacts/dataset.json` contains 500 training contrast pairs, 50 calibration pairs,
and 50 test pairs. Each pair shares a premise and contains one entailment hypothesis
and one contradiction hypothesis from SNLI. Official train/validation/test splits
supply these partitions; normalized premises are disjoint across all three. Candidate
premises are shuffled before selection. Source revision, seed and SHA-256 are saved.
SNLI consensus labels are imperfect, and its public availability creates a possible
pretraining-contamination limitation. This binary contrast task is not the standard
three-class SNLI benchmark.

## Interventions

1. Apply the model's own chat template and an opening thought marker. Generate an
   explicit `FINAL: ENTAILMENT` or `FINAL: CONTRADICTION` field. Preserve full text,
   token IDs, stop reason, prompt, seed and parse status.
2. At each selected layer, compute the mean activation difference between contradiction
   and entailment prompts using only training pairs. Extract all selected layers in
   the same forward pass; accumulate in float32.
3. Add alpha times that mean difference for the first 15 prediction positions,
   including the last prefill position predicting the first generated token.
   Alpha is in mean-difference units, not arbitrary cross-model residual units.
4. Use a separate baseline for every example. At every tested layer and alpha, compare
   semantic steering with an equal-norm orthogonal direction and an equal-norm
   normalized Gaussian direction. Control directions remain fixed across the 15
   positions and are independently seeded per example and layer.
5. Retain 0%, 25%, 50%, 75% or 100% of a completed baseline reasoning trace, close the
   thought block, and regenerate the answer. The 100% condition controls for the
   answer-regeneration procedure. Also substitute a trace generated for the paired
   contradiction hypothesis, and use shuffled reasoning tokens as a corruption control.

Calibration selects operating settings before the held-out comparison. The same
configuration, quantization, task prompt and evaluation examples are used for both
32B models. Full-precision 1.5B results are an implementation pilot; comparisons
with quantized models do not identify pure scale effects.

## Measurement and inference

The primary behavioral outcome is the answer shift toward contradiction. Report
baseline accuracy, answer completion/format rates, conditional flips among baseline
entailment answers, and bounds treating missing intervention answers as unknown.
Paired control differences are bootstrapped over examples. Repeated layer/alpha
observations are not independent samples. Report exploratory intervals and the
multiple-comparison limitation; do not select a favorable test cell as confirmatory.

An independent model assesses only the reasoning text, blinded to the final answer.
Its JSON schema and evidence quotations are validated. Categories describe alignment
or disagreement between stated reasoning and final answer, not causal faithfulness.
Synthetic validation is an implementation check, not human validation on model traces.
Retest agreement and errors are reported separately. Undefined Cohen's kappa is null,
not perfect reliability. Quote-based drift distance is an approximate textual indicator,
not a measurement of internal decision time.

Steering changes hidden states that can affect both reasoning and answer directly.
It therefore does not identify mediation through the written reasoning. Text truncation,
substitution and corruption provide additional behavioral interventions, but unchanged
answers may reflect redundant reasoning, and changed answers may reflect distribution
shift. Neither observation alone proves or disproves post-hoc rationalization.

## Deliverables and completion gates

See [the completion checklist](docs/completion-plan.md). Deliver executed data and
controls, reproducible manifests, tests, analysis tables and plots, a manuscript PDF,
and an arXiv source archive. The manuscript must describe only completed experiments;
missing work must remain explicitly pending. A PRM-versus-ORM study is a separate
future extension requiring suitably documented and matched training conditions.

## Verified primary references

- [SNLI corpus and original paper](https://nlp.stanford.edu/projects/snli/).
- [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702).
- [DeepSeek-R1 report](https://arxiv.org/abs/2501.12948).
- [QwQ-32B official model card](https://huggingface.co/Qwen/QwQ-32B).
- [Chain-of-Thought Reasoning In The Wild Is Not Always Faithful](https://arxiv.org/abs/2503.08679).
- [Post-Hoc Reasoning in Chain of Thought: Decoding and Steering Pre-Committed Answers](https://arxiv.org/abs/2603.01437).
- [Faithfulness as Information Flow](https://arxiv.org/abs/2605.24286).
