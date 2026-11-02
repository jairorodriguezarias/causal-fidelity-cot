# Reference and interpretation audit — 2026-09-25

The earlier proposal mixed hypotheses, empirical claims and implementation status.
The current plan replaces it with a reproducible protocol. Primary sources checked:

| Source | Verified use in this project |
|---|---|
| https://nlp.stanford.edu/projects/snli/ | SNLI task, Bowman et al. (2015), CC BY-SA 4.0 attribution |
| https://arxiv.org/abs/2307.13702 | Existing work on behavioral tests of CoT faithfulness |
| https://arxiv.org/abs/2501.12948 | DeepSeek-R1 and distillation context |
| https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | Distilled checkpoint identification |
| https://huggingface.co/Qwen/QwQ-32B | 32.5B parameters, 64 layers, Qwen2.5 foundation, **SFT and RL**, chat-template/thought-prefix and sampling guidance |
| https://qwenlm.github.io/blog/qwq-32b/ | Official description of QwQ reinforcement learning |
| https://arxiv.org/abs/2503.08679 | Naturally occurring unfaithfulness; original submission 2025, revised 2026 |
| https://arxiv.org/abs/2603.01437 | Cox, Kianersi and Garriga-Alonso: pre-CoT probes and steering; v2 title verified |
| https://arxiv.org/abs/2605.24286 | Jia, Benton and Easley: structural information-flow perspective |

Removed unsupported generalizations about a universal 2.8B threshold, guaranteed
benefits of process rewards, and a single-variable SFT-versus-pure-RL comparison.
The cited percentages in the old overview were results from another study, not this
project. They are unnecessary to the present empirical claims. Sources not checked
or not used are omitted from the manuscript bibliography rather than treated as
verified support. The manuscript distinguishes answer steering, textual consistency,
and causal mediation.
