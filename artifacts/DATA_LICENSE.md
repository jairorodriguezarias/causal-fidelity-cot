# SNLI attribution

`dataset.json` is a selected and regrouped subset of the Stanford Natural Language
Inference corpus (SNLI 1.0). Source: https://nlp.stanford.edu/projects/snli/ .
Authors: Samuel R. Bowman, Gabor Angeli, Christopher Potts and Christopher D. Manning.
Paper: *A large annotated corpus for learning natural language inference*, EMNLP 2015.

The source corpus and this adapted data subset are distributed under Creative Commons
Attribution-ShareAlike 4.0 International: https://creativecommons.org/licenses/by-sa/4.0/ .
Changes: selecting entailment/contradiction examples, grouping by premise, deterministic
sampling, normalized-premise deduplication and separate construction/calibration/test
partitions. Revision, seed and checksum are recorded in `dataset.json`.

This attribution concerns the dataset; it does not license the downloaded model
weights or third-party software. Model weights retain their original licenses.
