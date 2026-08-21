"""
Builds the contrast dataset for the experiment.
Loads SNLI from HuggingFace and filters structurally parallel
Entailment/Contradiction pairs for steering vector computation.
"""

import json
import random
import os
from datasets import load_dataset

OUTPUT_FILE = "data/contrast_dataset.json"


def build_contrast_dataset(n_pairs: int = 500, seed: int = 42):
    print(f"[Dataset] Loading SNLI from HuggingFace...")
    snli = load_dataset("stanfordnlp/snli", split="train")
    random.seed(seed)

    premise_groups = {}
    for item in snli:
        premise = item["premise"].strip()
        hypothesis = item["hypothesis"].strip()
        label = item["label"]

        if label == 0:
            key = "entailment"
        elif label == 1:
            continue
        elif label == 2:
            key = "contradiction"
        else:
            continue

        if premise not in premise_groups:
            premise_groups[premise] = {"entailment": [], "contradiction": []}
        premise_groups[premise][key].append(hypothesis)

    pairs = []
    for premise, groups in premise_groups.items():
        if groups["entailment"] and groups["contradiction"]:
            entail_hyp = random.choice(groups["entailment"])
            contra_hyp = random.choice(groups["contradiction"])

            # CORREGIDO: Usar etiquetas nativas de DeepSeek-R1 (<think>)
            entail_prompt = f"Context: {premise} Hypothesis: {entail_hyp}\n<think>\n"
            contra_prompt = f"Context: {premise} Hypothesis: {contra_hyp}\n<think>\n"

            pairs.append({
                "premise": premise,
                "entail_hypothesis": entail_hyp,
                "contra_hypothesis": contra_hyp,
                "entail_prompt": entail_prompt,
                "contra_prompt": contra_prompt,
            })

        if len(pairs) >= n_pairs:
            break

    random.shuffle(pairs)
    pairs = pairs[:n_pairs]

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"[Dataset] {len(pairs)} contrast pairs saved to {OUTPUT_FILE}")
    return pairs


def load_contrast_dataset():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    build_contrast_dataset(n_pairs=500)
