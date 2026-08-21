import json
import time
from llm_judge import judge_single, cohen_kappa
from tqdm import tqdm

FILE = "results/DeepSeek-R1-Distill-Qwen-1.5B_v3.json"

print(f"Cargando {FILE}...")
with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Re-evaluando con el nuevo juez...")
sweep = data.get("layer_sweep", {})
keys = list(sweep.keys())

for key in tqdm(keys, desc="Re-judging"):
    entry = sweep[key]
    judgment = judge_single(
        entry.get("cot_text", ""), 
        entry.get("answer_text", ""), 
        entry.get("test_context", "")
    )
    entry["judge_category"] = judgment.get("categoria", "AMBIGUOUS")
    entry["judge_reasoning"] = judgment.get("razonamiento", "")
    time.sleep(0.3)

# Recalcular Kappa en los primeros 20
sample_keys = keys[:20]
first_labels = [sweep[k]["judge_category"] for k in sample_keys]
second_labels = []
for k in tqdm(sample_keys, desc="Kappa Re-judging"):
    entry = sweep[k]
    j = judge_single(entry.get("cot_text", ""), entry.get("answer_text", ""), entry.get("test_context", ""))
    second_labels.append(j.get("categoria", "AMBIGUOUS"))
    time.sleep(0.3)

kappa = cohen_kappa(first_labels, second_labels)
data["kappa"] = kappa
print(f"Nuevo Cohen's Kappa: {kappa:.3f}")

with open(FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("¡Re-evaluación guardada con éxito!")

# Imprimir resumen rápido
cats = [sweep[k]["judge_category"] for k in keys]
print(f"FAITHFUL: {cats.count('FAITHFUL_SHIFT')}/{len(cats)}")
print(f"THEATER: {cats.count('REASONING_THEATER')}/{len(cats)}")
