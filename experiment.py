"""
Causal Fidelity of Chain-of-Thought Reasoning in Distilled vs. RL-Trained Models
"""

import torch
import json
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from dataset_builder import build_contrast_dataset, load_contrast_dataset
from llm_judge import judge_single, cohen_kappa

# =====================================================================
# CONFIGURATION
# =====================================================================
# =====================================================================
# CONFIGURATION — ADJUSTED FOR MAC M1 (PILOT PHASE)
# =====================================================================
MODELS_CONFIG = [
    {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # <- Cambiado a 1.5B para tu Mac
        "type": "distill",
        "description": "Distillation (SFT) — Pilot model for M1",
        "n_layers": 28,
        "sweep_layers": [0, 4, 8, 12, 16, 20, 24, 27],
    },
    # Deja esto comentado. Lo usaremos en Colab o en Berkeley.
    # {
    #     "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    #     "type": "distill",
    #     "n_layers": 28,
    #     "sweep_layers": [0, 4, 8, 12, 16, 20, 24, 27],
    # },
]

ALPHA_SWEEP = [1.0, 2.0, 4.0, 8.0]
K_TOKENS = 15
MAX_NEW_TOKENS = 300
OUTPUT_DIR = "results"
N_CONTRAST_PAIRS = 500
N_TEST_PROMPTS = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"[System] Device: {device}")

# =====================================================================
# CORE FUNCTIONS
# =====================================================================
# =====================================================================
# MODEL LOADING (ADJUSTED FOR DGX MULTI-GPU)
# =====================================================================
def load_model(model_id: str):
    print(f"\n[Loading] {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    
    # Usamos device_map="auto" para repartir el modelo en las GPUs del DGX
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        device_map="auto",  # <-- Esto reparte el modelo en múltiples GPUs
        low_cpu_mem_usage=True
    )
    # No usamos .to(device) aquí porque device_map="auto" ya lo colocó
    model.eval()
    return model, tokenizer


def _extract_last_hidden(output):
    t = output[0] if isinstance(output, tuple) else output
    if t.ndim == 3: return t[:, -1, :].squeeze(0).detach().clone()
    elif t.ndim == 2: return t[-1, :].detach().clone()
    raise ValueError(f"Unexpected shape: {t.shape}")

def get_activation(model, tokenizer, prompt, layer_idx):
    activation = None
    def hook_fn(module, inp, out):
        nonlocal activation
        activation = _extract_last_hidden(out)
    if layer_idx >= len(model.model.layers):
        return torch.zeros(model.config.hidden_size, device=device)
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad(): model(**inputs)
    handle.remove()
    return activation

def compute_steering_vector(model, tokenizer, entail_prompts, contra_prompts, layer_idx):
    e_acts = [get_activation(model, tokenizer, p, layer_idx) for p in entail_prompts]
    c_acts = [get_activation(model, tokenizer, p, layer_idx) for p in contra_prompts]
    mu_e = torch.stack(e_acts).mean(dim=0)
    mu_c = torch.stack(c_acts).mean(dim=0)
    direction = mu_c - mu_e
    norm = torch.norm(direction, p=2)
    if norm.item() < 1e-8: return direction, 0.0
    return direction / norm, norm.item()

class SteeringHook:
    def __init__(self, vector, alpha, k_tokens, mode="steer"):
        # No lo enviamos a device aquí todavía, lo haremos dinámicamente
        self.vector = vector
        self.alpha = alpha
        self.k_tokens = k_tokens
        self.t = 0
        self.mode = mode

    def __call__(self, module, inp, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        seq_len = hidden.shape[1] if hidden.ndim == 3 else hidden.shape[0]
        
        if seq_len == 1 and self.t < self.k_tokens and self.mode != "none":
            # CAMBIO CLAVE: Enviar el vector a la misma GPU que el 'hidden'
            v = self.vector.to(hidden.device)
            
            if self.mode == "orthogonal":
                v_orth = torch.randn_like(v)
                proj = (v_orth @ v) * v
                v_orth = v_orth - proj
                v_orth = (v_orth / (torch.norm(v_orth, p=2) + 1e-8)) * torch.norm(v)
                delta = self.alpha * v_orth
            elif self.mode == "gaussian":
                delta = self.alpha * torch.randn_like(v) * v.std()
            else:  # steer
                delta = self.alpha * v
                
            if hidden.ndim == 3:
                hidden[:, -1, :] += delta
            else:
                hidden[-1, :] += delta
            self.t += 1
        return (hidden,) + output[1:] if is_tuple else hidden


def generate_with_steering_at_layer(model, tokenizer, prompt, v_steer, alpha, k_tokens, layer_idx, mode="steer"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hook = SteeringHook(v_steer, alpha, k_tokens, mode=mode)
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    handle.remove()
    return tokenizer.decode(out_ids[0], skip_special_tokens=False)

# CORREGIDO: Usar etiquetas nativas <think> y
def parse_cot_answer(text):
    match = re.search(r'<think>(.*?)', text, re.DOTALL)
    cot = match.group(1).strip() if match else ""
    parts = text.split("</think>")
    ans = parts[-1].strip() if len(parts) > 1 else ""
    return cot, ans

# =====================================================================
# MAIN PIPELINE
# =====================================================================
def run_experiment_for_model(model_cfg, dataset):
    model_id = model_cfg["id"]
    model_type = model_cfg["type"]
    sweep_layers = model_cfg["sweep_layers"]
    model_name = model_id.split("/")[-1]

    model, tokenizer = load_model(model_id)
    contrast_pairs = dataset[:N_CONTRAST_PAIRS - N_TEST_PROMPTS]
    test_pairs = dataset[-N_TEST_PROMPTS:]

    entail_prompts_train = [p["entail_prompt"] for p in contrast_pairs]
    contra_prompts_train = [p["contra_prompt"] for p in contrast_pairs]
    test_prompts = [p["entail_prompt"] for p in test_pairs]
    test_contexts = [f"Context: {p['premise']} Hypothesis: {p['entail_hypothesis']}" for p in test_pairs]

    all_results = {"model": model_id, "type": model_type, "layer_sweep": {}, "red_teaming": {}, "kappa": None}

    print(f"\nPHASE 2-3: Layer-Sweep — {model_name} ({model_type})")
    for layer_idx in tqdm(sweep_layers, desc=f"Layers ({model_type})"):
        v_steer, norm = compute_steering_vector(model, tokenizer, entail_prompts_train, contra_prompts_train, layer_idx)
        if norm < 1e-8: continue

        for alpha in ALPHA_SWEEP:
            n_test_per_cell = max(1, N_TEST_PROMPTS // (len(sweep_layers) * len(ALPHA_SWEEP)))
            for i in range(n_test_per_cell):
                idx = i % len(test_prompts)
                output_text = generate_with_steering_at_layer(model, tokenizer, test_prompts[idx], v_steer, alpha, K_TOKENS, layer_idx, mode="steer")
                cot, ans = parse_cot_answer(output_text)

                key = f"L{layer_idx}_a{alpha}_i{i}"
                all_results["layer_sweep"][key] = {
                    "layer": layer_idx, "alpha": alpha, "test_index": idx,
                    "cot_text": cot[:500], "answer_text": ans[:500],
                    "full_output": output_text[:1000], "test_context": test_contexts[idx],
                }

    print(f"\nPHASE 4: LLM-as-a-Judge — {model_name}")
    for key, entry in tqdm(all_results["layer_sweep"].items(), desc="Judging"):
        judgment = judge_single(entry["cot_text"], entry["answer_text"], entry["test_context"])
        entry["judge_category"] = judgment.get("categoria", "AMBIGUOUS")
        entry["judge_reasoning"] = judgment.get("razonamiento", "")
        entry["judge_drift_distance"] = judgment.get("distancia_deriva", 0)
        time.sleep(0.3)

    print("  Computing Cohen's Kappa (test-retest on 20 samples)...")
    sample_keys = list(all_results["layer_sweep"].keys())[:20]
    sample_results = [all_results["layer_sweep"][k] for k in sample_keys]
    first_labels = [r["judge_category"] for r in sample_results]
    second_labels = []
    for r in tqdm(sample_results, desc="Re-judging"):
        j = judge_single(r["cot_text"], r["answer_text"], r["test_context"])
        second_labels.append(j.get("categoria", "AMBIGUOUS"))
        time.sleep(0.3)
    kappa = cohen_kappa(first_labels, second_labels)
    all_results["kappa"] = kappa
    print(f"  Cohen's Kappa: {kappa:.3f}")

    print(f"\nPHASE 5: Red-Teaming — {model_name}")
    control_layer = sweep_layers[len(sweep_layers) // 2]
    v_steer, _ = compute_steering_vector(model, tokenizer, entail_prompts_train, contra_prompts_train, control_layer)
    for mode in ["steer", "orthogonal", "gaussian", "none"]:
        output_text = generate_with_steering_at_layer(model, tokenizer, test_prompts[0], v_steer, 4.0, K_TOKENS, control_layer, mode=mode)
        cot, ans = parse_cot_answer(output_text)
        judgment = judge_single(cot, ans, test_contexts[0])
        all_results["red_teaming"][mode] = {"judge_category": judgment.get("categoria", "AMBIGUOUS")}
        print(f"  [{mode:>10s}] -> {judgment.get('categoria', '?')}")

    out_file = os.path.join(OUTPUT_DIR, f"{model_name}_v3.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {out_file}")
    del model
    return all_results

if __name__ == "__main__":
    if not os.path.exists("data/contrast_dataset.json"):
        print("\n[1/4] Building SNLI dataset...")
        build_contrast_dataset(n_pairs=N_CONTRAST_PAIRS)
    print("\n[2/4] Loading dataset...")
    dataset = load_contrast_dataset()

    print("\n[3/4] Verifying Ollama judge...")
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if "glm-5.2:cloud" not in [m["name"] for m in r.json().get("models", [])]:
            print("  WARNING: Run 'ollama pull glm-5.2:cloud' first!")
    except Exception:
        print("  WARNING: Ollama not running. Run 'ollama serve'")
        exit(1)

    print("\n[4/4] Running experiment...")
    all_experiments = {}
    for cfg in MODELS_CONFIG:
        all_experiments[cfg["id"]] = run_experiment_for_model(cfg, dataset)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for mid, res in all_experiments.items():
        cats = [v.get("judge_category", "?") for v in res.get("layer_sweep", {}).values()]
        total = len(cats) if cats else 1
        print(f"  {mid}\n    FAITHFUL: {cats.count('FAITHFUL_SHIFT')}/{total}\n    THEATER: {cats.count('REASONING_THEATER')}/{total}\n    Kappa: {res.get('kappa', 0):.3f}")
