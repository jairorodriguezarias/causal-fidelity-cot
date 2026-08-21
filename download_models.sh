#!/usr/bin/env bash
set -euo pipefail

echo "=== Downloading models for the experiment ==="

# --- Judge (GLM-5.2:cloud via Ollama) ---
echo "1/3 Downloading GLM-5.2:cloud (judge)..."
ollama pull glm-5.2:cloud

# --- Distilled model (SFT) ---
echo "2/3 Downloading DeepSeek-R1-Distill-Qwen-7B..."
ollama pull deepseek-r1:7b

# --- RL model (for comparison, requires large GPU) ---
echo "3/3 Downloading QwQ-32B..."
ollama pull qwq:32b

echo "=== Download complete ==="
ollama list
