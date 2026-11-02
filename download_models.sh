#!/usr/bin/env bash
set -euo pipefail
# The experiment requires Transformers weights, not separate Ollama copies.
exec env/bin/python scripts/download_models.py "$@"
