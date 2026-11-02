"""Download exactly the model revisions in the study manifest."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("models", nargs="*", default=["distill-1.5b", "distill-7b", "distill-32b", "qwq-32b"])
args = parser.parse_args()
models = json.loads((root / "configs/models.json").read_text())
for name in args.models:
    cfg = models[name]
    print(f"Downloading {name}: {cfg['id']} @ {cfg['revision']}", flush=True)
    path = snapshot_download(cfg["id"], revision=cfg["revision"],
                             cache_dir=str(root / "hf_cache/hub"), max_workers=4,
                             allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "README.md", "LICENSE"])
    print(f"Ready: {path}", flush=True)
