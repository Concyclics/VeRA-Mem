"""Pin and download only the public model's runtime artifacts; no credentials logged."""
import argparse
import importlib.metadata
import json
import pathlib
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--workspace", type=pathlib.Path, required=True)
p.add_argument("--revision", default=None)
a = p.parse_args()
w = a.workspace.resolve()
(w / "runs").mkdir(parents=True, exist_ok=True)
model_id = "Qwen/Qwen3-4B-Instruct-2507"
info = HfApi().model_info(model_id, revision=a.revision)
revision = info.sha
dest = w / "models" / "Qwen3-4B-Instruct-2507"
if shutil.disk_usage(w).free < 13 * 1024**3:
    raise RuntimeError("At least 13 GiB free required before model download; no files removed.")
print(json.dumps({"model": model_id, "revision": revision, "destination": str(dest)}), flush=True)
path = snapshot_download(model_id, revision=revision, local_dir=dest,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE*"], max_workers=2)
versions = {}
for name in ["torch", "transformers", "huggingface_hub", "numpy", "safetensors", "tokenizers", "datasets"]:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
manifest = {"model_id": model_id, "revision": revision, "path": path,
            "python": sys.version, "packages": versions}
(w / "models" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("MODEL_READY", flush=True)
