FROM python:3.11

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "docling[ort]" \
    docling-ibm-models \
    "kfp==2.14.6"

RUN python - << 'PY'
from pathlib import Path
from docling.utils.model_downloader import download_models

models_dir = Path("/tmp/docling_artifacts")
models_dir.mkdir(parents=True, exist_ok=True)

print("Downloading Docling models into:", models_dir)
download_models(output_dir=models_dir)

found = list(models_dir.rglob("model.safetensors"))
print("Found model.safetensors files:")
for p in found:
    print(" -", p)
PY

ENV DOCLING_ARTIFACTS_PATH=/tmp/docling_artifacts
