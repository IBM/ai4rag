FROM python:3.11

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      poppler-utils \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      "docling[ort]" \
      docling-ibm-models \
      "kfp==2.14.6" \
      "pandas>=2.3.0" \
      "openpyxl>=3.1.5" \
      "xlrd>=2.0.1" \
      "odfpy>=1.4.1" \
      "huggingface_hub>=0.23.0"

RUN python - << 'PY'
from pathlib import Path

from docling.utils.model_downloader import download_models

docling_dir = Path("/opt/models/docling")
docling_dir.mkdir(parents=True, exist_ok=True)
print("Downloading Docling models into:", docling_dir)
download_models(output_dir=docling_dir)

from huggingface_hub import snapshot_download

rapidocr_dir = Path("/opt/models/RapidOCR")
rapidocr_dir.mkdir(parents=True, exist_ok=True)

print("Downloading RapidOCR models into:", rapidocr_dir)
snapshot_download(
    repo_id="SWHL/RapidOCR",
    local_dir=str(rapidocr_dir),
    local_dir_use_symlinks=False,
)

det = rapidocr_dir / "PP-OCRv4" / "en_PP-OCRv3_det_infer.onnx"
rec = rapidocr_dir / "PP-OCRv4" / "ch_PP-OCRv4_rec_server_infer.onnx"
cls = rapidocr_dir / "PP-OCRv3" / "ch_ppocr_mobile_v2.0_cls_train.onnx"

print("RapidOCR expected files:")
print(" -", det, det.exists())
print(" -", rec, rec.exists())
print(" -", cls, cls.exists())

if not (det.exists() and rec.exists() and cls.exists()):
    raise SystemExit("RapidOCR model files missing after download; check repo layout/version.")
PY

RUN chgrp -R 0 /opt/models && \
    chmod -R g=u /opt/models && \
    chmod 1777 /tmp

ENV DOCLING_ARTIFACTS_PATH=/opt/models/docling
ENV RAPIDOCR_MODELS_DIR=/opt/models/RapidOCR
