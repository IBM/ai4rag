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

RUN chmod 1777 /tmp

ENV DOCLING_ARTIFACTS_PATH=/tmp/docling_artifacts
ENV RAPIDOCR_MODELS_DIR=/tmp/RapidOCR