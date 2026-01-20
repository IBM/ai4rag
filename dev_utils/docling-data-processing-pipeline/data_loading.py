from kfp import dsl
# from kfp import kubernetes
from kfp.dsl import Input, Output, Artifact


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["boto3"],
)
def load_benchmark(
    test_data_references: dict,
    sampled_docs_names: Output[Artifact],
    test_data_metadata: Output[Artifact],
):
    import os
    import json
    import boto3
    import logging
    import sys

    logger = logging.getLogger("load_benchmark logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    access_key = ""
    secret_key = ""
    if not access_key or not secret_key:
        raise RuntimeError("Missing ACCESS_KEY or SECRET_KEY")

    expected_keys = {"connection_id", "bucket", "path"}
    reference = {}
    for key in expected_keys:
        if value := test_data_references.get(key):
            reference[key] = value
        else:
            raise Exception(f"Test data reference is missing: {key}")

    logger.info("Sampling documents from bucket: %s", reference["bucket"])
    logger.info("Benchmark file key/path: %s", reference["path"])
    logger.info("sampled_docs_names artifact path: %s", sampled_docs_names.path)
    logger.info("test_data_metadata artifact path: %s", test_data_metadata.path)

    endpoint_url = reference["connection_id"]
    if not isinstance(endpoint_url, str) or not endpoint_url.startswith(("http://", "https://")):
        raise ValueError(
            f"Invalid S3 endpoint_url={endpoint_url!r}. "
            f"Expected something like 'https://...'. "
            f"Got test_data_references={test_data_references!r}"
        )

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name="us-south",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    os.makedirs(sampled_docs_names.path, exist_ok=True)
    os.makedirs(test_data_metadata.path, exist_ok=True)

    docs_list_file = os.path.join(sampled_docs_names.path, "sampled_docs_names.json")
    meta_file = os.path.join(test_data_metadata.path, "test_data_metadata.json")

    try:
        logger.info("Fetching benchmark file from s3://%s/%s", reference["bucket"], reference["path"])
        resp = s3.get_object(Bucket=reference["bucket"], Key=reference["path"])  # ✅ FIXED
        body = resp["Body"].read()
        benchmark = json.loads(body)
    except Exception as e:
        logger.error("Failed to load benchmark file %s: %s", reference["path"], e)
        raise RuntimeError(f"Failed to load benchmark file {reference['path']}: {e}")

    documents_names: list[str] = []
    for question in benchmark:
        documents_names.extend(question.get("correct_answer_documents_ids", []))

    docs_to_download = sorted(set(documents_names))
    logger.info("Sampled documents: %s", docs_to_download)

    with open(docs_list_file, "w", encoding="utf-8") as f:
        json.dump(docs_to_download, f, ensure_ascii=False, indent=2)
    logger.info("Saved sampled_docs_names list to %s", docs_list_file)

    meta = {
        "bucket": reference["bucket"],
        "benchmark_key": reference["path"],
        "endpoint_url": endpoint_url,
        "questions_total": len(benchmark) if isinstance(benchmark, list) else None,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Saved test_data_metadata to %s", meta_file)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["boto3"],
)
def fetch_sampled_docs(
    input_data_references: dict,
    sampled_docs_names: Input[Artifact],
    input_data_sources_documents: Output[Artifact],
    input_data_sources_metadata: Output[Artifact],
):
    import os
    import boto3
    import logging
    import sys
    import json
    from pathlib import Path

    logger = logging.getLogger("fetch_sampled_docs logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    access_key = ""
    secret_key = ""
    if not access_key or not secret_key:
        raise RuntimeError("Missing ACCESS_KEY or SECRET_KEY")

    supported_docs = {
        "pdf",
        "docx",
        "png", "jpg", "jpeg", "webp",
    }
    expected_keys = {"connection_id", "bucket", "path"}
    reference = {}
    for key in expected_keys:
        value = input_data_references.get(key)
        if isinstance(value, str):
            reference[key] = value
        else:
            raise Exception(f"Test data reference is missing: {key}")

    s3 = boto3.client(
        "s3",
        endpoint_url=reference["connection_id"],
        region_name="us-south",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    os.makedirs(input_data_sources_documents.path, exist_ok=True)
    os.makedirs(input_data_sources_metadata.path, exist_ok=True)

    def normalize_prefix(prefix: str) -> str:
        if not prefix:
            return ""
        prefix = prefix.lstrip("/")
        if not prefix.endswith("/"):
            prefix += "/"
        return prefix

    def ext_supported(key: str) -> bool:
        ext = Path(key).suffix.lower().lstrip(".")
        return ext in supported_docs

    def compute_prefix_stats(bucket: str, prefix: str):
        total_docs = 0
        total_bytes = 0

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if ext_supported(key):
                    total_docs += 1
                    total_bytes += int(obj.get("Size", 0))

        return total_docs, total_bytes

    bucket = reference["bucket"]
    prefix = normalize_prefix(reference["path"])

    logger.info("Computing stats for bucket=%s prefix=%s", bucket, prefix)
    total_documents, total_documents_size_bytes = compute_prefix_stats(bucket, prefix)

    metadata = {
        **reference,
        "total_documents": total_documents,
        "total_documents_size_bytes": total_documents_size_bytes,
    }

    meta_path = os.path.join(input_data_sources_metadata.path, "input_data_sources_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Wrote metadata: %s (docs=%d, bytes=%d)",
        meta_path,
        total_documents,
        total_documents_size_bytes,
    )

    docs_list_file = os.path.join(sampled_docs_names.path, "sampled_docs_names.json")
    logger.info("sampled_docs_names path: %s", sampled_docs_names.path)
    logger.info("Target directory: %s", input_data_sources_documents.path)

    if not os.path.exists(docs_list_file):
        logger.warning("Sampled docs list file not found: %s", docs_list_file)
        return

    with open(docs_list_file, "r", encoding="utf-8") as f:
        docs_to_download = json.load(f)

    logger.info("Documents to download (sample): %s", docs_to_download)

    if not docs_to_download:
        logger.info("No documents listed to download.")
        return

    for key in docs_to_download:
        if not ext_supported(key):
            logger.info("Skipping unsupported doc for download: %s", key)
            continue

        safe_name = key.replace("/", "__")
        local_path = os.path.join(input_data_sources_documents.path, safe_name)

        try:
            s3.download_file(bucket, key, local_path)
            logger.info("Fetched file: %s", local_path)
        except Exception as e:
            logger.error("Failed to fetch %s: %s", key, e)
            raise


@dsl.component(
    base_image="quay.io/wnowogorski-org/docling-data-loading-pipeline:latest",
)
def process_docs(
    downloaded_docs: Input[Artifact],
    result: Output[Artifact],
    sampled_data_metadata: Output[Artifact],
):
    import os
    import json
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import zipfile
    import logging
    import sys

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling_core.types.doc import ImageRefMode
    from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

    input_dir = Path(downloaded_docs.path)
    archive_path = Path(result.path)

    tmp_md_dir = archive_path.parent / "md_tmp"
    os.makedirs(tmp_md_dir, exist_ok=True)
    os.makedirs(sampled_data_metadata.path, exist_ok=True)

    logger = logging.getLogger("process_docs logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    logger.info("Input directory:   %s", input_dir)
    logger.info("Temp directory:    %s", tmp_md_dir)
    logger.info("Archive path:      %s", archive_path)
    logger.info("Sampled metadata:  %s", sampled_data_metadata.path)

    files = sorted(f for f in os.listdir(input_dir) if not f.startswith("."))
    logger.info("Files to process: %s", files)

    sampled_meta_path = Path(sampled_data_metadata.path) / "sampled_data_metadata.json"

    supported_pdfs = {".pdf"}
    supported_images = {".png", ".jpg", ".jpeg", ".webp"}

    pdf_cfg = {
        "extract_tables": True,
        "extract_images": True,
        "ocr_enabled": False,
        "timeout_sec": 300,
        "num_threads": 4,
        "accelerator_device": "auto",
        "image_export_mode": "embedded",
    }

    if not files:
        logger.info("No files found to process.")
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED):
            pass

        meta = {
            "languages": ["en"],
            "processed_documents_total": 0,
            "processed_documents_success": 0,
            "processed_documents_failed": 0,
            "extracted_text_size_bytes": 0,
            "average_document_length_chars": 0.0,
            "by_type": {"pdf": 0, "spreadsheets": 0, "images": 0, "other": 0},
        }
        sampled_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Wrote sampled metadata: %s", sampled_meta_path)
        return

    device_map = {
        "auto": AcceleratorDevice.AUTO,
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "gpu": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
    }
    device = device_map.get(str(pdf_cfg["accelerator_device"]).lower(), AcceleratorDevice.AUTO)

    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_table_structure = bool(pdf_cfg["extract_tables"])
    pdf_opts.generate_page_images = bool(pdf_cfg["extract_images"])
    pdf_opts.do_ocr = bool(pdf_cfg["ocr_enabled"])
    pdf_opts.document_timeout = float(pdf_cfg["timeout_sec"])

    pdf_opts.table_structure_options.do_cell_matching = True

    pdf_opts.accelerator_options = AcceleratorOptions(
        num_threads=int(pdf_cfg["num_threads"]),
        device=device,
    )

    pdf_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_opts,
                backend=DoclingParseV4DocumentBackend,
                pipeline_cls=StandardPdfPipeline,
            )
        }
    )

    def process_one(fname: str) -> dict:
        file_path = input_dir / fname
        suffix = file_path.suffix.lower()

        # ---------------- PDF ----------------
        if suffix in supported_pdfs:
            md_path = tmp_md_dir / f"{fname}.md"
            try:
                logger.info("PDF -> MD: %s -> %s", file_path, md_path)
                conv_res = pdf_converter.convert(str(file_path))
                md = conv_res.document.export_to_markdown(
                    image_mode=ImageRefMode(pdf_cfg["image_export_mode"])
                )
                md_sanitized = md.encode("utf-8", errors="replace").decode("utf-8")
                md_path.write_text(md_sanitized, encoding="utf-8")

                text_chars = len(md_sanitized)
                text_bytes = len(md_sanitized.encode("utf-8", errors="replace"))
                num_pages = getattr(conv_res.document, "page_count", 0)

                return {"file": fname, "type": "pdf", "ok": True, "chars": text_chars, "bytes": text_bytes,
                        "num_pages": num_pages}
            except Exception as e:
                logger.error("Error processing PDF %s: %s", file_path, e)
                err_txt = f"_Error processing file: {e}_\n"
                md_path.write_text(err_txt, encoding="utf-8")
                return {"file": fname, "type": "pdf", "ok": False, "chars": len(err_txt),
                        "bytes": len(err_txt.encode('utf-8', errors='replace'))}

        # ---------------- IMAGES (NEW) ----------------
        if suffix in supported_images:
            md_path = tmp_md_dir / f"{fname}.md"
            try:
                logger.info("IMAGE(OCR) -> MD: %s -> %s", file_path, md_path)

                from docling.datamodel.pipeline_options import RapidOcrOptions
                from docling.document_converter import ImageFormatOption
                from docling.datamodel.base_models import ConversionStatus

                models_dir = os.environ.get("RAPIDOCR_MODELS_DIR", "").strip()
                download_root = None

                if not models_dir:
                    try:
                        from huggingface_hub import snapshot_download
                        logger.info("RAPIDOCR_MODELS_DIR not set; downloading RapidOCR models from HuggingFace...")
                        download_root = snapshot_download(repo_id="SWHL/RapidOCR")
                        models_dir = download_root
                    except Exception as e:
                        raise RuntimeError(
                            "RapidOCR models not available. "
                            "Set RAPIDOCR_MODELS_DIR to a directory containing RapidOCR model files, "
                            f"or ensure huggingface_hub + network access are available. Details: {e}"
                        )

                det_model_path = os.path.join(models_dir, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx")
                rec_model_path = os.path.join(models_dir, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx")
                cls_model_path = os.path.join(models_dir, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx")

                for p in (det_model_path, rec_model_path, cls_model_path):
                    if not os.path.exists(p):
                        raise FileNotFoundError(f"RapidOCR model not found: {p}")

                image_format_options = ImageFormatOption(
                    do_ocr=True,
                    ocr_options=RapidOcrOptions(
                        det_model_path=det_model_path,
                        rec_model_path=rec_model_path,
                        cls_model_path=cls_model_path,
                    ),
                )
                image_converter = DocumentConverter(
                    format_options={InputFormat.IMAGE: image_format_options}
                )

                conv_res = image_converter.convert(str(file_path))

                status = getattr(conv_res, "status", None)
                if status is not None and status != ConversionStatus.SUCCESS:
                    raise RuntimeError(f"Docling conversion status: {status}")

                md = conv_res.document.export_to_markdown()
                md_sanitized = md.encode("utf-8", errors="replace").decode("utf-8")
                md_path.write_text(md_sanitized, encoding="utf-8")

                text_chars = len(md_sanitized)
                text_bytes = len(md_sanitized.encode("utf-8", errors="replace"))

                return {"file": fname, "type": "images", "ok": True, "chars": text_chars, "bytes": text_bytes}

            except Exception as e:
                logger.error("Error processing IMAGE %s: %s", file_path, e)
                err_txt = f"_Error processing image: {e}_\n"
                md_path.write_text(err_txt, encoding="utf-8")
                return {"file": fname, "type": "images", "ok": False, "chars": len(err_txt),
                        "bytes": len(err_txt.encode("utf-8", errors="replace"))}

        return {"file": fname, "type": "other", "ok": False, "chars": 0, "bytes": 0, "skipped": True}

    max_workers = min(4, len(files))
    logger.info("Using max_workers=%d for ThreadPoolExecutor", max_workers)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_one, f): f for f in files}
        for future in as_completed(future_to_file):
            fname = future_to_file[future]
            try:
                results.append(future.result())
            except Exception as e:
                logger.error("Unexpected error for %s: %s", fname, e)
                results.append({"file": fname, "type": "other", "ok": False, "chars": 0, "bytes": 0})

    logger.info("Creating ZIP archive with markdown files...")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for md_file in sorted(tmp_md_dir.iterdir()):
            if md_file.is_file():
                zf.write(md_file, arcname=md_file.name)

    logger.info("Markdown archive created at: %s", archive_path)

    extracted_text_size_bytes = sum(r.get("bytes", 0) for r in results)
    ok_docs = [r for r in results if r.get("ok")]
    average_document_length_chars = (sum(r.get("chars", 0) for r in ok_docs) / len(ok_docs)) if ok_docs else 0.0

    by_type = {"pdf": 0, "spreadsheets": 0, "images": 0, "other": 0}
    for r in results:
        t = r.get("type", "other")
        by_type[t] = by_type.get(t, 0) + 1

    meta = {
        "languages": ["en"],
        "processed_documents_total": len(results),
        "processed_documents_success": len(ok_docs),
        "processed_documents_failed": len(results) - len(ok_docs),
        "extracted_text_size_bytes": int(extracted_text_size_bytes),
        "average_document_length_chars": float(average_document_length_chars),
        "by_type": by_type,
        "pdf_config": pdf_cfg,
    }

    sampled_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote sampled metadata: %s", sampled_meta_path)


@dsl.pipeline(
    name="s3-bucket-docling-parallel",
    description=(
        "Loads benchmark, selects documents, downloads them from S3, "
        "and processes each with Docling in parallel."
    ),
)
def s3_read_demo_pipeline(
    input_data_references: dict,
    test_data_references: dict,
):
    select_step = load_benchmark(
        test_data_references=test_data_references
    )
    select_step.set_caching_options(False)

    download_step = fetch_sampled_docs(
        input_data_references=input_data_references,
        sampled_docs_names=select_step.outputs["sampled_docs_names"],
    )
    download_step.set_caching_options(False)

    # Example: if you use k8s Secret to inject S3 creds, uncomment and adjust:
    # kubernetes.use_secret_as_env(
    #     select_step,
    #     secret_name="storage-secrets",
    #     secret_key_to_env={
    #         "access_key_id": "S3_ACCESS_KEY_ID",
    #         "secret_access_key": "S3_SECRET_ACCESS_KEY",
    #     },
    # )
    # kubernetes.use_secret_as_env(
    #     download_step,
    #     secret_name="storage-secrets",
    #     secret_key_to_env={
    #         "access_key_id": "S3_ACCESS_KEY_ID",
    #         "secret_access_key": "S3_SECRET_ACCESS_KEY",
    #     },
    # )

    process_docs(
        downloaded_docs=download_step.outputs["input_data_sources_documents"]
    )


if __name__ == "__main__":
    import pathlib
    from kfp import compiler

    here = pathlib.Path(__file__).resolve()
    out_path = here.with_name("data_loading.yaml")

    compiler.Compiler().compile(
        pipeline_func=s3_read_demo_pipeline,
        package_path=str(out_path),
    )