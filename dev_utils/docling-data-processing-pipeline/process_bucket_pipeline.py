from kfp import dsl
# from kfp import kubernetes
from kfp.dsl import Input, Output, Artifact


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["boto3"],
)
def load_benchmark(
    bucket: str,
    endpoint: str,
    region: str,
    benchmark_key: str,
    sampled_docs_names: Output[Artifact],
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

    access_key = os.environ.get("ACCESS_KEY")
    secret_key = os.environ.get("SECRET_KEY")

    logger.info("Sampling documents from bucket: %s", bucket)
    logger.info("Benchmark file: %s", benchmark_key)
    logger.info("sampled_docs_names artifact path: %s", sampled_docs_names.path)

    if not access_key or not secret_key:
        raise RuntimeError("Missing ACCESS_KEY or SECRET_KEY")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    os.makedirs(sampled_docs_names.path, exist_ok=True)
    docs_list_file = os.path.join(sampled_docs_names.path, "sampled_docs_names.json")

    try:
        logger.info("Fetching benchmark file: %s", benchmark_key)
        resp = s3.get_object(Bucket=bucket, Key=benchmark_key)
        body = resp["Body"].read()
        benchmark = json.loads(body)
    except Exception as e:
        logger.error("Failed to load benchmark file %s: %s", benchmark_key, e)
        raise RuntimeError(f"Failed to load benchmark file {benchmark_key}: {e}")

    documents_names: list[str] = []
    for question in benchmark:
        documents_names.extend(question["correct_answer_documents_ids"])
    docs_to_download = list(set(documents_names))

    logger.info("Sampled documents: %s", docs_to_download)

    if not docs_to_download:
        logger.warning("No documents to download according to benchmark.")
        with open(docs_list_file, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return

    with open(docs_list_file, "w", encoding="utf-8") as f:
        json.dump(docs_to_download, f, ensure_ascii=False, indent=2)

    logger.info("Saved sampled_docs_names list to %s", docs_list_file)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["boto3"],
)
def fetch_sampled_docs(
    bucket: str,
    endpoint: str,
    region: str,
    sampled_docs_names: Input[Artifact],
    downloaded_docs: Output[Artifact],
):
    import os
    import boto3
    import logging
    import sys
    import json

    logger = logging.getLogger("fetch_sampled_docs logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    access_key = os.environ.get("ACCESS_KEY")
    secret_key = os.environ.get("SECRET_KEY")

    logger.info("Downloading sampled documents from bucket: %s", bucket)
    logger.info("sampled_docs_names path: %s", sampled_docs_names.path)
    logger.info("Target directory: %s", downloaded_docs.path)

    if not access_key or not secret_key:
        raise RuntimeError("Missing ACCESS_KEY or SECRET_KEY")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    os.makedirs(downloaded_docs.path, exist_ok=True)
    docs_list_file = os.path.join(sampled_docs_names.path, "sampled_docs_names.json")

    if not os.path.exists(docs_list_file):
        logger.warning("Sampled docs list file not found: %s", docs_list_file)
        return

    with open(docs_list_file, "r", encoding="utf-8") as f:
        docs_to_download = json.load(f)  # list[str]

    logger.info("Documents to download: %s", docs_to_download)

    if not docs_to_download:
        logger.info("No documents listed to download.")
        return

    keys: list[str] = []

    for key in docs_to_download:
        keys.append(key)
        safe_name = key.replace("/", "__")
        local_path = os.path.join(downloaded_docs.path, safe_name)

        try:
            s3.download_file(bucket, key, local_path)
            logger.info("Fetched file: %s", local_path)
        except Exception as e:
            logger.error("Failed to fetch %s: %s", key, e)
            raise


@dsl.component(
    base_image="quay.io/openshift_trial/custom_images:docling",
)
def process_docs(
    downloaded_docs: Input[Artifact],
    result: Output[Artifact],
):
    import os
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import zipfile
    import logging

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    input_dir = Path(downloaded_docs.path)
    archive_path = Path(result.path)

    tmp_md_dir = archive_path.parent / "md_tmp"
    os.makedirs(tmp_md_dir, exist_ok=True)

    logger = logging.getLogger("fetch_sampled_docs logger")
    logger.setLevel(logging.INFO)

    logger.info("Input directory:   %s", input_dir)
    logger.info("Temp directory: %s", tmp_md_dir)
    logger.info("Archive path:      %s", archive_path)

    files = sorted(
        f for f in os.listdir(input_dir)
        if not f.startswith(".")
    )
    logger.info("Files to process: %s", files)

    if not files:
        logger.info("No files found to process.")
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED):
            pass
        return

    def process_one(fname: str) -> str:
        file_path = input_dir / fname
        md_path = tmp_md_dir / f"{fname}.md"

        logger.info("Converting %s -> %s", file_path, md_path)

        pdf_opts = PdfPipelineOptions(do_ocr=False)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
            }
        )

        try:
            result = converter.convert(file_path)
            md = result.document.export_to_markdown()
            md_sanitized = md.encode("utf-8", errors="replace").decode("utf-8")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_sanitized)

            logger.info("Done: %s", file_path)
            return fname
        except Exception as e:
            error_msg = f"Error processing %s: %s" % (file_path, e)
            logger.error("%s", error_msg)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"_Error processing file: {e}_\n")
            return fname + " (error)"

    max_workers = min(4, len(files))
    logger.info("Using max_workers=%d for ThreadPoolExecutor", max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_one, f): f for f in files}
        for future in as_completed(future_to_file):
            fname = future_to_file[future]
            try:
                _ = future.result()
            except Exception as e:
                logger.error("Unexpected error for %s: %s", fname, e)

    logger.info("Docling processing finished.")
    logger.info("Creating ZIP archive with markdown files...")

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for md_file in sorted(tmp_md_dir.iterdir()):
            if md_file.is_file():
                zf.write(md_file, arcname=md_file.name)
                logger.info("Added to archive: %s as %s", md_file, md_file.name)

    logger.info("Markdown archive created at: %s", archive_path)


@dsl.pipeline(
    name="s3-bucket-docling-parallel",
    description=(
        "Loads benchmark, selects documents, downloads them from S3, "
        "and processes each with Docling in parallel."
    ),
)
def s3_read_demo_pipeline(
    bucket: str = "wnowogorski-test-bucket",
    endpoint: str = "https://s3.us-south.cloud-object-storage.appdomain.cloud",
    region: str = "us-south",
    benchmark_key: str = "benchmark.json",
):
    select_step = load_benchmark(
        bucket=bucket,
        endpoint=endpoint,
        region=region,
        benchmark_key=benchmark_key,
    )

    download_step = fetch_sampled_docs(
        bucket=bucket,
        endpoint=endpoint,
        region=region,
        sampled_docs_names=select_step.outputs["sampled_docs_names"],
    )

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
        downloaded_docs=download_step.outputs["downloaded_docs"]
    )


if __name__ == "__main__":
    import pathlib
    from kfp import compiler

    here = pathlib.Path(__file__).resolve()
    out_path = here.with_name("process_bucket_pipeline.yaml")

    compiler.Compiler().compile(
        pipeline_func=s3_read_demo_pipeline,
        package_path=str(out_path),
    )