# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from typing import Literal

from ai4rag.assets_generator.components import NotebookCell, AssetGenerationError


def get_notebook_template(
    template: Literal[
        "ls_inference",
        "ls_indexing",
        "chroma",
    ],
) -> dict[str, NotebookCell]:
    banner = (
        "<img src='"
        "data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0id"
        "XRmLTgiPz4KPHN2ZyB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9y"
        "Zy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGx"
        "pbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAxNzk2IDEwMCIgc3R5bG"
        "U9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgMTc5NiAxMDA7IiB4bWw6c3BhY2U9I"
        "nByZXNlcnZlIj4KPHN0eWxlIHR5cGU9InRleHQvY3NzIj4KCS5zdDB7ZmlsbC1ydWxl"
        "OmV2ZW5vZGQ7Y2xpcC1ydWxlOmV2ZW5vZGQ7ZmlsbDp1cmwoI1NWR0lEXzFfKTt9Cgk"
        "uc3Qxe2ZpbGw6bm9uZTtzdHJva2U6I0ZGRkZGRjtzdHJva2Utd2lkdGg6MjtzdHJva2"
        "UtbWl0ZXJsaW1pdDoxMDt9Cgkuc3Qye2ZpbGw6bm9uZTtzdHJva2U6I0ZGRkZGRjtzd"
        "HJva2Utd2lkdGg6MS41O3N0cm9rZS1taXRlcmxpbWl0OjEwO30KCS5zdDN7ZmlsbDoj"
        "RkZGRkZGO30KCS5zdDR7Zm9udC1mYW1pbHk6J0hlbHZldGljYSBOZXVlJywgQXJpYWw"
        "sIHNhbnMtc2VyaWY7fQoJLnN0NXtmb250LXNpemU6MzJweDt9Cgkuc3Q2e2ZpbGw6Iz"
        "NEM0QzRDt9Cgkuc3Q3e2ZpbGw6IzkzOTU5ODt9Cgkuc3Q4e29wYWNpdHk6MC4yO2Zpb"
        "Gw6dXJsKCNTVkdJRF8yXyk7ZW5hYmxlLWJhY2tncm91bmQ6bmV3O30KCS5zdDl7Zm9u"
        "dC13ZWlnaHQ6NTAwO30KPC9zdHlsZT4KPHJlY3Qgd2lkdGg9IjE3OTYiIGhlaWdodD0"
        "iMTAwIiBmaWxsPSIjMTYxNjE2Ii8+CjxsaW5lYXJHcmFkaWVudCBpZD0iU1ZHSURfMV"
        "8iIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4MT0iNDIuODYiIHkxPSI1M"
        "CIgeDI9Ijc5LjcxIiB5Mj0iNTAiPgoJPHN0b3Agb2Zmc2V0PSIwIiBzdHlsZT0ic3Rv"
        "cC1jb2xvcjojRkY2QjZCIi8+Cgk8c3RvcCBvZmZzZXQ9IjAuMjEiIHN0eWxlPSJzdG9"
        "wLWNvbG9yOiNFRTAwMDAiLz4KCTxzdG9wIG9mZnNldD0iMC43NSIgc3R5bGU9InN0b3"
        "AtY29sb3I6I0NDMDAwMCIvPgoJPHN0b3Agb2Zmc2V0PSIxIiBzdHlsZT0ic3RvcC1jb"
        "2xvcjojQUEwMDAwIi8+CjwvbGluZWFyR3JhZGllbnQ+CjwhLS0gQXV0b1JBRyBJY29u"
        "L0xvZ28gcGxhY2Vob2xkZXIgLSBzaW1wbGlmaWVkIGdlb21ldHJpYyBzaGFwZSAtLT4"
        "KPHBhdGggY2xhc3M9InN0MCIgZD0iTTUyLjQsNDUuOWMwLTIuMywxLjgtNC4xLDQuMS"
        "00LjFzNC4xLDEuOCw0LjEsNC4xUzU4LjgsNTAsNTYuNSw1MGwwLDBjLTIuMiwwLjEtN"
        "C0xLjctNC4xLTMuOQoJQzUyLjQsNDYsNTIuNCw0Niw1Mi40LDQ1Ljl6IE03Ny41LDUy"
        "LjVjLTAuOC0xLjEtMS40LTIuMy0xLjktMy41YzEuMi00LjUsMC43LTguNi0xLjgtMTE"
        "uOWMtMi45LTMuOC04LjItNi0xNC41LTYuMQoJYy00LjUtMC4xLTguOCwxLjctMTIsNC"
        "44Yy0zLDMtNC42LDcuMi00LjUsMTEuNWMtMC4xLDIuOSwwLjksNS44LDIuNyw4LjFjM"
        "C44LDAuOCwxLjMsMS45LDEuNCwzdjQuNWMtMC44LDAuNS0xLjQsMS4zLTEuNCwyLjMK"
        "CWMwLjIsMS41LDEuNSwyLjYsMywyLjRjMS4yLTAuMiwyLjItMS4xLDIuNC0yLjRjMC0"
        "xLTAuNS0xLjktMS40LTIuM3YtNC41YzAtMi0xLTMuMy0xLjktNC42Yy0xLjUtMS45LT"
        "IuMi00LjItMi4xLTYuNQoJYzAtMy41LDEuNC02LjksMy44LTkuNGMyLjctMi43LDYuM"
        "y00LjEsMTAtNC4xYzUuNSwwLDkuOCwxLjksMTIuMSw1YzIsMi44LDIuNSw2LjMsMS40"
        "LDkuNmMtMC40LDEuMiwwLjYsMi43LDIuMyw1LjYKCWMwLjYsMC45LDEuMiwxLjksMS4"
        "2LDIuOWMtMC45LDAuNy0yLDEuMi0zLjEsMS41Yy0wLjUsMC40LTAuNywwLjktMC44LD"
        "EuNVY2NWMwLDAuNC0wLjEsMC44LTAuNCwxLjFjLTAuMywwLjItMC43LDAuMy0xLjEsM"
        "C4zCgljLTEuNi0wLjMtMy40LTAuNy01LjItMS4xdi00LjhjMC44LTAuNSwxLjQtMS40"
        "LDEuNC0yLjNjMC0xLjUtMS4yLTIuNy0yLjctMi43cy0yLjcsMS4yLTIuNywyLjdjMCw"
        "xLDAuNSwxLjksMS40LDIuM3Y0LjEKCWMtMC40LTAuMS0wLjctMC4xLTEuMS0wLjNjLT"
        "QuNS0xLjEtNC41LTIuNi00LjUtMy40di04LjNjMy4yLTAuNyw1LjQtMy41LDUuNS02L"
        "jdjLTAuMS0zLjgtMy4zLTYuNy03LjEtNi42Yy0zLjYsMC4xLTYuNCwzLTYuNiw2LjYK"
        "CWMwLDMuMiwyLjMsNiw1LjUsNi43djguM2MwLDIsMC43LDQuNiw2LjYsNi4xYzMsMC4"
        "4LDYsMS41LDkuMSwxLjljMC4zLDAsMC42LDAuMSwwLjgsMC4xYzEsMCwxLjktMC4zLD"
        "IuNi0xCgljMC45LTAuOCwxLjQtMS45LDEuNC0zLjF2LTQuNWMyLTAuOCw0LjEtMiw0L"
        "jEtMy43Qzc5LjcsNTUuOSw3OSw1NC42LDc3LjUsNTIuNXoiLz4KPGNpcmNsZSBjbGFz"
        "cz0ic3QxIiBjeD0iNTYuNSIgY3k9IjQ1LjkiIHI9IjUuNCIvPgo8Y2lyY2xlIGNsYXN"
        "zPSJzdDIiIGN4PSI0OC4zIiBjeT0iNjUiIHI9IjEuNiIvPgo8Y2lyY2xlIGNsYXNzPS"
        "JzdDIiIGN4PSI2NC44IiBjeT0iNTguMiIgcj0iMS42Ii8+Cjx0ZXh0IHRyYW5zZm9yb"
        "T0ibWF0cml4KDEgMCAwIDEgMTAxLjAyIDU5LjMzKSIgY2xhc3M9InN0MyBzdDQgc3Q1"
        "Ij5BdXRvUkFHPC90ZXh0Pgo8cmVjdCB4PSIyNDIiIHk9IjM0IiBjbGFzcz0ic3Q2IiB"
        "3aWR0aD0iMSIgaGVpZ2h0PSIzMiIvPgo8dGV4dCB0cmFuc2Zvcm09Im1hdHJpeCgxID"
        "AgMCAxIDI1Ni4yOSA1OS42NikiIGNsYXNzPSJzdDcgc3Q0IHN0NSI+UGFydCBvZiBSZ"
        "WQgSGF0IE9wZW5TaGlmdCBBSTwvdGV4dD4KPGxpbmVhckdyYWRpZW50IGlkPSJTVkdJR"
        "F8yXyIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiIHgxPSI3NzMuOCIgeTE9"
        "IjUwIiB4Mj0iMTc5NiIgeTI9IjUwIj4KCTxzdG9wIG9mZnNldD0iMCIgc3R5bGU9InN"
        "0b3AtY29sb3I6IzE2MTYxNiIvPgoJPHN0b3Agb2Zmc2V0PSIwLjUyIiBzdHlsZT0ic3"
        "RvcC1jb2xvcjojRkY2QjZCIi8+Cgk8c3RvcCBvZmZzZXQ9IjAuNjIiIHN0eWxlPSJzd"
        "G9wLWNvbG9yOiNFRTAwMDAiLz4KCTxzdG9wIG9mZnNldD0iMC44OCIgc3R5bGU9InN0"
        "b3AtY29sb3I6I0NDMDAwMCIvPgoJPHN0b3Agb2Zmc2V0PSIxIiBzdHlsZT0ic3RvcC1"
        "jb2xvcjojQUEwMDAwIi8+CjwvbGluZWFyR3JhZGllbnQ+CjxyZWN0IHg9Ijc3My44Ii"
        "BjbGFzcz0ic3Q4IiB3aWR0aD0iMTAyMi4yIiBoZWlnaHQ9IjEwMCIvPgo8dGV4dCB0c"
        "mFuc2Zvcm09Im1hdHJpeCgxIDAgMCAxIDE0MjggNTkuNDYpIiBjbGFzcz0ic3QzIHN0"
        "NCBzdDUgc3Q5Ij5SQUcgUGF0dGVybiBOb3RlYm9vazwvdGV4dD4KPC9zdmc+Cg==' />"
    )

    ls_indexing_notebook_template: dict[
        str,
        NotebookCell,
    ] = {
        "BANNER": NotebookCell(
            cell_type="markdown",
            source=banner,
        ),
        "TABLE_OF_CONTENTS": NotebookCell(
            cell_type="markdown",
            source=[
                "## Pattern {PATTERN_NAME} Index Building Content\n",
                "\n",
                "This notebook demonstrates how to process documents and build a vector store index for RAG applications. It covers document discovery, text extraction, chunking, and uploading embeddings to a vector database using Llama Stack.\n",
                "\n",
                "### &#x1F4CB; Contents \n",
                "This notebook contains the following sections:\n",
                "\n",
                "- **[Setup](#Setup)**\n",
                "  - [Install packages](#Install-packages)\n",
                "  - [Import required libraries](#Import-required-libraries)\n",
                "  - [Configure S3 credentials](#Configure-S3-credentials)\n",
                "  - [Prepare S3 client](#Prepare-S3-client)\n",
                "- **[Process input documents](#Process-input-documents)**\n",
                "  - [Documents discovery](#Documents-discovery)\n",
                "  - [Text extraction](#Text-extraction)\n",
                "- **[Upload documents content into vector store database](#Upload-documents-content-into-vector-store-database)**\n",
                "  - [Prepare Llama Stack Client](#Prepare-Llama-Stack-Client)\n",
                "  - [Prepare chunker](#Prepare-chunker)\n",
                "  - [Initialize vector store](#Initialize-vector-store)\n",
                "  - [Upload chunks to vector store](#Upload-chunks-to-vector-store)\n",
                "  - [Retrieve chunks for sample question](#Retrieve-chunks-for-sample-question)\n",
                "- **[Summary](#Summary)**",
            ],
        ),
        "CHAPTER_1": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Setup\n",
                "\n",
                "This section sets up the notebook environment by installing required packages, importing libraries, and configuring access to S3 storage.\n",
                "\n",
                "### Install packages\n",
                "\n",
                "Install all required Python packages for document processing and RAG operations:\n",
                "- **boto3**: AWS SDK for Python to interact with S3 storage\n",
                "- **pipelines-components**: Red Hat's pipeline components for data processing\n",
                "- **docling**: Document processing and text extraction library\n",
                "- **ai4rag**: The AutoRAG framework for building RAG applications",
            ],
        ),
        "DEPENDENCIES": NotebookCell(
            cell_type="code",
            source=[
                "!pip install boto3 | tail -n 1\n",
                "!pip install -U --no-cache-dir git+https://github.com/LukaszCmielowski/pipelines-components.git@rhoai_autorag | tail -n 1\n",
                "!pip install docling | tail -n 1\n",
                "!pip install 'ai4rag' | tail -n 1",
            ],
        ),
        "MD_1_1": NotebookCell(
            cell_type="markdown",
            source="### Import required libraries\n\nImport all necessary Python modules and configure logging to suppress verbose output from component loggers.",
        ),
        "MAIN_IMPORTS": NotebookCell(
            cell_type="code",
            source=[
                "import os\n",
                "import json\n",
                "import logging\n",
                "from pathlib import Path\n",
                "from types import SimpleNamespace\n",
                "import getpass\n",
                "\n",
                "import warnings\n",
                'warnings.filterwarnings("ignore")\n',
                "\n",
                "import boto3\n",
                "from langchain_core.documents import Document\n",
                "\n",
                "for logger_name in (\n",
                '        "httpx",\n',
                '        "Document Loader component logger",\n',
                '        "Text Extraction component logger",\n',
                "):\n",
                "    logging.getLogger(logger_name).propagate = False",
            ],
        ),
        "MD_1_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Configure S3 credentials\n",
                "\n",
                "To load documents from S3-compatible object storage, you need to provide credentials. If you're using OpenShift AI, these can be configured as data connections.\n",
                "\n",
                "&#x1F4CC; **Action**: Provide the credentials for your S3 instance if they are not already set in the notebook environment.\n",
                "\n",
                "&#x1F4A1; **Tip**: In the project, open **Connections** and add an **S3 compatible object storage connection** to a bucket you will use for documents and test data. Open **Workbenches**, edit your workbench, and attach the S3 connection you created so the notebook can read from the bucket. Save and restart the workbench if prompted.",
            ],
        ),
        "AWS_ENV": NotebookCell(
            cell_type="code",
            source=[
                'required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT", "AWS_DEFAULT_REGION", "AWS_S3_BUCKET"]\n',
                "missing = [var for var in required_vars if not os.environ.get(var)]\n",
                "if missing:\n",
                '    raise ValueError(f"Missing required environment variables: {{missing}}")',
            ],
        ),
        "MD_1_3": NotebookCell(
            cell_type="markdown",
            source="### Prepare S3 client\n\nCreates an S3 client session using the provided credentials. This client will be used to discover and download documents from the specified S3 bucket.",
        ),
        "S3_CLIENT": NotebookCell(
            cell_type="code",
            source=[
                "session = boto3.session.Session(\n",
                '    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],\n',
                '    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],\n',
                ")\n",
                "s3_client = session.client(\n",
                "    service_name='s3',\n",
                '    endpoint_url=os.environ["AWS_S3_ENDPOINT"],\n',
                ")",
            ],
        ),
        "CHAPTER_2": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Process input documents\n",
                "\n",
                "This section handles document discovery and text extraction. Documents are first discovered in S3 storage, then their content is extracted and converted to markdown format for further processing.",
            ],
        ),
        "MD_2_1": NotebookCell(
            cell_type="markdown",
            source=[
                "The data processing pipeline prepares documents for the RAG system in multiple steps. Each step runs as a standalone component with outputs stored under `step_outputs/`. \n",
                "\n",
                "| Step | Component | Purpose |\n",
                "|------|-----------|---------|\n",
                "| 1 | **Documents discovery** | List documents in the bucket, prioritize benchmark-referenced docs, apply a size cap, and write a JSON manifest (no content download). |\n",
                "| 2 | **Text extraction** | Download the listed documents from S3 and extract text to Markdown using Docling. |",
            ],
        ),
        "LOAD_DATA": NotebookCell(
            cell_type="code",
            source=[
                "from kfp_components.components.data_processing.autorag.documents_discovery.component import documents_discovery\n",
                "from kfp_components.components.data_processing.autorag.text_extraction.component import text_extraction\n",
                "\n",
                'step_output_dir = Path("./step_outputs")\n',
                "input_data_bucket_name = os.environ['AWS_S3_BUCKET']\n",
                'input_data_key = "{INPUT_DATA_KEY}"\n',
                "step_output_dir.mkdir(parents=True, exist_ok=True)",
            ],
        ),
        "MD_2_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Documents discovery\n",
                "\n",
                "Lists objects in the S3 input bucket, filters by supported extensions (e.g., `.pdf`, `.docx`, `.pptx`, `.md`, `.html`, `.txt`), and builds a document set. Documents referenced in the benchmark are prioritized, then others are added until a configurable size limit (1 GB by default) is reached. This step does not download document contents but writes a JSON manifest (`documents_descriptor.json`) containing the bucket, prefix, and list of selected object keys and sizes for the next step.",
            ],
        ),
        "DOCUMENTS_DISCOVERY": NotebookCell(
            cell_type="code",
            source=[
                "\n",
                'discovered_documents_out = SimpleNamespace(path=str(step_output_dir / "discovered_documents"))\n',
                "\n",
                "documents_discovery.python_func(\n",
                "    input_data_bucket_name=input_data_bucket_name,\n",
                "    input_data_path=input_data_key,\n",
                "    discovered_documents=discovered_documents_out,\n)\n",
                "\n",
                'descriptor_path = step_output_dir / "discovered_documents" / "documents_descriptor.json"\n',
                "with open(descriptor_path) as f:\n",
                "    descriptor = json.load(f)\n",
                "\n",
                "print(json.dumps(descriptor, indent=4, ensure_ascii=False))",
            ],
        ),
        "MD_2_3": NotebookCell(
            cell_type="markdown",
            source=[
                "### Text extraction\n",
                "\n",
                "Reads the `documents_descriptor.json` produced by the discovery step, downloads each listed document from S3 into a temporary directory, and runs **Docling** to extract text. Output is one Markdown file per document (e.g., `document_0.md`, `document_1.md`) written to the artifact output path. These files form the final text corpus for the RAG system.",
            ],
        ),
        "TEXT_EXTRACTION": NotebookCell(
            cell_type="code",
            source=[
                'descriptor_in = SimpleNamespace(path=str(step_output_dir / "discovered_documents"))\n',
                'extracted_text_out = SimpleNamespace(path=str(step_output_dir / "extracted_text"))\n',
                "\n",
                "text_extraction.python_func(\n",
                "    documents_descriptor=descriptor_in,\n",
                "    extracted_text=extracted_text_out,\n",
                ")",
            ],
        ),
        "CHAPTER_3": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Upload documents content into vector store\n",
                "\n",
                "This section configures the vector store, chunks the extracted documents, and uploads embeddings to the database for semantic search.\n",
                "\n",
                "&#x1F516; **Note**: This notebook requires a Llama Stack server to be available for the AutoRAG experiment. Detailed instructions on how to setup Llama Stack server for AutoRAG can be found here: https://github.com/LukaszCmielowski/prototypes/blob/main/llamastack/SETUP.md",
            ],
        ),
        "MD_3_1": NotebookCell(
            cell_type="markdown",
            source=[
                "### Prepare Llama Stack Client\n",
                "\n",
                "The Llama Stack client provides the interface to the embedding models and vector database. This section initializes the client using API credentials from environment variables or prompts.\n",
                "\n",
                "**Prerequisites:**\n",
                "- `LLAMA_STACK_CLIENT_API_KEY`: Your authentication key for the Llama Stack API\n",
                "- `LLAMA_STACK_CLIENT_BASE_URL`: The base URL of your Llama Stack instance\n",
                "\n",
                "&#x1F4A1; **Tip**: In OpenShift AI Workbench, you can add these as environment variables or data connections to avoid entering them manually each time.",
            ],
        ),
        "LS_CLIENT": NotebookCell(
            cell_type="code",
            source=[
                "from llama_stack_client import LlamaStackClient\n",
                "\n",
                'if not os.getenv("LLAMA_STACK_CLIENT_API_KEY") or not os.getenv("LLAMA_STACK_CLIENT_BASE_URL"):\n',
                '    os.environ["LLAMA_STACK_CLIENT_API_KEY"] = getpass.getpass("Please enter \'LLAMA_STACK_CLIENT_API_KEY\': ")\n',
                '    os.environ["LLAMA_STACK_CLIENT_BASE_URL"] = getpass.getpass("Please enter \'LLAMA_STACK_CLIENT_BASE_URL\': ")\n',
                "\n",
                "client = LlamaStackClient(\n",
                '    base_url=os.getenv("LLAMA_STACK_CLIENT_BASE_URL"),\n',
                '    api_key=os.getenv("LLAMA_STACK_CLIENT_API_KEY"),\n',
                ")",
            ],
        ),
        "MD_3_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Prepare chunker\n",
                "\n",
                "The chunker splits extracted documents into smaller chunks for more effective retrieval. Configuration includes:\n",
                "- **Chunking Method**: The algorithm used to split text (e.g., recursive character splitting)\n",
                "- **Chunk Size**: Maximum number of characters per chunk\n",
                "- **Chunk Overlap**: Number of overlapping characters between consecutive chunks to preserve context\n",
                "\n",
                "Proper chunking ensures that retrieved context is both relevant and fits within the model's context window.",
            ],
        ),
        "CHUNKER": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.chunking import LangChainChunker\n",
                "\n",
                'chunking_method = "{CHUNKING_METHOD}"\n',
                "chunk_size = {CHUNK_SIZE}\n",
                "chunk_overlap = {CHUNK_OVERLAP}\n",
                "\n",
                "chunker = LangChainChunker(method=chunking_method, chunk_size=chunk_size, chunk_overlap=chunk_overlap)",
            ],
        ),
        "MD_3_3": NotebookCell(
            cell_type="markdown",
            source=[
                "### Initialize vector store\n",
                "\n",
                "The vector store manages document embeddings and enables semantic search. This section configures:\n",
                "- **Embedding Model**: Converts text chunks into vector representations\n",
                "- **Vector Database Provider**: The backend storage system (e.g., Milvus)\n",
                "- **Distance Metric**: How similarity is calculated (cosine, euclidean, etc.)\n",
                "- **Collection Name**: A named collection where embeddings are stored\n",
                "\n",
                "The vector store is initialized and ready to receive document chunks.",
            ],
        ),
        "VECTOR_STORE": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel, LSEmbeddingParams\n",
                "from ai4rag.rag.vector_store.llama_stack import LSVectorStore\n",
                "\n",
                'model_id = "{MODEL_ID}"\n',
                "params = LSEmbeddingParams(**{EMBEDDING_PARAMS})\n",
                "\n",
                "embedding_model = LSEmbeddingModel(client=client, model_id=model_id, params=params)\n",
                "\n",
                'provider_id = "{PROVIDER_ID}"\n',
                'distance_metric = "{DISTANCE_METRIC}"\n',
                'collection_name = "{COLLECTION_NAME}"\n',
                "\n",
                "ls_vectorstore = LSVectorStore(\n",
                "    embedding_model=embedding_model,\n",
                "    client=client,\n",
                "    provider_id=provider_id,\n",
                "    distance_metric=distance_metric,\n",
                "    reuse_collection_name=collection_name\n",
                ")",
            ],
        ),
        "MD_3_4": NotebookCell(
            cell_type="markdown",
            source=[
                "### Upload chunks to vector store\n",
                "\n",
                "This section processes each extracted markdown file by:\n",
                "- Loading the document content with metadata\n",
                "- Splitting it into chunks using the configured chunker\n",
                "- Generating embeddings and uploading them to the vector store\n",
                "\n",
                "Once complete, all document chunks are indexed and ready for semantic search queries.",
            ],
        ),
        "CHUNKS_UPLOAD": NotebookCell(
            cell_type="code",
            source=[
                'paths = list(Path("step_outputs/extracted_text").glob("*.md"))\n',
                "\n",
                "for p in sorted(paths):\n",
                "    document = Document(\n",
                '            page_content=p.read_text(encoding="utf-8", errors="replace"),\n',
                '            metadata={{"document_id": p.stem}},\n',
                "        )\n",
                "\n",
                "    chunked_documents = chunker.split_documents([document])\n",
                "    ls_vectorstore.add_documents(chunked_documents)",
            ],
        ),
        "MD_3_5": NotebookCell(
            cell_type="markdown",
            source="### Retrieve chunks for sample question\n\nThis section demonstrates how to perform a semantic search query against the populated vector store. You can test retrieval by searching for relevant chunks based on a sample question.",
        ),
        "SAMPLE_SEARCH": NotebookCell(
            cell_type="code",
            source=[
                "from pprint import pprint\n",
                "\n",
                "sample_question = input()\n",
                "\n",
                "results = ls_vectorstore.search(query=sample_question, k=5)\n",
                "for result in results:\n",
                "    if isinstance(result, tuple):\n",
                "        pprint(result[0].model_dump(mode='python'), indent=4)\n",
                "        continue\n",
                "    pprint(result.model_dump(mode='python'), indent=4)",
            ],
        ),
        "SUMMARY": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Summary\n",
                "\n",
                "This notebook successfully processed documents from S3 storage, extracted their text content using Docling, chunked the text into manageable pieces, and uploaded the embeddings to a vector store. The indexed documents are now ready for semantic search and retrieval in RAG applications.",
            ],
        ),
    }

    ls_generation_notebook_template: dict[str, NotebookCell] = {
        "BANNER": NotebookCell(cell_type="markdown", source=banner),
        "TABLE_OF_CONTENTS": NotebookCell(
            cell_type="markdown",
            source=[
                "## Pattern {PATTERN_NAME} Retrieve & Generation Content\n",
                "\n",
                "This notebook demonstrates how to implement and test a Retrieval-Augmented Generation (RAG) pattern using Llama Stack. It guides you through setting up the necessary components, loading test data from an S3 bucket, and querying the RAG system to generate responses based on retrieved context.\n",
                "\n",
                "&#x26A0;&#xFE0F; **Important**: Before running this notebook, you must first run the corresponding **indexing.ipynb** notebook to populate the vector store with document embeddings. The indexing process prepares the knowledge base that this notebook queries.\n",
                "\n",
                "### &#x1F4CB; Contents \n",
                "This notebook contains the following sections:\n",
                "\n",
                "- **[Setup](#Setup)**\n",
                "- **[Prepare LlamaStackClient](#Prepare-LlamaStackClient)**\n",
                "- **[Initialize RAG Components](#Initialize-RAG-Components)**\n",
                "   - [Initialize LlamaStack Foundation Model](#Initialize-LlamaStack-Foundation-Model)\n",
                "   - [Initialize Vector Store Client](#Initialize-Vector-Store-Client)\n",
                "   - [Initialize Retriever](#Initialize-Retriever)\n",
                "   - [Initialize RAG Pattern](#Initialize-RAG-Pattern)\n",
                "   - [Query RAG Pattern](#Query-RAG-Pattern)\n",
                "- **[Next steps](#Next-steps)**\n",
                "   - [Load Test Data](#Load-Test-Data)\n",
                "   - [Configure S3 Credentials](#Configure-S3-Credentials)\n",
                "   - [Initialize S3 Client](#Initialize-S3-Client)\n",
                "   - [Load Benchmark Data](#Load-Benchmark-Data)\n",
                "   - [Build Evaluation Data](#Build-Evaluation-Data)\n",
                "   - [Evaluate Response](#Evaluate-Response)\n",
                "- **[Summary](#Summary)**",
            ],
        ),
        "CHAPTER_1": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Setup\n",
                "\n",
                "This section installs all the required Python packages for running the RAG experiment:\n",
                "- **boto3**: AWS SDK for Python to interact with S3 storage\n",
                "- **pipelines-components**: Red Hat's pipeline components for data processing\n",
                "- **ai4rag**: The main RAG framework for AutoRAG experiments",
            ],
        ),
        "DEPENDENCIES": NotebookCell(
            cell_type="code",
            source=[
                "!pip install boto3 | tail -n 1\n",
                "!pip install -U --no-cache-dir git+https://github.com/LukaszCmielowski/pipelines-components.git@rhoai_autorag | tail -n 1\n",
                "!pip install 'ai4rag' | tail -n 1",
            ],
        ),
        "CHAPTER_2": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Prepare LlamaStackClient\n",
                "\n",
                "The Llama Stack client is the core interface for interacting with the Llama Stack API. This section initializes the client by:\n",
                "- Retrieving API credentials from environment variables or prompting for them\n",
                "- Establishing a connection to the Llama Stack endpoint\n",
                "\n",
                "**Prerequisites:**\n",
                "- `LLAMA_STACK_CLIENT_API_KEY`: Your authentication key for the Llama Stack API\n",
                "- `LLAMA_STACK_CLIENT_BASE_URL`: The base URL of your Llama Stack instance\n",
                "\n",
                "&#x1F4A1; **Tip**: In OpenShift AI Workbench, you can add these as environment variables or data connections to avoid entering them manually each time.",
            ],
        ),
        "LS_CLIENT": NotebookCell(
            cell_type="code",
            source=[
                "import os\n",
                "import getpass\n",
                "import warnings\n",
                "import logging\n",
                "\n",
                "from llama_stack_client import LlamaStackClient\n",
                "\n",
                'warnings.filterwarnings("ignore")\n',
                "logging.getLogger('httpx').propagate = False\n",
                "\n",
                'if not os.getenv("LLAMA_STACK_CLIENT_API_KEY") or not os.getenv("LLAMA_STACK_CLIENT_BASE_URL"):\n',
                '    os.environ["LLAMA_STACK_CLIENT_API_KEY"] = getpass.getpass("Please enter \'LLAMA_STACK_CLIENT_API_KEY\': ")\n',
                '    os.environ["LLAMA_STACK_CLIENT_BASE_URL"] = getpass.getpass("Please enter \'LLAMA_STACK_CLIENT_BASE_URL\': ")\n',
                "\n",
                "client = LlamaStackClient(\n",
                '    base_url=os.getenv("LLAMA_STACK_CLIENT_BASE_URL"),\n',
                '    api_key=os.getenv("LLAMA_STACK_CLIENT_API_KEY"),\n',
                ")",
            ],
        ),
        "CHAPTER_3": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Initialize RAG Components\n",
                "\n",
                "This section sets up all the components needed for the RAG pattern: foundation model, vector store, retriever, and the RAG pattern itself.",
            ],
        ),
        "MD_3_1": NotebookCell(
            cell_type="markdown",
            source=[
                "### Initialize LlamaStack Foundation Model\n",
                "\n",
                "The foundation model is the core language model that generates responses. This section configures:\n",
                "- **Model ID**: The specific Llama model to use for generation\n",
                "- **System Message**: Instructions that define the model's behavior and role\n",
                "- **User Message Template**: The format for user queries\n",
                "- **Context Template**: How retrieved context is incorporated into prompts\n",
                "\n",
                "These templates control how the RAG system structures prompts to the language model.",
            ],
        ),
        "FOUNDATION_MODEL": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel\n",
                "\n",
                'chat_model_id = """{FM_MODEL_ID}"""\n',
                'system_message_text = """{SYSTEM_MESSAGE}"""\n',
                'user_message_text = """{USER_MESSAGE}"""\n',
                'context_template_text = """{CONTEXT_TEXT}"""\n',
                "\n",
                "lsfoundationmodel = LSFoundationModel(\n",
                "    client=client,\n",
                "    model_id=chat_model_id,\n",
                "    system_message_text=system_message_text,\n",
                "    user_message_text=user_message_text,\n",
                "    context_template_text=context_template_text,\n",
                ")",
            ],
        ),
        "MD_3_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Initialize Vector Store Client\n",
                "\n",
                "The vector store is responsible for storing and retrieving document embeddings. This section sets up:\n",
                "- **Embedding Model**: Converts text into vector representations for semantic search\n",
                "- **Vector Database**: Stores embeddings with configurable distance metrics (cosine, euclidean, etc.)\n",
                "- **Collection**: A named collection where document vectors are stored and can be reused\n",
                "\n",
                "The vector store enables semantic similarity search to find relevant context for user queries.",
            ],
        ),
        "VECTOR_STORE": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel, LSEmbeddingParams\n",
                "from ai4rag.rag.vector_store.llama_stack import LSVectorStore\n",
                "\n",
                'embedding_model_id = "{MODEL_ID}"\n',
                "params = LSEmbeddingParams(**{EMBEDDING_PARAMS})\n",
                "embedding_model = LSEmbeddingModel(client=client, model_id=embedding_model_id, params=params)\n",
                'provider_id = "{PROVIDER_ID}"\n',
                'distance_metric = "{DISTANCE_METRIC}"\n',
                'collection_name = "{COLLECTION_NAME}"\n',
                "\n",
                "ls_vectorstore = LSVectorStore(\n",
                "    embedding_model=embedding_model,\n",
                "    client=client,\n",
                "    provider_id=provider_id,\n",
                "    distance_metric=distance_metric,\n",
                "    reuse_collection_name=collection_name\n",
                ")",
            ],
        ),
        "MD_3_3": NotebookCell(
            cell_type="markdown",
            source=[
                "### Initialize Retriever\n",
                "\n",
                "The retriever finds the most relevant document chunks for a given query. Configuration includes:\n",
                "- **Retrieval Method**: The algorithm used to find relevant documents (e.g., similarity search, hybrid search)\n",
                "- **Number of Chunks**: How many document chunks to retrieve and include in the context\n",
                "\n",
                "The retriever acts as the bridge between user questions and the knowledge base.",
            ],
        ),
        "RETRIEVER": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.retrieval.retriever import Retriever\n",
                "\n",
                'method = "{RETRIEVAL_METHOD}"\n',
                "number_of_chunks = {NUMBER_OF_CHUNKS}\n",
                "\n",
                "retriever = Retriever(vector_store=ls_vectorstore, method=method, number_of_chunks=number_of_chunks)",
            ],
        ),
        "MD_3_4": NotebookCell(
            cell_type="markdown",
            source=[
                "### Initialize RAG Pattern\n",
                "\n",
                "This section brings together all components into a complete RAG pattern:\n",
                "- Combines the foundation model with the retriever\n",
                "- Creates a unified interface for question-answering\n",
                "- Coordinates the retrieve-then-generate workflow\n",
                "\n",
                "The RAG pattern orchestrates: query, retrieve context, generate response.",
            ],
        ),
        "RAG_PATTERN": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.template.llama_stack_rag_template import LlamaStackRAG\n",
                "\n",
                "rag_pattern = LlamaStackRAG(foundation_model=lsfoundationmodel, retriever=retriever)",
            ],
        ),
        "MD_3_4": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "### Query RAG Pattern\n",
                "\n",
                "This section executes the RAG workflow by submitting test questions to the system and generating responses based on retrieved context.",
            ],
        ),
        "TEST_RESPONSE": NotebookCell(
            cell_type="code",
            source=[
                "from pprint import pprint\n",
                "\n",
                "question = input()\n",
                "response = rag_pattern.generate(question=question)\n",
                "pprint(response, indent=4, width=50)",
            ],
        ),
        "CHAPTER_4": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Next steps\n",
                "\n",
                "The following sections provide optional next steps for loading test data, running queries, and evaluating the RAG pattern's performance. These steps are useful for systematic testing and benchmarking, but can be skipped if you prefer to interact with the RAG system directly using the pattern configured above.",
            ],
        ),
        "MD_4_1": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "### Load Test Data\n",
                "\n",
                "This section prepares the test environment and loads benchmark questions from S3 storage. The test data is used to evaluate the RAG system's performance.",
            ],
        ),
        "LOAD_DATA_IMPORTS": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "import os\n",
                "import json\n",
                "from pathlib import Path\n",
                "from types import SimpleNamespace\n",
                "\n",
                "import boto3\n",
                "\n",
                "logging.getLogger('Test Data Loader component logger').propagate = False\n",
                "```",
            ],
        ),
        "MD_4_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Configure S3 Credentials\n",
                "\n",
                "To load test data from S3-compatible object storage, you need to provide credentials. If you're using OpenShift AI, these can be configured as data connections.\n",
                "\n",
                "&#x1F4CC; **Action**: Provide the credentials for your S3 instance if they are not already set in the notebook environment.\n",
                "\n",
                "&#x1F4A1; **Tip**: In the project, open **Connections** and add an **S3 compatible object storage connection** to a bucket you will use for documents and test data. Open **Workbenches**, edit your workbench, and attach the S3 connection you created so the notebook can read from the bucket. Save and restart the workbench if prompted.",
            ],
        ),
        "AWS_ENV": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                'required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT", "AWS_DEFAULT_REGION", "AWS_S3_BUCKET"]\n',
                "missing = [var for var in required_vars if not os.environ.get(var)]\n",
                "if missing:\n",
                '    raise ValueError(f"Missing required environment variables: {{missing}}")\n',
                "```",
            ],
        ),
        "MD_4_3": NotebookCell(
            cell_type="markdown",
            source="### Initialize S3 Client\n\nCreates an S3 client session using the provided credentials. This client is used to download test data from the specified S3 bucket.",
        ),
        "S3_CLIENT": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "session = boto3.session.Session(\n",
                '    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],\n',
                '    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],\n',
                ")\n",
                "s3_client = session.client(\n",
                "    service_name='s3',\n",
                '    endpoint_url=os.environ["AWS_S3_ENDPOINT"],\n',
                ")\n",
                "```",
            ],
        ),
        "MD_4_4": NotebookCell(
            cell_type="markdown",
            source=[
                "### Load Benchmark Data\n",
                "\n",
                "Downloads and loads the benchmark test data from S3. The benchmark file should be a JSON file containing:\n",
                "- **question**: The test question to ask the RAG system\n",
                "- **correct_answers**: The expected answers for evaluation\n",
                "- **correct_answer_document_ids**: IDs of documents that contain the correct information\n",
                "\n",
                "This data is used to measure the RAG system's accuracy and retrieval quality.",
            ],
        ),
        "TEST_DATA_LOADER": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "from kfp_components.components.data_processing.autorag.test_data_loader.component import test_data_loader\n",
                "\n",
                "\n",
                'step_output_dir = Path("./step_outputs")\n',
                "step_output_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "test_data_bucket_name = os.environ['AWS_S3_BUCKET']\n",
                'test_data_key = "{TEST_DATA_KEY}"\n',
                'test_data_out = SimpleNamespace(path=str(step_output_dir / "test_data.json"))\n',
                "\n",
                "test_data_loader.python_func(\n",
                "    test_data_bucket_name=test_data_bucket_name,\n",
                "    test_data_path=test_data_key,\n",
                "    test_data=test_data_out,\n",
                ")\n",
                "\n",
                "output_path = Path(test_data_out.path)\n",
                'with output_path.open("r", encoding="utf-8") as f:\n',
                "    test_data = json.load(f)\n",
                "\n",
                "print(json.dumps(test_data, indent=4, ensure_ascii=False))\n",
                "```",
            ],
        ),
        "EXECUTE_QUERIES": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "inference_responses = []\n",
                "\n",
                "for test_data_item in test_data:\n",
                '    response = rag_pattern.generate(question=test_data_item["question"])\n',
                "    inference_responses.append(response)\n",
                "```",
            ],
        ),
        "MD_4_5": NotebookCell(
            cell_type="markdown",
            source=[
                "### Build Evaluation Data\n",
                "\n",
                "This section transforms the RAG system's inference responses into a structured format for evaluation. It combines:\n",
                "- **Benchmark Data**: The original test questions and expected answers\n",
                "- **Inference Responses**: The actual responses generated by the RAG system\n",
                "\n",
                "The resulting evaluation data structure allows for systematic comparison between expected and actual outputs, enabling metric calculation for assessing the RAG system's performance.",
            ],
        ),
        "BUILD_EVAL_DATA": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "from pandas import DataFrame\n",
                "\n",
                "from ai4rag.core.experiment.utils import build_evaluation_data\n",
                "from ai4rag.core.experiment.benchmark_data import BenchmarkData\n",
                "\n",
                "\n",
                "benchmark_data = BenchmarkData(\n",
                "    DataFrame(\n",
                "        data=test_data\n",
                "    )\n",
                ")\n",
                "\n",
                "evaluation_data = build_evaluation_data(\n",
                "    benchmark_data=benchmark_data, \n",
                "    inference_response=inference_responses\n",
                ")\n",
                "```",
            ],
        ),
        "MD_4_6": NotebookCell(
            cell_type="markdown",
            source=[
                "### Evaluate Response\n",
                "\n",
                "This section evaluates the quality of the RAG system's responses by comparing them against the expected answers from the benchmark data. Evaluation metrics may include accuracy, relevance, and retrieval precision.",
            ],
        ),
        "EVALUATE_RESPONSE": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator\n",
                "from ai4rag.evaluator.base_evaluator import MetricType\n",
                "\n",
                "evaluator = UnitxtEvaluator()\n",
                "evaluator.evaluate_metrics(evaluation_data=evaluation_data, metrics=(MetricType.ANSWER_CORRECTNESS, MetricType.FAITHFULNESS, MetricType.CONTEXT_CORRECTNESS))\n",
                "```",
            ],
        ),
        "SUMMARY": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Summary\n",
                "\n",
                "This notebook successfully demonstrates a complete RAG pattern implementation using Llama Stack, from initializing the foundation model and vector store to querying the system with test data. The evaluation framework allows you to measure the quality of generated responses against benchmark answers using multiple metrics including answer correctness, faithfulness, and context correctness.",
            ],
        ),
    }

    chroma_notebook_template: dict[
        str,
        NotebookCell,
    ] = {
        "BANNER": NotebookCell(
            cell_type="markdown",
            source=banner,
        ),
        "TABLE_OF_CONTENTS": NotebookCell(
            cell_type="markdown",
            source=[
                "## Pattern {PATTERN_NAME} Index Building, Retrieve & Generation Content\n",
                "\n",
                "This notebook demonstrates RAG Pattern using ChromaDB for vector storage and OpenAI for embeddings and generation. It covers:\n",
                "- Document discovery and text extraction from S3 storage\n",
                "- Building a vector store index with ChromaDB and OpenAI embeddings\n",
                "- Implementing retrieval-augmented generation for question answering\n",
                "- Evaluating RAG performance with benchmark data\n",
                "\n",
                "### &#x1F4CB; Contents \n",
                "This notebook contains the following sections:\n",
                "\n",
                "- **[Setup](#Setup)**\n",
                "  - [Install packages](#Install-packages)\n",
                "  - [Import required libraries](#Import-required-libraries)\n",
                "  - [Configure S3 credentials](#Configure-S3-credentials)\n",
                "  - [Prepare S3 client](#Prepare-S3-client)\n",
                "- **[Process input documents](#Process-input-documents)**\n",
                "  - [Documents discovery](#Documents-discovery)\n",
                "  - [Text extraction](#Text-extraction)\n",
                "- **[Upload documents content into vector store database](#Upload-documents-content-into-vector-store-database)**\n",
                "  - [Configure OpenAI credentials](#Configure-OpenAI-credentials)\n",
                "  - [Prepare chunker](#Prepare-chunker)\n",
                "  - [Initialize ChromaDB vector store](#Initialize-ChromaDB-vector-store)\n",
                "  - [Upload chunks to vector store](#Upload-chunks-to-vector-store)\n",
                "  - [Retrieve chunks for sample question](#Retrieve-chunks-for-sample-question)\n",
                "- **[Retrieve & Generation](#Retrieve-&-Generation)**\n",
                "  - [Initialize OpenAI Foundation Model](#Initialize-OpenAI-Foundation-Model)\n",
                "  - [Initialize Retriever](#Initialize-Retriever)\n",
                "  - [Initialize RAG Pattern](#Initialize-RAG-Pattern)\n",
                "  - [Query RAG Pattern](#Query-RAG-Pattern)\n",
                "- **[Next steps](#Next-steps)**\n",
                "  - [Load Test Data](#Load-Test-Data)\n",
                "  - [Build Evaluation Data](#Build-Evaluation-Data)\n",
                "  - [Evaluate Response](#Evaluate-Response)\n",
                "- **[Summary](#Summary)**",
            ],
        ),
        "CHAPTER_1": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Setup\n",
                "\n",
                "This section sets up the notebook environment by installing required packages, importing libraries, and configuring access to S3 storage.\n",
                "\n",
                "### Install packages\n",
                "\n",
                "Install all required Python packages for document processing and RAG operations:\n",
                "- **boto3**: AWS SDK for Python to interact with S3 storage\n",
                "- **pipelines-components**: Red Hat's pipeline components for data processing\n",
                "- **docling**: Document processing and text extraction library\n",
                "- **ai4rag**: The AutoRAG framework for building RAG applications\n",
                "- **openai**: OpenAI Python SDK for embeddings and chat completions\n",
                "- **chromadb**: Vector database for semantic search",
            ],
        ),
        "DEPENDENCIES": NotebookCell(
            cell_type="code",
            source=[
                "!pip install boto3 | tail -n 1\n",
                "!pip install -U --no-cache-dir git+https://github.com/LukaszCmielowski/pipelines-components.git@rhoai_autorag | tail -n 1\n",
                "!pip install docling | tail -n 1\n",
                "!pip install 'ai4rag' | tail -n 1",
            ],
        ),
        "MD_1_1": NotebookCell(
            cell_type="markdown",
            source="### Import required libraries\n\nImport all necessary Python modules and configure logging to suppress verbose output from component loggers.",
        ),
        "MAIN_IMPORTS": NotebookCell(
            cell_type="code",
            source=[
                "import os\n",
                "import json\n",
                "import logging\n",
                "from pathlib import Path\n",
                "from types import SimpleNamespace\n",
                "import getpass\n",
                "\n",
                "import warnings\n",
                'warnings.filterwarnings("ignore")\n',
                "\n",
                "import boto3\n",
                "from langchain_core.documents import Document\n",
                "\n",
                "for logger_name in (\n",
                '        "httpx",\n',
                '        "Document Loader component logger",\n',
                '        "Text Extraction component logger",\n',
                '        "Test Data Loader component logger", \n',
                "):\n",
                "    logging.getLogger(logger_name).propagate = False",
            ],
        ),
        "MD_1_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Configure S3 credentials\n",
                "\n",
                "To load documents from S3-compatible object storage, you need to provide credentials. If you're using OpenShift AI, these can be configured as data connections.\n",
                "\n",
                "&#x1F4CC; **Action**: Provide the credentials for your S3 instance if they are not already set in the notebook environment.\n",
                "\n",
                "&#x1F4A1; **Tip**: In the project, open **Connections** and add an **S3 compatible object storage connection** to a bucket you will use for documents and test data. Open **Workbenches**, edit your workbench, and attach the S3 connection you created so the notebook can read from the bucket. Save and restart the workbench if prompted.",
            ],
        ),
        "AWS_ENV": NotebookCell(
            cell_type="code",
            source=[
                'required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT", "AWS_DEFAULT_REGION", "AWS_S3_BUCKET"]\n',
                "missing = [var for var in required_vars if not os.environ.get(var)]\n",
                "if missing:\n",
                '    raise ValueError(f"Missing required environment variables: {{missing}}")',
            ],
        ),
        "MD_1_3": NotebookCell(
            cell_type="markdown",
            source="### Prepare S3 client\n\nCreates an S3 client session using the provided credentials. This client will be used to discover and download documents from the specified S3 bucket.",
        ),
        "S3_CLIENT": NotebookCell(
            cell_type="code",
            source=[
                "session = boto3.session.Session(\n",
                '    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],\n',
                '    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],\n',
                ")\n",
                "s3_client = session.client(\n",
                "    service_name='s3',\n",
                '    endpoint_url=os.environ["AWS_S3_ENDPOINT"],\n',
                ")",
            ],
        ),
        "CHAPTER_2": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Process input documents\n",
                "\n",
                "This section handles document discovery and text extraction. Documents are first discovered in S3 storage, then their content is extracted and converted to markdown format for further processing.",
            ],
        ),
        "MD_2_1": NotebookCell(
            cell_type="markdown",
            source=[
                "The data processing pipeline prepares documents for the RAG system in multiple steps. Each step runs as a standalone component with outputs stored under `step_outputs/`. \n",
                "\n",
                "| Step | Component | Purpose |\n",
                "|------|-----------|---------|\n",
                "| 1 | **Documents discovery** | List documents in the bucket, prioritize benchmark-referenced docs, apply a size cap, and write a JSON manifest (no content download). |\n",
                "| 2 | **Text extraction** | Download the listed documents from S3 and extract text to Markdown using Docling. |",
            ],
        ),
        "LOAD_DATA": NotebookCell(
            cell_type="code",
            source=[
                "from kfp_components.components.data_processing.autorag.documents_discovery.component import documents_discovery\n",
                "from kfp_components.components.data_processing.autorag.text_extraction.component import text_extraction\n",
                "\n",
                'step_output_dir = Path("./step_outputs")\n',
                "input_data_bucket_name = os.environ['AWS_S3_BUCKET']\n",
                'input_data_key = "{INPUT_DATA_KEY}"\n',
                "step_output_dir.mkdir(parents=True, exist_ok=True)",
            ],
        ),
        "MD_2_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Documents discovery\n",
                "\n",
                "Lists objects in the S3 input bucket, filters by supported extensions (e.g., `.pdf`, `.docx`, `.pptx`, `.md`, `.html`, `.txt`), and builds a document set. Documents referenced in the benchmark are prioritized, then others are added until a configurable size limit (1 GB by default) is reached. This step does not download document contents but writes a JSON manifest (`documents_descriptor.json`) containing the bucket, prefix, and list of selected object keys and sizes for the next step.",
            ],
        ),
        "DOCUMENTS_DISCOVERY": NotebookCell(
            cell_type="code",
            source=[
                'discovered_documents_out = SimpleNamespace(path=str(step_output_dir / "discovered_documents"))\n',
                "\n",
                "documents_discovery.python_func(\n",
                "    input_data_bucket_name=input_data_bucket_name,\n",
                "    input_data_path=input_data_key,\n",
                "    discovered_documents=discovered_documents_out,\n",
                ")\n",
                "\n",
                'descriptor_path = step_output_dir / "discovered_documents" / "documents_descriptor.json"\n',
                "with open(descriptor_path) as f:\n",
                "    descriptor = json.load(f)\n",
                "\n",
                "print(json.dumps(descriptor, indent=4, ensure_ascii=False))",
            ],
        ),
        "MD_2_3": NotebookCell(
            cell_type="markdown",
            source=[
                "### Text extraction\n",
                "\n",
                "Reads the `documents_descriptor.json` produced by the discovery step, downloads each listed document from S3 into a temporary directory, and runs **Docling** to extract text. Output is one Markdown file per document (e.g., `document_0.md`, `document_1.md`) written to the artifact output path. These files form the final text corpus for the RAG system.",
            ],
        ),
        "TEXT_EXTRACTION": NotebookCell(
            cell_type="code",
            source=[
                'descriptor_in = SimpleNamespace(path=str(step_output_dir / "discovered_documents"))\n',
                'extracted_text_out = SimpleNamespace(path=str(step_output_dir / "extracted_text"))\n',
                "\n",
                "text_extraction.python_func(\n",
                "    documents_descriptor=descriptor_in,\n",
                "    extracted_text=extracted_text_out,\n",
                ")",
            ],
        ),
        "CHAPTER_3": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Upload documents content into vector store\n",
                "\n",
                "This section configures the ChromaDB vector store, chunks the extracted documents, and uploads embeddings using OpenAI's embedding model to the database for semantic search.",
            ],
        ),
        "MD_3_1": NotebookCell(
            cell_type="markdown",
            source=[
                "### Configure OpenAI credentials\n",
                "\n",
                "Configure OpenAI API credentials for embeddings and chat completions. You need to provide:\n",
                "- **OPENAI_API_KEY**: Your OpenAI API key or compatible API key\n",
                "- **OPENAI_BASE_URL**: The base URL for the OpenAI API (or compatible endpoint)\n",
                "\n",
                "&#x1F4CC; **Action**: Provide the credentials if they are not already set in the notebook environment.\n",
                "\n",
                "&#x1F4A1; **Tip**: For OpenAI-compatible endpoints (like Azure OpenAI or local models), adjust the base URL accordingly.",
            ],
        ),
        "OPENAI_CLIENT": NotebookCell(
            cell_type="code",
            source=[
                "from openai import OpenAI\n",
                "\n",
                'if not os.getenv("CHAT_MODEL_TOKEN"):\n',
                '    os.environ["CHAT_MODEL_TOKEN"] = getpass.getpass("Please enter \'CHAT_MODEL_TOKEN\': ")\n',
                "\n",
                'if not os.getenv("EMBEDDING_MODEL_TOKEN"):\n',
                '    os.environ["EMBEDDING_MODEL_TOKEN"] = getpass.getpass("Please enter \'EMBEDDING_MODEL_TOKEN\': ")\n',
                "\n",
                "openai_foundation_model_client = OpenAI(\n",
                '    api_key=os.environ["CHAT_MODEL_TOKEN"],\n',
                '    base_url="{CHAT_MODEL_URL}",\n',
                ")\n",
                "\n",
                "openai_embedding_model_client = OpenAI(\n",
                '    api_key=os.environ["EMBEDDING_MODEL_TOKEN"],\n',
                '    base_url="{EMBEDDING_MODEL_URL}",\n',
                ")",
            ],
        ),
        "MD_3_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Prepare chunker\n",
                "\n",
                "The chunker splits extracted documents into smaller chunks for more effective retrieval. Configuration includes:\n",
                "- **Chunking Method**: The algorithm used to split text (e.g., recursive character splitting)\n",
                "- **Chunk Size**: Maximum number of characters per chunk\n",
                "- **Chunk Overlap**: Number of overlapping characters between consecutive chunks to preserve context\n",
                "\n",
                "Proper chunking ensures that retrieved context is both relevant and fits within the model's context window.",
            ],
        ),
        "CHUNKER": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.chunking import LangChainChunker\n",
                "\n",
                'chunking_method = "{CHUNKING_METHOD}"\n',
                "chunk_size = {CHUNK_SIZE}\n",
                "chunk_overlap = {CHUNK_OVERLAP}\n",
                "\n",
                "chunker = LangChainChunker(method=chunking_method, chunk_size=chunk_size, chunk_overlap=chunk_overlap)",
            ],
        ),
        "MD_3_3": NotebookCell(
            cell_type="markdown",
            source=[
                "### Initialize ChromaDB vector store\n",
                "\n",
                "The ChromaDB vector store manages document embeddings and enables semantic search. This section configures:\n",
                "- **OpenAI Embedding Model**: Converts text chunks into vector representations using OpenAI's embedding API\n",
                "- **ChromaDB**: The vector database backend for storing and querying embeddings\n",
                "- **Collection Name**: A named collection where embeddings are stored\n",
                "\n",
                "The vector store is initialized and ready to receive document chunks.",
            ],
        ),
        "VECTOR_STORE": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel\n",
                "from ai4rag.rag.vector_store.chroma import ChromaVectorStore\n",
                "\n",
                'embedding_model_id = "{EMBEDDING_MODEL_ID}"\n',
                "params = dict(**{{EMBEDDING_PARAMS}})\n",
                "open_ai_embedding_model = OpenAIEmbeddingModel(\n",
                "    client=openai_embedding_model_client,\n",
                "    model_id=embedding_model_id,\n",
                "    params=params\n",
                "\n",
                ")\n",
                'collection_name = "{COLLECTION_NAME}"\n',
                "\n",
                "chroma_vectorstore = ChromaVectorStore(\n",
                "    embedding_model=open_ai_embedding_model,\n",
                "    collection_name=collection_name,\n",
                ")",
            ],
        ),
        "MD_3_4": NotebookCell(
            cell_type="markdown",
            source=[
                "### Upload chunks to vector store\n",
                "\n",
                "This section processes each extracted markdown file by:\n",
                "- Loading the document content with metadata\n",
                "- Splitting it into chunks using the configured chunker\n",
                "- Generating embeddings using OpenAI and uploading them to ChromaDB\n",
                "\n",
                "Once complete, all document chunks are indexed and ready for semantic search queries.",
            ],
        ),
        "CHUNKS_UPLOAD": NotebookCell(
            cell_type="code",
            source=[
                'paths = list(Path("step_outputs/extracted_text").glob("*.md"))\n',
                "\n",
                "for p in sorted(paths):\n",
                "    document = Document(\n",
                '            page_content=p.read_text(encoding="utf-8", errors="replace"),\n',
                '            metadata={{"document_id": p.stem}},\n',
                "        )\n",
                "\n",
                "    chunked_documents = chunker.split_documents([document])\n",
                "    chroma_vectorstore.add_documents(chunked_documents)",
            ],
        ),
        "MD_3_5": NotebookCell(
            cell_type="markdown",
            source="### Retrieve chunks for sample question\n\nThis section demonstrates how to perform a semantic search query against the populated vector store. You can test retrieval by searching for relevant chunks based on a sample question.",
        ),
        "SAMPLE_SEARCH": NotebookCell(
            cell_type="code",
            source=[
                "from pprint import pprint\n",
                "\n",
                "sample_question = input()\n",
                "\n",
                "results = chroma_vectorstore.search(query=sample_question, k=5)\n",
                "for result in results:\n",
                "    if isinstance(result, tuple):\n",
                "        pprint(result[0].model_dump(mode='python'), indent=4)\n",
                "        continue\n",
                "    pprint(result.model_dump(mode='python'), indent=4)",
            ],
        ),
        "CHAPTER_4": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Retrieve & Generation\n",
                "\n",
                "This section sets up the RAG pattern for generating answers to questions using retrieved context from the ChromaDB vector store and OpenAI's chat completion API.",
            ],
        ),
        "MD_4_1": NotebookCell(
            cell_type="markdown",
            source=[
                "### Initialize OpenAI Foundation Model\n",
                "\n",
                "The foundation model is the core language model that generates responses. This section configures:\n",
                "- **Model ID**: The specific OpenAI model to use for generation (e.g., gpt-4, gpt-3.5-turbo)\n",
                "- **System Message**: Instructions that define the model's behavior and role\n",
                "- **User Message Template**: The format for user queries\n",
                "- **Context Template**: How retrieved context is incorporated into prompts\n",
                "\n",
                "These templates control how the RAG system structures prompts to the language model.",
            ],
        ),
        "FOUNDATION_MODEL": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel\n",
                "\n",
                'foundation_model_id = "{FOUNDATION_MODEL_ID}"\n',
                'system_message_text = """{SYSTEM_MESSAGE}"""\n',
                'user_message_text = """{USER_MESSAGE}"""\n',
                'context_template_text = """{CONTEXT_TEXT}"""\n',
                "\n",
                "openai_foundation_model = OpenAIFoundationModel(\n",
                "    client=openai_foundation_model_client,\n",
                "    model_id=foundation_model_id,\n",
                "    system_message_text=system_message_text,\n",
                "    user_message_text=user_message_text,\n",
                "    context_template_text=context_template_text,\n",
                ")",
            ],
        ),
        "MD_4_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Initialize Retriever\n",
                "\n",
                "The retriever finds the most relevant document chunks for a given query. Configuration includes:\n",
                "- **Retrieval Method**: The algorithm used to find relevant documents (e.g., similarity search, hybrid search)\n",
                "- **Number of Chunks**: How many document chunks to retrieve and include in the context\n",
                "\n",
                "The retriever acts as the bridge between user questions and the knowledge base.",
            ],
        ),
        "RETRIEVER": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.retrieval.retriever import Retriever\n",
                "\n",
                'method = "{RETRIEVAL_METHOD}"\n',
                "number_of_chunks = {NUMBER_OF_CHUNKS}\n",
                "\n",
                "retriever = Retriever(vector_store=chroma_vectorstore, method=method, number_of_chunks=number_of_chunks)",
            ],
        ),
        "MD_4_3": NotebookCell(
            cell_type="markdown",
            source="### Initialize RAG Pattern\n\nCombines the foundation model and retriever into a complete RAG pattern that can answer questions by retrieving relevant context and generating responses.",
        ),
        "RAG_PATTERN": NotebookCell(
            cell_type="code",
            source=[
                "from ai4rag.rag.template.simple_rag_template import SimpleRAG\n",
                "\n",
                "rag_pattern = SimpleRAG(foundation_model=openai_foundation_model, retriever=retriever)",
            ],
        ),
        "MD_4_4": NotebookCell(
            cell_type="markdown",
            source="### Query RAG Pattern\n\nThis section executes the RAG workflow by submitting test questions to the system and generating responses based on retrieved context from ChromaDB.",
        ),
        "TEST_RESPONSE": NotebookCell(
            cell_type="code",
            source=[
                "from pprint import pprint\n",
                "\n",
                "question = input()\n",
                "response = rag_pattern.generate(question=question)\n",
                "pprint(response, indent=4, width=50)",
            ],
        ),
        "CHAPTER_5": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Next steps\n",
                "\n",
                "The following sections provide optional next steps for loading test data, running queries, and evaluating the RAG pattern's performance. These steps are useful for systematic testing and benchmarking, but can be skipped if you prefer to interact with the RAG system directly using the pattern configured above.",
            ],
        ),
        "MD_5_1": NotebookCell(
            cell_type="markdown",
            source=[
                "### Load Test Data\n",
                "\n",
                "Downloads and loads the benchmark test data from S3. The benchmark file should be a JSON file containing:\n",
                "- **question**: The test question to ask the RAG system\n",
                "- **correct_answers**: The expected answers for evaluation\n",
                "- **correct_answer_document_ids**: IDs of documents that contain the correct information\n",
                "\n",
                "This data is used to measure the RAG system's accuracy and retrieval quality.",
            ],
        ),
        "TEST_DATA_LOADER": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "from kfp_components.components.data_processing.autorag.test_data_loader.component import test_data_loader\n",
                "\n",
                "\n",
                'step_output_dir = Path("./step_outputs")\n',
                "step_output_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "test_data_bucket_name = os.environ['AWS_S3_BUCKET']\n",
                'test_data_key = "{TEST_DATA_KEY}"\n',
                'test_data_out = SimpleNamespace(path=str(step_output_dir / "test_data.json"))\n',
                "\n",
                "test_data_loader.python_func(\n",
                "    test_data_bucket_name=test_data_bucket_name,\n",
                "    test_data_path=test_data_key,\n",
                "    test_data=test_data_out,\n",
                ")\n",
                "\n",
                "output_path = Path(test_data_out.path)\n",
                'with output_path.open("r", encoding="utf-8") as f:\n',
                "    test_data = json.load(f)\n",
                "\n",
                "print(json.dumps(test_data, indent=4, ensure_ascii=False))\n",
                "```",
            ],
        ),
        "EXECUTE_QUERIES": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "inference_responses = []\n",
                "\n",
                "for test_data_item in test_data:\n",
                '    response = rag_pattern.generate(question=test_data_item["question"])\n',
                "    inference_responses.append(response)\n",
                "```",
            ],
        ),
        "MD_5_2": NotebookCell(
            cell_type="markdown",
            source=[
                "### Build Evaluation Data\n",
                "\n",
                "This section transforms the RAG system's inference responses into a structured format for evaluation. It combines:\n",
                "- **Benchmark Data**: The original test questions and expected answers\n",
                "- **Inference Responses**: The actual responses generated by the RAG system\n",
                "\n",
                "The resulting evaluation data structure allows for systematic comparison between expected and actual outputs, enabling metric calculation for assessing the RAG system's performance.",
            ],
        ),
        "BUILD_EVAL_DATA": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "from pandas import DataFrame\n",
                "\n",
                "from ai4rag.core.experiment.utils import build_evaluation_data\n",
                "from ai4rag.core.experiment.benchmark_data import BenchmarkData\n",
                "\n",
                "\n",
                "benchmark_data = BenchmarkData(\n",
                "    DataFrame(\n",
                "        data=test_data\n",
                "    )\n",
                ")\n",
                "\n",
                "evaluation_data = build_evaluation_data(\n",
                "    benchmark_data=benchmark_data, \n",
                "    inference_response=inference_responses\n",
                ")\n",
                "```",
            ],
        ),
        "MD_5_3": NotebookCell(
            cell_type="markdown",
            source=[
                "### Evaluate Response\n",
                "\n",
                "This section evaluates the quality of the RAG system's responses by comparing them against the expected answers from the benchmark data. Evaluation metrics may include accuracy, relevance, and retrieval precision.",
            ],
        ),
        "EVALUATE_RESPONSE": NotebookCell(
            cell_type="markdown",
            source=[
                "```python\n",
                "from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator\n",
                "from ai4rag.evaluator.base_evaluator import MetricType\n",
                "\n",
                "evaluator = UnitxtEvaluator()\n",
                "evaluator.evaluate_metrics(evaluation_data=evaluation_data, metrics=(MetricType.ANSWER_CORRECTNESS, MetricType.FAITHFULNESS, MetricType.CONTEXT_CORRECTNESS))\n",
                "```",
            ],
        ),
        "SUMMARY": NotebookCell(
            cell_type="markdown",
            source=[
                "---\n",
                "\n",
                "## Summary\n",
                "\n",
                "This notebook successfully demonstrates a complete RAG pattern implementation using ChromaDB for vector storage and OpenAI for embeddings and generation. The pipeline includes:\n",
                "- Document discovery and text extraction from S3 storage\n",
                "- Chunking and embedding documents using OpenAI's embedding API\n",
                "- Storing embeddings in ChromaDB for semantic search\n",
                "- Retrieving relevant context and generating answers using OpenAI's chat completion API\n",
                "\n",
                "The indexed documents are now ready for semantic search and retrieval in RAG applications.",
            ],
        ),
    }

    match template:
        case "chroma":
            return chroma_notebook_template
        case "ls_indexing":
            return ls_indexing_notebook_template
        case "ls_inference":
            return ls_generation_notebook_template
        case _:
            raise AssetGenerationError(
                "Unknown teamplate provided, you can only choose from 'chroma', 'ls_indexing', 'ls_inference'."
            )
