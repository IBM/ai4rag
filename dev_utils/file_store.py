# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, settings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".html", ".txt"}


class FileStoreException(Exception):
    pass


class FileStore:
    """
    Class used to load locally saved input files.
    Uses docling library to extract content from documents as markdown.
    """

    def __init__(self, path: str | Path | Sequence[str] | Sequence[Path], save_dir: str | Path | None = None):
        """
        Parameters
        ----------
        path : str | Path | Sequence[str] | Sequence[Path]
            Path to a single file or a directory of files.
        save_dir : str | Path | None
            Optional directory to save extracted markdown files. If None, no files are saved.
        """
        self.path = Path(path)
        self.is_dir = self.path.is_dir()
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.files = {}

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        pipeline_options.accelerator_options = AcceleratorOptions(device="auto")

        num_workers = os.cpu_count() or 1
        settings.perf.doc_batch_size = num_workers
        settings.perf.doc_batch_concurrency = num_workers

        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"

    def load_as_documents(self) -> list[Document]:
        """Read files as langchain documents"""
        contents = self._load_content()
        return [Document(page_content=content[0], metadata={"document_id": content[1]}) for content in contents]

    @lru_cache(maxsize=2)
    def _load_content(self) -> list[tuple[str, str]]:
        """Load file(s) from given path"""
        if self.is_dir:
            all_files = [f for f in self.path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
            txt_files = [f for f in all_files if f.suffix.lower() == ".txt"]
            docling_files = [f for f in all_files if f.suffix.lower() != ".txt"]

            contents = [(self._read_txt(f), f.name) for f in txt_files]
            contents.extend(self._convert_batch(docling_files))
            return contents

        if self.path.suffix.lower() == ".txt":
            return [(self._read_txt(self.path), self.path.name)]

        return self._convert_batch([self.path])

    def _read_txt(self, filepath: Path) -> str:
        """Read a plain text file directly."""
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        self.files[str(filepath)] = content
        self._save_markdown(filepath, content)
        return content

    def _convert_batch(self, filepaths: list[Path]) -> list[tuple[str, str]]:
        """Convert multiple files to markdown using docling's batch processing."""
        if not filepaths:
            return []

        results = []
        try:
            for conv_result in self._converter.convert_all(filepaths, raises_on_error=False):
                filepath = Path(conv_result.input.file)
                if conv_result.errors:
                    error_msgs = "; ".join(e.error_message for e in conv_result.errors)
                    raise FileStoreException(f"Failed to convert file: {filepath.name}: {error_msgs}")

                markdown_content = conv_result.document.export_to_markdown()
                self.files[str(filepath)] = markdown_content
                self._save_markdown(filepath, markdown_content)
                results.append((markdown_content, filepath.name))
        except FileStoreException:
            raise
        except Exception as exc:
            raise FileStoreException(f"Failed to convert files") from exc

        return results

    def _save_markdown(self, filepath: Path, content: str) -> None:
        """Save extracted content as a markdown file if save_dir is set."""
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            output_file = self.save_dir / f"{filepath.stem}.md"
            output_file.write_text(content, encoding="utf-8")
            logger.info("Saved extracted markdown to %s", output_file)
