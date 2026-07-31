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
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".md",
    ".html",
    ".txt",
    ".odt",
    ".odp",
    ".adoc",
    ".tex",
    ".epub",
    ".eml",
    ".qmd",
    ".rmd",
    ".xhtml",
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
}

_DEFAULT_CACHE_DIR = Path(__file__).parent / "local" / "docling_cache"


class FileStoreException(Exception):
    pass


class FileStore:
    """
    Class used to load locally saved input files.
    Uses docling library to extract content from documents.
    """

    def __init__(
        self,
        path: str | Path | Sequence[str] | Sequence[Path],
        save_dir: str | Path | None = None,
        cache_dir: str | Path | None = _DEFAULT_CACHE_DIR,
    ):
        """
        Parameters
        ----------
        path : str | Path | Sequence[str] | Sequence[Path]
            Path to a single file or a directory of files.
        save_dir : str | Path | None
            Optional directory to save extracted markdown files. If None, no files are saved.
        cache_dir : str | Path | None
            Directory for caching DoclingDocument JSON files. Set to None to disable caching.
            Defaults to ``dev_utils/local/docling_cache``.
        """
        self.path = Path(path)
        self.is_dir = self.path.is_dir()
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.files = {}

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.accelerator_options = AcceleratorOptions(device="auto")

        num_workers = os.cpu_count() or 1
        settings.perf.doc_batch_size = num_workers
        settings.perf.doc_batch_concurrency = num_workers

        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"

    def load_as_documents(self) -> list[DoclingDocument]:
        """Load files as ``DoclingDocument`` objects preserving document structure.

        Returns
        -------
        list[DoclingDocument]
            Parsed documents with full structural representation.
        """
        return self._load_docling_documents()

    @lru_cache(maxsize=2)
    def _load_docling_documents(self) -> list[DoclingDocument]:
        """Load files as ``DoclingDocument`` objects."""
        if self.is_dir:
            all_files = [f for f in self.path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
            txt_files = [f for f in all_files if f.suffix.lower() == ".txt"]
            docling_files = [f for f in all_files if f.suffix.lower() != ".txt"]

            docs = [self._txt_to_docling_document(f) for f in txt_files]
            docs.extend(self._convert_batch_as_docling(docling_files))
            return docs

        if self.path.suffix.lower() == ".txt":
            return [self._txt_to_docling_document(self.path)]

        return self._convert_batch_as_docling([self.path])

    def _cache_path_for(self, filepath: Path) -> Path | None:
        """Return the JSON cache path for a source file, or None if caching is disabled."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{filepath.name}.json"

    def _load_from_cache(self, filepath: Path) -> DoclingDocument | None:
        """Load a ``DoclingDocument`` from the JSON cache if it exists."""
        cache_path = self._cache_path_for(filepath)
        if cache_path is None or not cache_path.exists():
            return None

        try:
            doc = DoclingDocument.model_validate_json(cache_path.read_bytes())
            logger.info("Loaded from cache: %s", cache_path.name)
            return doc
        except Exception:
            logger.warning("Corrupted cache file %s — will re-convert", cache_path.name)
            return None

    def _save_to_cache(self, filepath: Path, doc: DoclingDocument) -> None:
        """Persist a ``DoclingDocument`` as JSON to the cache directory."""
        cache_path = self._cache_path_for(filepath)
        if cache_path is None:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(doc.model_dump_json(), encoding="utf-8")
        logger.info("Cached DoclingDocument: %s", cache_path.name)

    @staticmethod
    def _txt_to_docling_document(filepath: Path) -> DoclingDocument:
        """Wrap a plain text file into a ``DoclingDocument``."""
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        doc = DoclingDocument(name=filepath.name)
        doc.add_text(label=DocItemLabel.PARAGRAPH, text=content)
        return doc

    def _convert_batch_as_docling(self, filepaths: list[Path]) -> list[DoclingDocument]:
        """Convert multiple files to ``DoclingDocument`` objects.

        Checks the JSON cache first; only files without a cached representation
        are sent through the docling converter.
        """
        if not filepaths:
            return []

        cached: dict[str, DoclingDocument] = {}
        to_convert: list[Path] = []

        for fp in filepaths:
            doc = self._load_from_cache(fp)
            if doc is not None:
                cached[str(fp)] = doc
            else:
                to_convert.append(fp)

        if to_convert:
            logger.info("Converting %d file(s) (cache hit for %d)", len(to_convert), len(cached))
        elif cached:
            logger.info("All %d file(s) loaded from cache", len(cached))

        results: list[DoclingDocument] = []
        try:
            for conv_result in self._converter.convert_all(to_convert, raises_on_error=False):
                filepath = Path(conv_result.input.file)
                if conv_result.errors:
                    error_msgs = "; ".join(e.error_message for e in conv_result.errors)
                    raise FileStoreException(f"Failed to convert file: {filepath.name}: {error_msgs}")

                doc = conv_result.document
                doc.name = filepath.name

                self._save_markdown(filepath, doc.export_to_markdown())
                self._save_to_cache(filepath, doc)
                results.append(doc)
        except FileStoreException:
            raise
        except Exception as exc:
            raise FileStoreException("Failed to convert files") from exc

        converted_by_name = {r.name: r for r in results}
        ordered = [cached.get(str(fp)) or converted_by_name[fp.name] for fp in filepaths]
        return ordered

    def _save_markdown(self, filepath: Path, content: str) -> None:
        """Save extracted content as a markdown file if save_dir is set."""
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            output_file = self.save_dir / f"{filepath.stem}.md"
            output_file.write_text(content, encoding="utf-8")
            logger.info("Saved extracted markdown to %s", output_file)
