# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from pathlib import Path
from typing import Sequence
from functools import lru_cache
import io

from pypdf import PdfReader
from langchain_core.documents import Document


def _txt_to_string(binary_data: bytes) -> str:
    return binary_data.decode("utf-8", errors="ignore")


def _pdf_to_string(binary_data: bytes) -> str:
    with io.BytesIO(binary_data) as open_pdf_file:
        reader = PdfReader(open_pdf_file)
        full_text = [page.extract_text() for page in reader.pages]
        return "\n".join(full_text)


class FileStoreException(Exception):
    pass


class FileStore:
    """
    Class used to load locally saved input files.
    This class reused logics from the TextLoader class in ibm_watsonx_ai package
    """

    suffix_to_func = {
        ".txt": _txt_to_string,
        ".pdf": _pdf_to_string,
    }

    def __init__(self, path: str | Path | Sequence[str] | Sequence[Path]):
        self.path = Path(path)
        self.is_dir = path.is_dir()
        self.files = {}

    def __repr__(self) -> str:
        ret = f"{self.__class__.__name__}(path={self.path})"
        return ret

    def load_as_documents(self) -> list[Document]:
        """Read files as langchain documents"""
        contents = self._load_content()
        documents = [Document(page_content=content[0], metadata={"document_id": content[1]}) for content in contents]

        return documents

    @lru_cache(maxsize=2)
    def _load_content(self) -> list[tuple[str, str]]:
        """Load file(s) from given path"""
        if self.is_dir:
            contents = [(self._read_single_content(file), file.name) for file in self.path.iterdir()]
            return contents

        return [(self._read_single_content(self.path), self.path.name)]

    def _process_file(self, filepath: Path, file_content: bytes | None) -> str:
        """
        Extracts text from bytes for various file types.

        Parameters
        ----------
        filepath : Path
            Path to file

        file_content : bytes | None
            File content as bytes

        Returns
        -------
        str
            Extracted file text
        """

        handler = self.suffix_to_func.get(filepath.suffix, None)

        try:
            text = handler(file_content)
        except Exception as exc:
            raise FileStoreException(f"Failed to load file.") from exc

        return text

    def _read_single_content(self, filepath: Path) -> str:
        """Read single file"""
        with open(filepath, "rb") as file:
            content = file.read()
        text = self._process_file(filepath=filepath, file_content=content)
        self.files[str(filepath)] = text
        return text
