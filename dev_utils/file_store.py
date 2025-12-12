#
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
#
from pathlib import Path
from typing import Sequence
from functools import lru_cache

from ibm_watsonx_ai.data_loaders.text_loader import TextLoader
from ibm_watsonx_ai.wml_client_error import WMLClientError
from langchain_core.documents import Document


class FileStoreException(Exception):
    pass


class FileStore:
    """
    Class used to load locally saved input files.
    This class reused logics from the TextLoader class in ibm_watsonx_ai package
    """

    file_type_handlers = {
        "text/plain": TextLoader._txt_to_string,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": TextLoader._docs_to_string,
        "application/pdf": TextLoader._pdf_to_string,
        "text/html": TextLoader._html_to_string,
        "text/markdown": TextLoader._md_to_string,
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
        try:
            file_type = TextLoader.identify_file_type(filename=filepath.suffix)
        except WMLClientError:
            raise FileStoreException(f"Not supported file type: {filepath.suffix}")

        handler = self.file_type_handlers.get(file_type, None)

        try:
            text = handler(file_content)
        except Exception as exception:
            raise FileStoreException(f"Failed to load file.") from exception

        return text

    def _read_single_content(self, filepath: Path) -> str:
        """Read single file"""
        with open(filepath, "rb") as file:
            content = file.read()
        text = self._process_file(filepath=filepath, file_content=content)
        self.files[str(filepath)] = text
        return text


if __name__ == "__main__":
    """Example usage with folder full of .txt docs"""
    _filepath = Path(__file__)
    documents_path = _filepath.parents[1] / "test_docs" / "documents" / "watson_x_ai"
    file_store = FileStore(documents_path)
    documents = file_store._load_content()
