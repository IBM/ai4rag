# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from pathlib import Path

from docling_core.types.doc.document import DoclingDocument


def load_docling_documents(path: str | Path) -> list[DoclingDocument]:
    """Load :class:`DoclingDocument` instances from JSON files.

    Parameters
    ----------
    path
        A local path to either a single ``DoclingDocument`` JSON file or a
        directory containing such files.  Only files with a ``.json``
        extension are loaded from directories.

    Returns
    -------
    list[DoclingDocument]
        Loaded documents, sorted by filename when *path* is a directory.
    """
    path = Path(path)

    if path.is_dir():
        return [DoclingDocument.load_from_json(p) for p in sorted(path.rglob("*.json")) if p.is_file()]

    if path.is_file():
        return [DoclingDocument.load_from_json(path)]

    return []
