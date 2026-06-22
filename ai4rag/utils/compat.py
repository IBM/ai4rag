# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import sys


def ensure_sqlite3() -> None:
    """Patch stdlib ``sqlite3`` with ``pysqlite3-binary`` if available.

    ChromaDB requires sqlite3 >= 3.35. On platforms with an older system
    sqlite (e.g. RHEL 9), this function swaps the stdlib module with the
    ``pysqlite3`` wheel so that ChromaDB (and LangChain-Chroma) can work.

    Safe to call multiple times — the patch is idempotent.
    """
    if "pysqlite3" in sys.modules or getattr(sys.modules.get("sqlite3"), "__name__", None) == "pysqlite3":
        return
    try:
        import pysqlite3  # type: ignore[import-untyped]

        sys.modules["sqlite3"] = pysqlite3
    except ImportError:
        pass
