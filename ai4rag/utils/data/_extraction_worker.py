# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Fresh-interpreter conversion worker; its parent supplies the native limits."""

import os
import sys
from multiprocessing.connection import Connection

from ._extraction_pool import _WORKER_MEMORY_EXIT_CODE


def main(descriptor: int) -> None:
    """Initialize one converter and process sequential requests from its owner."""
    from ai4rag.utils.data.text_extraction import (
        DoclingExtractionConfig,
        _text_extraction_pool_initializer,
        _worker_process_document,
    )

    with Connection(descriptor) as connection:
        config = DoclingExtractionConfig(**connection.recv())
        _text_extraction_pool_initializer(config)
        connection.send(("ready",))
        while (task := connection.recv()) is not None:
            success, error = _worker_process_document(*task)
            # Keep control messages small; the full traceback is logged locally.
            connection.send((success, error[-8000:] if error else None))


if __name__ == "__main__":
    try:
        main(int(sys.argv[1]))
    except MemoryError:
        # Exit rather than continuing with potentially compromised native state.
        # Avoid native finalizers delaying or masking the dedicated exit status.
        os._exit(_WORKER_MEMORY_EXIT_CODE)
