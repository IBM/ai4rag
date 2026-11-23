# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Fresh-interpreter conversion worker; its parent supplies the native limits."""

import os
import socket
import sys

from ._extraction_pool import _WORKER_MEMORY_EXIT_CODE
from ._extraction_protocol import _read_message, _send_message


def main(descriptor: int) -> None:
    """Initialize one converter and process sequential requests from its owner."""
    from ai4rag.utils.data.text_extraction import (
        DoclingExtractionConfig,
        _text_extraction_pool_initializer,
        _worker_process_document,
    )

    with socket.socket(fileno=descriptor) as connection:
        requests = bytearray()
        config = DoclingExtractionConfig(**_read_message(connection, requests))
        _text_extraction_pool_initializer(config)
        _send_message(connection, ("ready",))
        while (task := _read_message(connection, requests)) is not None:
            success, error = _worker_process_document(*task)
            # Keep control messages small; the full traceback is logged locally.
            _send_message(connection, (success, error[-8000:] if error else None))


if __name__ == "__main__":
    try:
        main(int(sys.argv[1]))
    except MemoryError:
        # Exit rather than continuing with potentially compromised native state.
        # Avoid native finalizers delaying or masking the dedicated exit status.
        try:
            os.write(2, b"Text extraction worker caught MemoryError.\n")
        finally:
            os._exit(_WORKER_MEMORY_EXIT_CODE)
