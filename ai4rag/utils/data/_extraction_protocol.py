# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Framed messages over an extraction worker's private socket."""

import json
import os
import socket
import struct
from typing import Any

_HEADER = struct.Struct("!I")
_MAX_MESSAGE_BYTES = 128 * 1024
_INCOMPLETE = object()


def _send_message(connection: socket.socket, message: Any, *, timeout: float | None = None) -> None:
    """Send one message with an optional deadline for the entire write."""
    payload = json.dumps(message, default=os.fspath).encode("utf-8")
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError("Text extraction control message is too large.")
    connection.settimeout(timeout)
    try:
        connection.sendall(_HEADER.pack(len(payload)) + payload)
    finally:
        connection.settimeout(None)


def _read_message(connection: socket.socket, buffer: bytearray, *, nonblocking: bool = False) -> Any:
    """Read a frame, retaining partial data between nonblocking reads."""
    flags = socket.MSG_DONTWAIT if nonblocking else 0
    while True:
        remaining = _HEADER.size - len(buffer)
        if remaining <= 0:
            (size,) = _HEADER.unpack_from(buffer)
            if not 0 < size <= _MAX_MESSAGE_BYTES:
                raise ValueError("Invalid text extraction control message length.")
            remaining = _HEADER.size + size - len(buffer)
            if remaining == 0:
                message = json.loads(buffer[_HEADER.size :])
                buffer.clear()
                return message

        try:
            data = connection.recv(remaining, flags)
        except BlockingIOError:
            return _INCOMPLETE
        if not data:
            raise EOFError("Text extraction control socket closed.")
        buffer.extend(data)
