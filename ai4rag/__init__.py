# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import logging
import os

from ai4rag.version import __version__


logger = logging.getLogger("ai4rag")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
