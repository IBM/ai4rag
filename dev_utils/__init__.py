# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

print("Loading local .env settings")
load_dotenv(find_dotenv(".env"))

BASE_DIR = Path(__file__).parents[1]
