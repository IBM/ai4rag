# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

print("Loading local .env settings")
load_dotenv(find_dotenv(".env"))

BASE_DIR = Path(__file__).parents[1]
