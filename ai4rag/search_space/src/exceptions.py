# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
class SearchSpaceError(Exception):
    """Error representing problem with search space initialisation"""


class SearchSpaceValueError(ValueError):
    """Error representing incorrect value given for search space type"""
