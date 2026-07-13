# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from ai4rag.components.optimization.judge_selection import calibration_subset_size


def test_calibration_subset_size():
    assert calibration_subset_size(5) == 1
    assert calibration_subset_size(100) == 10
    assert calibration_subset_size(500) == 20
