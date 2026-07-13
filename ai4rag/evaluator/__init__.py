# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from ai4rag import logger
from ai4rag.evaluator.base_evaluator import BaseEvaluator
from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator

try:
    from ai4rag.evaluator.llmaj_evaluator import LLMaJConfig, LLMaJEvaluator
except ImportError as exc:
    logger.info(
        "LLM-as-a-Judge evaluator is unavailable (%s). "
        "Install optional dependencies with: pip install ai4rag[llm-judge]",
        exc,
    )
