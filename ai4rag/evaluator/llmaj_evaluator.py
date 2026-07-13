# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import os
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ai4rag import logger
from ai4rag.evaluator.base_evaluator import BaseEvaluator, EvaluationData, MetricType
from ai4rag.evaluator.score_utils import compute_confidence_interval

try:
    from openai import OpenAI, OpenAIError
except ImportError as exc:
    raise ImportError(
        "openai package is required for LLM-as-a-Judge evaluation. Install with: pip install ai4rag[llm-judge]"
    ) from exc


@dataclass
class LLMaJConfig:
    """
    Configuration for the LLM-as-a-Judge evaluator.

    Parameters
    ----------
    base_url : str
        Base URL of the OpenAI-compatible API endpoint (e.g. OGX ``.../v1``).

    api_key : str
        API key for authentication.

    model : str
        Model name as reported by the endpoint (e.g. ``"llama-31-8b-instruct"``).

    temperature : float
        Temperature for the judge model.
    """

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url is required for LLMaJConfig.")
        if not self.api_key.strip():
            raise ValueError("api_key is required for LLMaJConfig.")
        if not self.model.strip():
            raise ValueError("model is required for LLMaJConfig.")


JUDGE_PROMPT_TEMPLATE = """\
You are an impartial judge evaluating the quality of an AI assistant's response.

## Task
{guidelines}

## Context
Question: {question}

## Response to evaluate
{answer}

## Instructions
Respond with ONLY a JSON object (no markdown, no extra text):
{{"score": <integer 1-5>, "rationale": "<brief explanation>"}}

Where:
- 1 = completely fails the criterion
- 2 = mostly fails with some relevant elements
- 3 = partially meets the criterion
- 4 = mostly meets with minor gaps
- 5 = fully meets the criterion
"""


def _llmaj_log_io_enabled() -> bool:
    """Return whether judge prompt/response bodies should be logged."""
    value = os.getenv("AI4RAG_LLMAJ_LOG_IO", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


class LLMaJEvaluator(BaseEvaluator):
    """
    Evaluator that scores ``answer_relevance`` with an LLM judge via an OpenAI-compatible API.

    All scores are normalized from the judge scale (1-5) to [0.0, 1.0].
    """

    METRIC_GUIDELINES = {
        MetricType.ANSWER_RELEVANCE: (
            "Evaluate whether the response directly and helpfully addresses the user's question. "
            "Consider relevance, helpfulness, accuracy, and whether the answer stays on topic."
        ),
    }

    def __init__(self, config: LLMaJConfig):
        self.config = config
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def evaluate_metrics(
        self,
        evaluation_data: list[EvaluationData],
        metrics: Sequence[str],
    ) -> dict:
        """Evaluate responses with the configured judge model."""
        scores: dict[str, dict[str, float | None]] = {}
        question_scores: dict[str, dict[str, float | None]] = {}
        question_ids = [ed.question_id or str(i) for i, ed in enumerate(evaluation_data)]

        for metric_name in metrics:
            guidelines = self.METRIC_GUIDELINES.get(metric_name)
            if guidelines is None:
                continue

            row_scores: list[float] = []
            question_scores[metric_name] = {}
            for i, ed in enumerate(evaluation_data):
                qid = question_ids[i]
                normalized = self._judge_row(ed, guidelines)
                question_scores[metric_name][qid] = round(normalized, 4) if normalized is not None else None
                if normalized is not None:
                    row_scores.append(normalized)

            ci_low, ci_high = compute_confidence_interval(row_scores)
            scores[metric_name] = {
                "mean": round(float(np.mean(row_scores)), 4) if row_scores else None,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }

        return {"scores": scores, "question_scores": question_scores}

    def _judge_row(self, evaluation_data: EvaluationData, guidelines: str) -> float | None:
        """Score a single row with the judge model."""
        question_id = evaluation_data.question_id or "unknown"
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            guidelines=guidelines,
            question=evaluation_data.question or "",
            answer=evaluation_data.answer or "",
        )
        if _llmaj_log_io_enabled():
            logger.info(
                "LLM judge request [model=%s question_id=%s]\n--- PROMPT ---\n%s\n--- END PROMPT ---",
                self.config.model,
                question_id,
                prompt,
            )
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=256,
            )
            content = response.choices[0].message.content.strip()
            parsed_score = _parse_score(content)
            normalized = _normalize_score(parsed_score)
            if _llmaj_log_io_enabled():
                logger.info(
                    "LLM judge response [model=%s question_id=%s raw_score=%s normalized=%s]\n"
                    "--- RESPONSE ---\n%s\n--- END RESPONSE ---",
                    self.config.model,
                    question_id,
                    parsed_score,
                    normalized,
                    content,
                )
            return normalized
        except (OpenAIError, ValueError, KeyError, AttributeError, IndexError) as exc:
            if _llmaj_log_io_enabled():
                logger.warning(
                    "LLM judge call failed [model=%s question_id=%s]: %s",
                    self.config.model,
                    question_id,
                    exc,
                )
            return None

    def get_supported_metrics(self) -> list[str]:
        """Return metric names supported by this evaluator."""
        return list(self.METRIC_GUIDELINES.keys())


def _extract_json(content: str) -> dict | None:
    """Extract a JSON object from LLM output that may contain markdown fences or extra text."""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    brace_match = re.search(r"\{[^{}]*\}", content)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _parse_score(content: str) -> int | None:
    """Parse a score (1-5) from the judge's JSON response."""
    data = _extract_json(content)
    if data is None:
        return None
    try:
        score = int(data["score"])
        if 1 <= score <= 5:
            return score
    except (KeyError, ValueError, TypeError):
        pass
    return None


def _normalize_score(score: int | None) -> float | None:
    """Normalize a score from 1-5 to [0.0, 1.0]."""
    if score is None:
        return None
    return (score - 1) / 4
