# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import logging
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml as yml
from ogx_client import OgxClient

from ai4rag import handler
from ai4rag.components.utils.docling_io import load_docling_documents
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.mps import ModelsPreSelector
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
from ai4rag.search_space.prepare.prepare_search_space import prepare_search_space_with_ogx

_logger = logging.getLogger("search-space-preparation")
_logger.addHandler(handler)

SUPPORTED_METRICS = ("faithfulness", "answer_correctness", "context_correctness")

_DEFAULT_METRIC = "faithfulness"
_DEFAULT_TOP_N_GENERATION = 3
_DEFAULT_TOP_K_EMBEDDING = 2
_DEFAULT_SAMPLE_SIZE = 5
_DEFAULT_SEED = 17

LANGUAGE_MAP: dict[str, str] = {
    "ja": "Japanese",
    "ko": "Korean",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "cs": "Czech",
    "tr": "Turkish",
}


def _represent_model_instance(dumper: yml.Dumper, model: BaseFoundationModel | BaseEmbeddingModel) -> yml.Node:
    """Instruct :mod:`yaml` on how to serialize model instances under a ``!Model`` tag.

    The resulting YAML mapping contains the model identifier as key with its
    parameters as value, plus a ``type_`` discriminator (``"embedding"`` or
    ``"generation"``).
    """
    type_ = "embedding" if isinstance(model, BaseEmbeddingModel) else "generation"

    params = model.params
    if is_dataclass(params):
        params = {
            field.name: getattr(model.params, field.name)
            for field in fields(model.params)
            if getattr(model.params, field.name)
        }
    elif hasattr(params, "model_dump"):
        params = params.model_dump(exclude_unset=True)
    elif hasattr(params, "dict"):
        params = params.dict(exclude_unset=True)

    return dumper.represent_mapping("!Model", {model.model_id: params or {}, "type_": type_})


# Register the multi-representer so SafeDumper can handle any subclass.
yml.add_multi_representer(BaseFoundationModel, _represent_model_instance, Dumper=yml.SafeDumper)
yml.add_multi_representer(BaseEmbeddingModel, _represent_model_instance, Dumper=yml.SafeDumper)


@dataclass
class SearchSpaceReport:
    """Result of the search-space preparation step.

    Attributes
    ----------
    search_space : dict[str, Any]
        Verbose representation of the search space, including selected
        model lists and non-model parameter ranges.
    selected_models : dict[str, list]
        Foundation and embedding model lists that survived pre-selection.
    detected_language : dict[str, str] | None
        Detected language code and name, or ``None`` when English or when
        detection was not performed.
    """

    search_space: dict[str, Any]
    selected_models: dict[str, list]
    detected_language: dict[str, str] | None

    def save_yaml(self, path: str | Path) -> None:
        """Serialize the report to a YAML file.

        The file is suitable as input for the RAG optimization step.

        Parameters
        ----------
        path
            Destination file path.
        """
        report = dict(self.search_space)
        if self.detected_language:
            report["detected_language"] = self.detected_language

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yml.safe_dump(report, f)


def prepare_search_space_report(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    test_data_path: str | Path,
    extracted_text_path: str | Path,
    ogx_client: OgxClient,
    embedding_models: list[str] | None = None,
    generation_models: list[str] | None = None,
    metric: str = _DEFAULT_METRIC,
    top_n_generation: int = _DEFAULT_TOP_N_GENERATION,
    top_k_embedding: int = _DEFAULT_TOP_K_EMBEDDING,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    random_seed: int = _DEFAULT_SEED,
) -> SearchSpaceReport:
    """Run model pre-selection and prepare a search-space report.

    Builds an :class:`AI4RAGSearchSpace` from the given model lists, runs
    :class:`ModelsPreSelector` when the number of models exceeds the
    configured caps, detects the benchmark language, and returns a
    structured report.

    Parameters
    ----------
    test_data_path
        Path to a JSON file containing benchmark questions and expected
        answers.
    extracted_text_path
        Path to a single DoclingDocument JSON file or a directory of such
        files.
    ogx_client
        An authenticated :class:`OgxClient` instance.
    embedding_models
        Embedding model identifiers.  ``None`` uses the server defaults.
    generation_models
        Generation model identifiers.  ``None`` uses the server defaults.
    metric
        Quality metric for intermediate pattern evaluation.  Must be one
        of ``"faithfulness"``, ``"answer_correctness"``, or
        ``"context_correctness"``.
    top_n_generation
        Maximum number of generation models to retain.
    top_k_embedding
        Maximum number of embedding models to retain.
    sample_size
        Number of benchmark records sampled for model pre-selection.
    random_seed
        Seed for reproducible sampling.

    Returns
    -------
    SearchSpaceReport
        Structured report containing the verbose search space, selected
        models, and detected language.

    Raises
    ------
    ValueError
        If *metric* is not one of the supported values.
    TypeError
        If *embedding_models* or *generation_models* contain invalid entries.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Metric {metric!r} is not supported. Supported metrics are {list(SUPPORTED_METRICS)}.")

    _validate_model_list(embedding_models, "embedding_models")
    _validate_model_list(generation_models, "generation_models")

    # Build payload and create search space via OGX
    payload: dict[str, list[dict[str, str]]] = {}
    if generation_models:
        payload["foundation_models"] = [{"model_id": gm} for gm in generation_models]
    if embedding_models:
        payload["embedding_models"] = [{"model_id": em} for em in embedding_models]

    search_space = prepare_search_space_with_ogx(payload, client=ogx_client)

    # Load benchmark data and documents
    benchmark_df = pd.read_json(Path(test_data_path))
    detected_language = _detect_benchmark_language(
        benchmark_df, llm_client=ogx_client, generation_models=generation_models
    )

    benchmark_data = BenchmarkData(benchmark_df)
    documents = load_docling_documents(extracted_text_path)

    # Run model pre-selection when the number of models exceeds the caps
    fm_values = search_space["foundation_model"].values
    em_values = search_space["embedding_model"].values

    if len(fm_values) > top_n_generation or len(em_values) > top_k_embedding:
        mps = ModelsPreSelector(
            benchmark_data=benchmark_data.get_random_sample(n_records=sample_size, random_seed=random_seed),
            documents=documents,
            foundation_models=search_space._search_space["foundation_model"].values,  # pylint: disable=protected-access
            embedding_models=search_space._search_space["embedding_model"].values,  # pylint: disable=protected-access
            metric=metric,
        )
        mps.evaluate_patterns()
        selected = mps.select_models(
            n_embedding_models=top_k_embedding,
            n_foundation_models=top_n_generation,
        )
        selected_models = {
            "foundation_model": selected["foundation_models"],
            "embedding_model": selected["embedding_models"],
        }
    else:
        selected_models = {
            "foundation_model": list(fm_values),
            "embedding_model": list(em_values),
        }

    # Build verbose representation
    verbose_repr: dict[str, Any] = {
        k: v.all_values()
        for k, v in search_space._search_space.items()  # pylint: disable=protected-access
        if k not in ("foundation_model", "embedding_model")
    }
    verbose_repr.update(selected_models)

    return SearchSpaceReport(
        search_space=verbose_repr,
        selected_models=selected_models,
        detected_language=detected_language,
    )


def _validate_model_list(models: list[str] | None, name: str) -> None:
    """Validate that a model list, if provided, contains only non-empty strings."""
    if models is None:
        return
    if not isinstance(models, list):
        raise TypeError(f"{name} must be a list.")
    for i, m in enumerate(models):
        if not m:
            raise TypeError(f"{name}[{i}] must be a non-empty string.")


def _detect_language_via_llm(  # pylint: disable=too-many-locals
    questions: list[str],
    llm_client: OgxClient,
    allowed_generation_models: list[str] | None = None,
) -> dict[str, str] | None:
    """Detect the dominant language from sample questions using an LLM.

    Sends a small sample of questions to a generation model registered in OGX
    and asks it to return the ISO 639-1 code.  Models listed in
    *allowed_generation_models* are preferred when available.

    Parameters
    ----------
    questions
        Raw question texts to classify.  Only the first five are sent to the
        model.
    llm_client
        An authenticated :class:`OgxClient` instance.
    allowed_generation_models
        Optional whitelist of model identifiers to prefer.

    Returns
    -------
    dict[str, str] | None
        A dictionary with ``code`` and ``name`` keys when a non-English
        language is detected, or ``None`` for English / on failure.
    """
    sample_text = "\n".join(f"- {q}" for q in questions[:5])
    valid_codes = ", ".join(sorted(LANGUAGE_MAP.keys()))

    try:
        models_response = llm_client.models.list()
        models_list = models_response.data if hasattr(models_response, "data") else list(models_response)
        registered_ids = {(m.identifier if hasattr(m, "identifier") else str(m.id)) for m in models_list}

        model_id: str | None = None
        if allowed_generation_models:
            for gm in allowed_generation_models:
                if gm in registered_ids:
                    model_id = gm
                    break
        if not model_id:
            for m in models_list:
                if hasattr(m, "model_type") and getattr(m, "model_type", "") == "llm":
                    model_id = m.identifier if hasattr(m, "identifier") else str(m.id)
                    break
        if not model_id:
            _logger.warning("No models available for LLM language detection.")
            return None

        response = llm_client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a language detection assistant. "
                        "Given text samples, respond with ONLY the ISO 639-1 language code "
                        f"(one of: {valid_codes}). "
                        "Nothing else — just the code."
                    ),
                },
                {
                    "role": "user",
                    "content": f"What language are these questions written in?\n{sample_text}",
                },
            ],
            max_completion_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip().lower().replace('"', "").replace("'", "")
        detected_code = raw.split()[0] if raw else None

        if not detected_code:
            return None

        name = LANGUAGE_MAP.get(detected_code)
        if not name:
            _logger.warning("LLM returned unsupported language code: %s", detected_code)
            return None

        _logger.info("Language detected via LLM: %s (%s)", detected_code, name)
        return {"code": detected_code, "name": name}

    except Exception as exc:
        _logger.warning("LLM language detection failed: %s", exc)
        return None


def _detect_benchmark_language(
    benchmark_df: pd.DataFrame,
    llm_client: OgxClient,
    generation_models: list[str] | None = None,
    sample_size: int = 10,
) -> dict[str, str] | None:
    """Detect the dominant language from benchmark question data.

    Extracts up to *sample_size* questions from the ``question`` column and
    delegates to :func:`detect_language_via_llm` for classification.

    Parameters
    ----------
    benchmark_df
        DataFrame with a ``question`` column.
    llm_client
        An authenticated :class:`OgxClient` instance.
    generation_models
        Optional whitelist of model identifiers passed through to the LLM
        detection step.
    sample_size
        Maximum number of questions to sample.

    Returns
    -------
    dict[str, str] | None
        A dictionary with ``code`` and ``name`` keys when a non-English
        language is detected, or ``None`` for English / on failure.
    """
    questions = benchmark_df["question"].dropna().astype(str).tolist()
    if not questions:
        return None

    sample = questions[:sample_size]
    return _detect_language_via_llm(sample, llm_client, allowed_generation_models=generation_models)
