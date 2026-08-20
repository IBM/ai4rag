# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import logging
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from ogx_client import OgxClient
from mlflow.genai.optimize.optimizers import GepaPromptOptimizer
from mlflow.genai.optimize import MetaPromptOptimizer

from ai4rag import handler
from ai4rag.components.utils.docling_io import load_docling_documents
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.mps import ModelsPreSelector
from ai4rag.rag.chunking.langchain_chunker import LangChainChunker
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
from ai4rag.search_space.prepare.prepare_search_space import prepare_search_space_with_ogx
from ai4rag.rag.retrieval.retriever import Retriever
from ai4rag.search_space.src.model_props import (
    QUESTION_PLACEHOLDER,
    REFERENCE_DOCUMENTS_PLACEHOLDER,
)
import os

_logger = logging.getLogger("search-space-preparation")
_logger.addHandler(handler)

_DEFAULT_METRIC = "faithfulness"
_DEFAULT_TOP_N_GENERATION = 3
_DEFAULT_TOP_K_EMBEDDING = 2
_DEFAULT_SAMPLE_SIZE = 5
_DEFAULT_SEED = 17


def _serialize_model(model: BaseFoundationModel | BaseEmbeddingModel) -> dict[str, Any]:
    """Convert a model instance to a plain dictionary with all its settings.

    Captures model identifier, type discriminator, inference parameters,
    and — for foundation models — the detected language.
    """
    is_embedding = isinstance(model, BaseEmbeddingModel)

    params = model.params
    if is_dataclass(params):
        params_dict = {
            field.name: getattr(params, field.name)
            for field in fields(params)
            if getattr(params, field.name) is not None
        }
    elif hasattr(params, "model_dump"):
        params_dict = params.model_dump()
    elif hasattr(params, "dict"):
        params_dict = params.dict()
    else:
        params_dict = {}

    result: dict[str, Any] = {
        "model_id": model.model_id,
        "type": "embedding" if is_embedding else "generation",
        "params": params_dict,
    }

    if not is_embedding:
        if hasattr(model, "language") and model.language is not None:
            result["language"] = model.language.to_dict()
        result["system_message_text"] = model.system_message_text
        result["user_message_text"] = model.user_message_text
        result["context_template_text"] = model.context_template_text

    return result


def precompute_retrieval_context(
    benchmark_sample: BenchmarkData,
    retriever: Retriever,
    context_template_text: str,
) -> list[dict[str, Any]]:
    """Build mlflow-compatible training data with pre-computed retrieval context.

    For each benchmark question, retrieves relevant chunks via *retriever*,
    formats them into a ``reference_documents`` string using
    *context_template_text*, and returns the result as a list of dicts
    suitable for ``mlflow.genai.optimize_prompts(train_data=...)``.

    Parameters
    ----------
    benchmark_sample
        A random sample of the benchmark data (questions + expected answers).
    retriever
        A fully initialised retriever backed by an indexed vector store.
    context_template_text
        Per-document formatting template (e.g. ``"Document {doc_number}:\\n{document}"``).

    Returns
    -------
    list[dict[str, Any]]
        Each element has ``{"inputs": {"question": ..., "reference_documents": ...}, "outputs": ...}``.
    """
    train_data = []
    for question, correct_answers in zip(benchmark_sample.questions, benchmark_sample.correct_answers):
        chunks = retriever.retrieve(question)
        reference_documents = "\n\n".join(
            context_template_text.format(document=chunk.text, doc_number=i) for i, chunk in enumerate(chunks, start=1)
        )
        train_data.append(
            {
                "inputs": {
                    QUESTION_PLACEHOLDER: question,
                    REFERENCE_DOCUMENTS_PLACEHOLDER: reference_documents,
                },
                "outputs": correct_answers[0] if isinstance(correct_answers, list) else correct_answers,
            }
        )
    return train_data


def optimize_prompts_for_models(
    foundation_models: list[BaseFoundationModel],
    train_data: list[dict[str, Any]],
    reflection_model: str,
    maas_client: OgxClient,
) -> None:
    """Optimize system/user prompts for each foundation model via mlflow.

    Registers each model's prompts as an mlflow chat prompt, runs
    ``mlflow.genai.optimize_prompts`` with ``GepaPromptOptimizer``, and
    mutates the model objects in-place with the optimised prompt texts.

    Parameters
    ----------
    foundation_models
        Foundation models whose prompts should be optimised.
    train_data
        Pre-computed training data as returned by
        :func:`precompute_retrieval_context`.
    reflection_model
        Model URI for the GEPA reflection LLM
        (e.g. ``"openai/ibm/granite-3-8b-instruct"``).
    maas_client
        An authenticated MaaS :class:`~openai.OpenAI` client.
    """
    original_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri("sqlite://")

    optimizer = GepaPromptOptimizer(reflection_model=reflection_model)

    try:
        for model in foundation_models:
            prompt_name = model.model_id.replace("/", "-")

            mlflow_user_message = model.user_message_text
            for placeholder in (QUESTION_PLACEHOLDER, REFERENCE_DOCUMENTS_PLACEHOLDER):
                mlflow_user_message = mlflow_user_message.replace(f"{{{placeholder}}}", f"{{{{{placeholder}}}}}")

            prompt = mlflow.genai.register_prompt(
                name=prompt_name,
                template=[
                    {"role": "system", "content": model.system_message_text},
                    {"role": "user", "content": mlflow_user_message},
                ],
            )

            def make_predict_fn(fm: BaseFoundationModel, uri: str):
                def predict_fn(**kwargs) -> str:
                    loaded = mlflow.genai.load_prompt(uri)
                    messages = loaded.format(**kwargs)
                    response = fm.chat(messages=messages)
                    return response[0]["content"]

                return predict_fn

            result = mlflow.genai.optimize_prompts(
                predict_fn=make_predict_fn(model, prompt.uri),
                train_data=train_data,
                prompt_uris=[prompt.uri],
                optimizer=optimizer,
                enable_tracking=False,
            )

            optimized_template = result.optimized_prompts[0].to_single_brace_format()
            for msg in optimized_template:
                if msg["role"] == "system":
                    model.system_message_text = msg["content"]
                elif msg["role"] == "user":
                    model.user_message_text = msg["content"]

            _logger.info(
                "Optimised prompts for model '%s' (score: %s -> %s).",
                model.model_id,
                result.initial_eval_score,
                result.final_eval_score,
            )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)


def outer_predict_function(fm, retriever):

    def mlflow_predict_function(question: str):
        system_prompt = mlflow.load_prompt("default_system")
        user_prompt = mlflow.load_prompt("default_user")
        context_prompt = mlflow.load_prompt("default_context")

        # extract ref docs and create `context`
        reference_documents = retriever.retrieve(question)

        contexts = []
        for numb, chunk in enumerate(reference_documents, start=1):
            contexts.append(context_prompt.format(doc_number=numb, document=chunk))

        # contexts = "\n\n".join(
        #     [
        #         context_prompt.format(doc_number=numb, document=chunk)
        #         for numb, chunk in enumerate(reference_documents, start=1)
        #     ]
        # )

        msgs = [
            {"role": "system", "content": system_prompt.format()},
            {
                "role": "user",
                "content": user_prompt.format(question=question, reference_documents="\n\n".join(contexts)),
            },
        ]

        res = fm.chat(messages=msgs)

        return res[0].message.content

    return mlflow_predict_function


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
    """

    search_space: dict[str, Any]
    selected_models: dict[str, list]

    def save_json(self, path: str | Path) -> None:
        """Serialize the report to a JSON file.

        The file is suitable as input for the RAG optimization step.

        Parameters
        ----------
        path
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.search_space, f, indent=2)


def prepare_search_space_report(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    test_data_path: str | Path,
    extracted_text_path: str | Path,
    ogx_client: OgxClient,
    embedding_models: list[str] | None = None,
    generation_models: list[str] | None = None,
    top_n_generation: int = _DEFAULT_TOP_N_GENERATION,
    top_k_embedding: int = _DEFAULT_TOP_K_EMBEDDING,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    random_seed: int = _DEFAULT_SEED,
    chunking_methods: list[str] | None = None,
    chunk_sizes: list[int] | None = None,
    chunk_overlaps: list[int] | None = None,
    inference_max_threads: int = 10,
    optimize_prompts: bool = True,
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
    top_n_generation
        Maximum number of generation models to retain.
    top_k_embedding
        Maximum number of embedding models to retain.
    sample_size
        Number of benchmark records sampled for model pre-selection.
    random_seed
        Seed for reproducible sampling.
    chunking_methods
        When provided, constrains the ``chunking_method`` dimension of the
        search space to only these methods (e.g. ``["recursive"]`` or
        ``["hybrid"]``).  ``None`` uses the platform defaults (both
        ``"recursive"`` and ``"hybrid"``).
    chunk_sizes
        When provided, constrains the ``chunk_size`` dimension of the
        search space to only these sizes (e.g. ``[256, 512]``).  ``None``
        uses the platform defaults.
    chunk_overlaps
        When provided, constrains the ``chunk_overlap`` dimension of the
        search space to only these values (e.g. ``[0, 128]``).  ``None``
        uses the platform defaults.
    inference_max_threads
        Maximum number of concurrent threads used when querying the
        RAG service during benchmark evaluation.  Lower values reduce
        per-request concurrency (useful when each request carries more
        retrieved context).  Defaults to ``10``.
    reflection_model
        Model URI for the GEPA reflection LLM used during prompt
        optimisation (e.g. ``"openai/ibm/granite-3-8b-instruct"``).
        When ``None`` (default), prompt optimisation is skipped.

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
    pydantic.ValidationError
        If *chunking_methods*, *chunk_sizes*, or *chunk_overlaps* fail structural
        validation (wrong type, empty list, or invalid element types).
    SearchSpaceValueError
        If *chunking_methods* contains values not in
        :attr:`~ai4rag.utils.constants.ChunkingConstraints.METHODS`, or
        *chunk_sizes* contains values outside
        ``[ChunkingConstraints.MIN_CHUNK_SIZE, ChunkingConstraints.MAX_CHUNK_SIZE]``,
        or *chunk_overlaps* contains values outside
        ``[ChunkingConstraints.MIN_CHUNK_OVERLAP, ChunkingConstraints.MAX_CHUNK_OVERLAP]``.
    """
    _validate_model_list(embedding_models, "embedding_models")
    _validate_model_list(generation_models, "generation_models")

    # Build payload and create search space via OGX
    payload: dict[str, Any] = {}
    if generation_models:
        payload["foundation_models"] = [{"model_id": gm} for gm in generation_models]
    if embedding_models:
        payload["embedding_models"] = [{"model_id": em} for em in embedding_models]
    if chunking_methods is not None:
        payload["chunking_methods"] = chunking_methods
    if chunk_sizes is not None:
        payload["chunk_sizes"] = chunk_sizes
    if chunk_overlaps is not None:
        payload["chunk_overlaps"] = chunk_overlaps

    # Load benchmark data and documents
    benchmark_df = pd.read_json(Path(test_data_path))
    benchmark_data = BenchmarkData(benchmark_df)
    documents = load_docling_documents(extracted_text_path)

    search_space = prepare_search_space_with_ogx(
        payload,
        client=ogx_client,
        benchmark_data=benchmark_df,
    )
    _logger.info(
        "Search space chunking_method=%s chunk_size=%s chunk_overlap=%s",
        list(search_space["chunking_method"].values),
        list(search_space["chunk_size"].values),
        list(search_space["chunk_overlap"].values),
    )

    # Run model pre-selection when the number of models exceeds the caps
    fm_values = search_space["foundation_model"].values
    em_values = search_space["embedding_model"].values

    mps: ModelsPreSelector | None = None
    if len(fm_values) > top_n_generation or len(em_values) > top_k_embedding:
        mps = ModelsPreSelector(
            benchmark_data=benchmark_data.get_random_sample(n_records=sample_size, random_seed=random_seed),
            documents=documents,
            foundation_models=search_space._search_space["foundation_model"].values,  # pylint: disable=protected-access
            embedding_models=search_space._search_space["embedding_model"].values,  # pylint: disable=protected-access
            max_threads=inference_max_threads,
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

    if optimize_prompts:
        os.environ["OPENAI_API_BASE"] = f"{ogx_client.base_url}/v1"
        os.environ["OPENAI_API_KEY"] = ogx_client.api_key

        selected_fm = selected_models["foundation_model"]
        selected_em = selected_models["embedding_model"]

        if mps is not None:
            retriever = mps.retrievers[selected_em[0].model_id]
        else:
            document_ids = []
            sample_for_docs = benchmark_data.get_random_sample(n_records=sample_size, random_seed=random_seed)
            for element in sample_for_docs.document_ids:
                document_ids.extend(element)
            sample_docs = [d for d in documents if d.name in document_ids]
            chunker = LangChainChunker(chunk_size=512, chunk_overlap=128, method="recursive")
            chunks = chunker.split_documents(sample_docs)
            vector_store = ModelsPreSelector._create_vector_store(
                selected_em[0], chunks, collection_name="ai4rag_prompt_opt"
            )
            retriever = Retriever(vector_store, number_of_chunks=3, method="simple", search_mode="vector")

        sample = benchmark_data.get_random_sample(n_records=sample_size, random_seed=random_seed)
        # prompt_train_data = precompute_retrieval_context(
        #     benchmark_sample=sample,
        #     retriever=retriever,
        #     context_template_text=selected_fm[0].context_template_text,
        # )

        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        user_prompt = mlflow.genai.register_prompt(
            name="default_user",
            template=selected_fm[0].user_message_text,
        )

        system_prompt = mlflow.genai.register_prompt(
            name="default_system",
            template=selected_fm[0].system_message_text,
        )

        context_template = mlflow.genai.register_prompt(
            name="default_context", template=selected_fm[0].context_template_text
        )

        meta_opt = MetaPromptOptimizer(reflection_model=f"openai:/{selected_fm[0].model_id}")
        result = mlflow.genai.optimize_prompts(
            predict_fn=outer_predict_function(fm=selected_fm[0], retriever=None),
            prompt_uris=[user_prompt.uri, system_prompt.uri, context_template.uri],
            optimizer=meta_opt,
            train_data=[],
            enable_tracking=False,
            scorers=[],
        )

        selected_fm[0].user_message_text = result.optimized_prompts[0].template
        selected_fm[0].system_message_text = result.optimized_prompts[1].template
        selected_fm[0].context_template_text = result.optimized_prompts[2].template

        # optimize_prompts_for_models(
        #     foundation_models=selected_fm,
        #     train_data=prompt_train_data,
        #     reflection_model=reflection_model,
        #     maas_client=maas_client,
        # )

    # Build verbose representation from valid (rule-filtered) combinations only
    valid_combinations = search_space.combinations
    if not valid_combinations:
        _logger.warning("No valid combinations remain after applying search space rules.")
    non_model_keys = [p.name for p in search_space.params if p.name not in ("foundation_model", "embedding_model")]
    verbose_repr: dict[str, Any] = {
        key: list(dict.fromkeys(combo[key] for combo in valid_combinations)) for key in non_model_keys
    }
    verbose_repr["foundation_model"] = [_serialize_model(m) for m in selected_models["foundation_model"]]
    verbose_repr["embedding_model"] = [_serialize_model(m) for m in selected_models["embedding_model"]]

    return SearchSpaceReport(
        search_space=verbose_repr,
        selected_models=selected_models,
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
