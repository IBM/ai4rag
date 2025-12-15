from dataclasses import fields
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ai4rag import logger
from ai4rag.search_space.prepare.input_payload_types import AI4RAGConstraints
from ai4rag.search_space.prepare.validation_error_decoder import validation_error_decoder
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace


__all__ = ["prepare_ai4rag_search_space"]


def prepare_ai4rag_search_space(payload: dict[str, Any]) -> AI4RAGSearchSpace:
    """
    Prepare AutoRAGSearchSpace.

    Parameters
    ----------
    payload : dict[str, Any]
        A mapping between parameter name and its associated values.

    Returns
    -------
    AI4RAGSearchSpace
        A valid AI4RAGSearchSpace used in RAG optimization process.

    Raises
    ------
    SearchSpaceValueError
        Raised when payload contains non-recognized parameter name.
    """
    logger.info("Preparing search space based on provided constraints: %s.", payload)
    payload_model = TypeAdapter(AI4RAGConstraints)

    try:
        validated_payload = payload_model.validate_python(
            payload,
        )

    except ValidationError as ve:
        # we want to catch only the first error
        validation_error_decoder(ve.errors()[0])

    params = []
    for field in fields(validated_payload):
        if (val := getattr(validated_payload, field.name)) is not None:
            params.append(val)

    return AI4RAGSearchSpace(
        params=params,
    )
