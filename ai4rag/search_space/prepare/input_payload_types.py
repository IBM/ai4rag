#
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
#
from dataclasses import asdict, fields
from functools import partial
from string import Formatter
from typing import Annotated, Any, Literal, Optional, Self, Sequence, TypeVar, cast

from annotated_types import Ge, Gt, Le, Len, MinLen
from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    SkipValidation,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass

from ai4rag import logger
from ai4rag.search_space.src.exceptions import SearchSpaceValueError
from ai4rag.search_space.src.model_props import (
    CONTEXT_TEXT_PLACEHOLDER,
    QUESTION_PLACEHOLDER,
    REFERENCE_DOCUMENTS_PLACEHOLDER,
    get_context_template_text,
    get_system_message_text,
    get_user_message_text,
)
from ai4rag.search_space.src.models import FoundationModels, EmbeddingModels
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.utils.constants import (
    ChatGenerationConstants,
    ChunkingConstraints,
    RetrievalConstraints,
)

# pylint: disable=invalid-name
config = ConfigDict(extra="forbid")


@dataclass(config=config)
class AI4RAGChunking:
    """Attributes to be included in the chunking payload."""

    method: Optional[Literal["recursive", "semantic"]] = None  # To-do should not be Optional (not acc. to API)
    chunk_size: Optional[
        Annotated[
            int,
            Ge(ChunkingConstraints.MIN_CHUNK_SIZE),
            Le(ChunkingConstraints.MAX_CHUNK_SIZE),
        ]
    ] = None
    chunk_overlap: Optional[
        Annotated[
            int,
            Ge(ChunkingConstraints.MIN_CHUNK_OVERLAP),
            Le(ChunkingConstraints.MAX_CHUNK_OVERLAP),
        ]
    ] = None

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        """
        Ensures that chunk_overlap is lower than chunk_size.

        Raises
        -------
        SearchSpaceValueError
            When chunk overlap is higher than chunk size.

        Returns
        -------
        Self
            Requirement enforced by Pydantic itself when function is wrapped with model_validator.
        """
        if self.chunk_overlap is not None and self.chunk_size is not None and self.chunk_overlap >= self.chunk_size:
            raise SearchSpaceValueError(
                "Incorrect 'chunking' payload. 'chunking.chunk_overlap' must be less than 'chunking.chunk_size'."
            )
        return self

    def __hash__(self) -> int:
        return hash((self.method, self.chunk_size, self.chunk_overlap))


@dataclass(config=config)
class AI4RAGRetrieval:
    """Attributes to be included in the retrieval payload."""

    method: Optional[Literal["simple", "window"]] = None  # TO-DO not make it optional
    window_size: Optional[
        Annotated[
            int,
            Ge(RetrievalConstraints.MIN_WINDOW_SIZE),
            Le(RetrievalConstraints.MAX_WINDOW_SIZE),
        ]
    ] = None
    number_of_chunks: Optional[
        Annotated[
            int,
            Ge(RetrievalConstraints.MIN_NUMBER_OF_RETRIEVED_CHUNKS),
            Le(RetrievalConstraints.MAX_NUMBER_OF_RETRIEVED_CHUNKS),
        ]
    ] = None

    @model_validator(mode="after")
    def validate_retrieval(self) -> Self:
        """
        Validates if user provided window_size is:
            - greater than 0 for `simple` retrieval method
            - not equal to 0 for `window` retrieval method.

        Raises
        ------
        SearchSpaceValueError
            When either of the above requirements is not met.

        Returns
        -------
        Self
            Requirement enforced by Pydantic itself when function is wrapped with model_validator.
        """
        if self.method is not None and self.window_size is not None:
            if self.method == "simple" and self.window_size > 0:
                raise SearchSpaceValueError(
                    "'window size' cannot be larger than 0 when retrieval method is set to 'simple'."
                )  # works bc it's subclass of ValueError
            if self.method == "window" and self.window_size == 0:
                raise SearchSpaceValueError(
                    "'window size' cannot be lower than 1 when retrieval method is set to 'window'."
                )
        return self

    def __hash__(self) -> int:
        return hash((self.method, self.window_size, self.number_of_chunks))


@dataclass(frozen=True, config=config)
class Language:
    """Attributes to be included in the generation.language payload."""

    auto_detect: bool = True


@dataclass(config=config)
class AI4RAGModelParams:
    """Attributes to be included in the generation.foundation_models payload."""

    max_completion_tokens: Annotated[int, Gt(0)] = ChatGenerationConstants.MAX_COMPLETION_TOKENS
    temperature: Annotated[float, Ge(0), Le(1)] = ChatGenerationConstants.TEMPERATURE

    def to_dict(self) -> dict[str, Any]:
        """Cast instance of AI4RAGModelParams class as dict."""

        return asdict(self)


def expected_template_placeholders_exist(template_str: str, info: ValidationInfo) -> str:
    """
    Validates if user provided correct placeholders in given template text in respect to default placeholders.

    Returns
    -------
    str
        Prompt template with filled placeholders.

    Raises
    ------
    SearchSpaceValueError
        When user provided less placeholders than expected.

        When user provided wrong placeholder name.
    """
    if info.field_name == "context_template_text":
        required_placeholders = (CONTEXT_TEXT_PLACEHOLDER,)
    elif info.field_name == "user_message_text":
        required_placeholders = (QUESTION_PLACEHOLDER, REFERENCE_DOCUMENTS_PLACEHOLDER)
    else:
        raise ValueError(f"Cannot validate presence of expected template placeholders on field: {info.field_name}")

    placeholders_count = 0

    for _, field_name, _, _ in Formatter().parse(template_str):
        if field_name is None:
            # when there is text NOT followed by a placeholder template
            continue
        if field_name not in required_placeholders:
            raise SearchSpaceValueError(
                f"Custom {info.field_name.split('_')[0]} template text got unexpected placeholder `{field_name}`, "
                f"valid placeholders are `{required_placeholders}`."
            )

        placeholders_count += 1

    if placeholders_count != len(required_placeholders):
        raise SearchSpaceValueError(
            f"Incorrect number of placeholders required for {info.field_name.split('_')[0]} template text, "
            f"expected {len(required_placeholders)} but got {placeholders_count}."
        )
    return template_str


@dataclass(config=config)
class AI4RAGChatTemplateMessages:
    """Attributes to be included in the generation.foundation_models[n].chat_template_messages."""

    system_message_text: Optional[str] = None
    user_message_text: Optional[Annotated[str, AfterValidator(expected_template_placeholders_exist)]] = None

    @model_validator(mode="after")
    def validate_both_messages_set(self):
        """
        Check whether both prompts are set and raise error if not.
        It overrides default validation message that is less transient.
        """
        if self.system_message_text is None or self.user_message_text is None:
            raise SearchSpaceValueError(
                "When specifying chat_template_messages both user_message_text and "
                "system_message_text need to be provided."
            )

        return self


@dataclass(config=config)  # should be frozen dataclass if we make it hashable
class AI4RAGModel:
    """Attributes to be included in the generation.foundation_models payload."""

    model_id: Annotated[str, MinLen(1)]
    language: Language = Language()
    parameters: Optional[AI4RAGModelParams] = Field(default_factory=AI4RAGModelParams)
    # cannot set a default value at this point as we need for that `language.auto_detect` which is
    # defined on `Generation` object (inaccessible at this point)
    chat_template_messages: Optional[AI4RAGChatTemplateMessages] = None
    context_template_text: Annotated[str, AfterValidator(expected_template_placeholders_exist)] = Field(
        default_factory=lambda data: get_context_template_text(data.get("model_id", ""))
    )
    word_to_token_ratio: Optional[Annotated[float, Gt(0)]] = None
    # This value is set during model validation
    max_sequence_length: SkipValidation[Optional[int]] = Field(
        default_factory=lambda data: FoundationModels.get_default_max_sequence_length(data.get("model_id", ""))
    )

    @field_validator("model_id", mode="before")
    @classmethod
    def validate_model_id(cls, model_id: str) -> str:
        """
        Validate whether model is in the list of supported models.

        Parameters
        ----------
        model_id : str
            Model identifier.

        Returns
        -------
        str
            Validated model identifier.

        Raises
        ------
        SearchSpaceValueError
            When model_id is not contained within FoundationModels.
        """

        if model_id not in FoundationModels:
            raise SearchSpaceValueError(f"Foundation model: {model_id} is no supported in ai4rag.")

        return model_id

    @model_validator(mode="after")
    def validate_chat_template_messages(self) -> Self:
        """Create default chat template messages."""

        if self.chat_template_messages is None:
            self.chat_template_messages = AI4RAGChatTemplateMessages(
                system_message_text=get_system_message_text(self.model_id),
                user_message_text=get_user_message_text(self.model_id, language_autodetect=self.language.auto_detect),
            )

        return self

    def get_id_as_dict(self) -> dict[Literal["model_id"], str]:
        """Represents model identifier as dict."""

        return {"model_id": self.model_id}

    def __eq__(self, other: str | Self) -> bool:
        if isinstance(other, str) and self.model_id == other:  # let's delete that
            return True

        if isinstance(other, self.__class__):
            return fields(self) == fields(other) and [getattr(self, f.name) for f in fields(self)] == [
                getattr(other, f.name) for f in fields(other)
            ]
        return NotImplemented

    def __str__(self) -> str:
        return self.model_id

    def __hash__(self) -> int:
        # not good choice base on the __eq__ definition.
        # Will generate frequent hash conflicts which deletes the purpose of hashing
        return hash(self.model_id)


@dataclass(config=config)
class Generation:
    """Attributes describing the schema for Generation object from OpenAPI."""

    foundation_models: Annotated[list[AI4RAGModel], MinLen(1)]


ELEMENT = TypeVar(
    "ELEMENT",
    bound=str | AI4RAGChunking | AI4RAGRetrieval | Literal["simple", "window"],
)

ATOMIC = TypeVar("ATOMIC", bound=Generation | None)


@dataclass(config=config)
class AI4RAGConstraints:
    """Attributes to be included in constraints payload."""

    embedding_models: Optional[Annotated[list[Annotated[str, MinLen(1)]], MinLen(1)]] = None
    retrieval: Optional[Annotated[list[AI4RAGRetrieval], Len(1, 4)]] = None
    chunking: Optional[Annotated[list[AI4RAGChunking], Len(1, 4)]] = None
    generation: Generation = None

    @field_validator("*", mode="after")  # * for all the attributes
    @classmethod
    def validate_duplicates(cls, attribute: list[ELEMENT] | ATOMIC) -> list[ELEMENT] | ATOMIC:
        """
        Deduplicates each attribute.

        Field validator iteratively visits all the attributes one by one.
        It is triggering right after the pydantic validation hence each attribute has
        their corresponding type while visited by this method.

        Parameters
        ----------
        attribute : list[ITERABLES] | ATOMIC
            When attribute belongs to ITERABLES type then it means that it is the list of those types.
            If attribute has ATOMIC type then it means that it accepts single object.

            The 'attribute' has the type of currently visited attribute.
            For example if currently visited attribute is `embedding_models`
            then the attribute has list[str] type.

        Returns
        -------
        list[ITERABLES] | ATOMIC
            Type of input attribute.
        """
        if attribute is None:  # if `None` is passed explicitly
            return attribute

        if isinstance(attribute, Generation):
            # TBD It could be done better, but for now we don't support
            # multiple configurations for the same foundation model
            # dump python (attribute.foundation_models), iterate over
            # TO-DO need work on __hash__ implementation for Model classes and AI4RAGParams as well
            attribute.foundation_models = list(set(attribute.foundation_models))
            return attribute

        if len((unique_attribute_values := list(set(attribute)))) != len(attribute):
            return unique_attribute_values
        return attribute

    @field_validator("embedding_models", mode="before")
    @classmethod
    def validate_embedding_models(cls, embedding_models: list[str]) -> list[str]:
        """
        Validates user provided embedding models.

        Parameters
        ----------
        embedding_models : list[str]
            Pydantic initially-validated list of embedding models for further validation.

        Returns
        -------
        list[str]
            List of chosen embedding models based on environment's availability.
        """
        not_supported_em = [em for em in embedding_models if em not in EmbeddingModels]
        if not_supported_em:
            raise SearchSpaceValueError(f"Embedding models: {not_supported_em} are not supported.")

        return embedding_models


    @model_validator(mode="after")
    def cast_to_parameter(self):
        """
        After the payload is successfully validated all fields (constraints) should be cast to `Parameter` type which
        is used internally to represent ai4rag search space's constraints.

        Notes
        -----
        Bc this validator coerces pydantic types to our internal `Parameter` one it needs to run as the last in queue.
        """
        for field in fields(self):
            constraint = getattr(self, field.name)
            if constraint is None:
                continue
            match constraint_name := field.name:
                case "retrieval" | "chunking":
                    ta = TypeAdapter(constraint[0].__class__)  # adapter for AI4RAGRetrieval or AI4RAGChunking
                    settings = list(map(partial(ta.dump_python, exclude_none=True), constraint))
                    setattr(self, constraint_name, Parameter(constraint_name, param_type="C", values=settings))

                case "embedding_models":
                    setattr(self, "embedding_models", Parameter("embedding_model", param_type="C", values=constraint))

                case "retrieval_methods":
                    setattr(
                        self,
                        constraint_name,
                        Parameter("retrieval", param_type="C", values=[{"method": meth} for meth in constraint]),
                    )

                case "generation":
                    setattr(
                        self,
                        constraint_name,
                        Parameter("inference_model_id", param_type="C", values=constraint.foundation_models),
                    )

        return self
