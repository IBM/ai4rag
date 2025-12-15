from typing import Literal

from ai4rag.utils.constants import ConstantMeta

__all__ = ["EmbeddingModels", "FoundationModels"]

DEFAULT_MAX_SEQUENCE_LENGTH = 8192


class Models(metaclass=ConstantMeta):
    """Parent class from models classes that is an instance of ConstantMeta."""

    @classmethod
    def get_models(cls) -> list[str]:
        """
        Get list of supported models from class attributes.

        Returns
        -------
        list[str]
            Supported models
        """
        # pylint: disable=no-member
        models = list(el for el in cls._constant_attributes)

        return models


class EmbeddingModels(Models):
    """
    Available embedding models on watsonx.ai. Similar structure is supported
    by FoundationModelsManager.EmbeddingModels(), but this is hardcoded here, so
    we do not need to create api client for reading models.

    List of supported embedding models can be found here:
    https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models-embed.html?context=wx
    """

    IBM_SLATE_125M_ENG = "ibm/slate-125m-english-rtrvr"
    IBM_SLATE_125M_ENG_V2 = "ibm/slate-125m-english-rtrvr-v2"
    IBM_GRANITE_278M_MULTILINGUAL = "ibm/granite-embedding-278m-multilingual"
    MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large"

    @classmethod
    def get_max_tokens(cls, model_id: str) -> int:
        """
        This function returns the maximum number of tokens that can be passed
        to the embed function for a given model ID. If no match is found in the map,
        a default value of 512 is returned.

        Parameters
        ----------
        model_id : str
            The model ID for which to retrieve the maximum number of input tokens.

        Returns
        -------
        int
            The maximum number of input tokens for the given model ID. Default is 512.
        """
        max_input_tokens_map = {
            cls.IBM_SLATE_125M_ENG: 512,
            cls.IBM_SLATE_125M_ENG_V2: 512,
            cls.IBM_GRANITE_278M_MULTILINGUAL: 512,
            cls.MULTILINGUAL_E5_LARGE: 512,
        }

        return max_input_tokens_map.get(model_id, 512)

    @classmethod
    def get_distance_metric(cls, model_id: str) -> Literal["cosine", "euclidean"]:
        """
        This function returns the appropriate distance metric to be used for
         constructing a vector store and retrieving passages for given model id.

        Parameters
        ----------
        model_id : str
            The model ID for which to retrieve the maximum number of input tokens.

        Returns
        -------
        str
            Distance metric to be used for constructing a vector store and retrieving.
        """
        distance_metric_map = {
            cls.IBM_SLATE_125M_ENG: "cosine",
            cls.MULTILINGUAL_E5_LARGE: "cosine",
            cls.IBM_SLATE_125M_ENG_V2: "cosine",
            cls.IBM_GRANITE_278M_MULTILINGUAL: "cosine",
        }
        return distance_metric_map.get(model_id, "cosine")


class FoundationModels(Models):
    """Available foundation models."""

    MISTRAL_SMALL_3_1_24B_INSTRUCT = "mistralai/mistral-small-3-1-24b-instruct-2503"
    MISTRAL_MISTRAL_LARGE = "mistralai/mistral-large"
    MISTRAL_MEDIUM_2505 = "mistralai/mistral-medium-2505"
    GRANITE_3_8B_INSTRUCT = "ibm/granite-3-8b-instruct"
    GRANITE_3_3_8B_INSTRUCT = "ibm/granite-3-3-8b-instruct"
    GRANITE_4H_SMALL = "ibm/granite-4-h-small"
    META_LLAMA_3_1_70B_INSTRUCT = "meta-llama/llama-3-1-70b-instruct"
    META_LLAMA_3_1_8B_INSTRUCT = "meta-llama/llama-3-1-8b-instruct"
    META_LLAMA_3_3_70B_INSTRUCT = "meta-llama/llama-3-3-70b-instruct"
    META_LLAMA_4_MAVERICK_17B_128E_INSTRUCT_FP8 = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    OPENAI_GPT_OSS_120B = "openai/gpt-oss-120b"

    @classmethod
    def get_default_max_sequence_length(cls, model_id: str, default: int = DEFAULT_MAX_SEQUENCE_LENGTH) -> int:
        """
        Get default max sequence length specific to model. This data should be
        updated as this is some mapping made based on the models hosted on watsonx.
        It is introduced as a response to unstable endpoints containing 'model_limits'.

        Parameters
        ----------
        model_id : str
            Model ID

        default : int, default=8192
            Default value to be returned when no value is set for the model.

        Returns
        -------
        int
            Max sequence length for a specific model. Default is 8192.
        """

        defaults = {
            cls.MISTRAL_MISTRAL_LARGE: 128_000,
            cls.MISTRAL_SMALL_3_1_24B_INSTRUCT: 131_072,
            cls.MISTRAL_MEDIUM_2505: 131_072,
            cls.META_LLAMA_3_1_70B_INSTRUCT: 128_000,
            cls.META_LLAMA_3_1_8B_INSTRUCT: 128_000,
            cls.META_LLAMA_3_3_70B_INSTRUCT: 131_072,
            cls.META_LLAMA_4_MAVERICK_17B_128E_INSTRUCT_FP8: 131_072,
            cls.GRANITE_3_8B_INSTRUCT: 131_072,
            cls.GRANITE_3_3_8B_INSTRUCT: 131_072,
            cls.OPENAI_GPT_OSS_120B: 131_072,
            cls.GRANITE_4H_SMALL: 131_072,
        }

        return defaults.get(model_id, default)
