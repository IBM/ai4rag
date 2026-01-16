# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import ast
import inspect
import tempfile
import textwrap
from typing import Any, Callable, Literal, cast

import black
from ai4rag.core.ai_service.nodes.generate import chat_node
from ai4rag.core.ai_service.nodes.retrieve import multi_index_retrieve_node, retrieve_node
from ai4rag.core.ai_service.states import AI4RAGState
from ai4rag.search_space.src.models import EmbeddingModels, FoundationModels

from .function_transformer import FunctionTransformer, FunctionVisitor, _get_components_replace_data

# import template functions
from .templates.sequential_inference_service import inference_service as sequential_inference_service
from .templates.utils_templates import (
    elasticsearch_vector_store_initialization,
    indexing_recursive_chunking,
    indexing_recursive_chunking_imports,
    indexing_semantic_chunking,
    indexing_semantic_chunking_imports,
    milvus_vector_store_initialization,
    vector_store_initialization,
)

VECTOR_STORE_INITIALIZATION_MAPPING = {
    "VectorStore": vector_store_initialization,
    "MilvusVectorStore": milvus_vector_store_initialization,
    "ElasticsearchVectorStore": elasticsearch_vector_store_initialization,
}


DEFAULT_GENERATE_SQL_MODEL_ID = FoundationModels.GRANITE_3_3_8B_INSTRUCT


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
class RAGService:
    """Initialize ``RAGService`` object.

    Parameters
    ----------
    api_client : APIClient
        Initialized APIClient object.

    model : ModelInference
        Instance allowing user communication with foundation models deployed
        on the chosen watsonx.ai environment.

    context_template_text : str
        Template for a single context chunk. It needs to contain {document}
        placeholder.

    agent : Literal["sequential", "react"]
        Agent architecture.

    system_message_text : str
        System part of the prompt used for models supporting chat.

    user_message_text : str
        Message template for the user part of the message.

    default_max_sequence_length : int | None, default=None
        Max number of tokens that model can accept. This value is used as fallback when
         max_sequence_length is not found in model_limits. It is used for prompt
        building.

    retrievers : list[BaseRetriever] | None, default=None
        Retrievers to retrieve context for answer generation from.

    vector_stores : list[BaseVectorStore] | None, default=None
        Vector stores to retrieve context for answer generation from.

    chunker : LangChainChunker | HybridSemanticChunker | None, default=None
        Instance used for code generation in index building services.
        Required for chroma scenario to rebuild indexes on runtime.

    multi_index_reranking_number_chunks : int
        Number of chunks retrieved during re-ranking step in multi index scenario.

    word_to_token_ratio : float, dfault=1.5
        Used for better estimation of number of allowed characters in the prompt.

    input_data_references : list[dict] | None, default=None
        Chroma-specific. When provided, generated AI service will contain documents indexing section.
        Chroma is an in-memory DB and needs to be re-populated with documents every time the code runs.

    ranker_config : list[dict | None] | dict | None, default=None
        Used for configuration of Milvus ranker.

    Raises
    ------
    InvalidMultipleArguments
        When there are instances of Both VectorStore and Retriever passed.

    ValueError
        When 'ranker_config' is used for vector store other than 'MilvusVectorStore'.
    """

    QUESTION_PLACEHOLDER = "{question}"
    DOCUMENT_PLACEHOLDER = "{document}"
    REFERENCE_DOCUMENTS_PLACEHOLDER = "{reference_documents}"

    def __init__(
        self,
        *,
        agent: Literal["sequential"],
        context_template_text: str | None = None,
        system_message_text: str | None = None,
        user_message_text: str | None = None,
        default_max_sequence_length: int | None = None,
        chunker: "LangChainChunker | HybridSemanticChunker | None" = None,
        multi_index_reranking_number_chunks: int | None = None,
        word_to_token_ratio: float = 1.5,
        ### Scenario with Chroma
        input_data_references: list[dict] | None = None,
        ### Milvus ranker
        ranker_config: list[dict | None] | dict | None = None,
        ## multindex
    ) -> None:
        self.space_id = self.api_client.default_space_id
        self.project_id = self.api_client.default_project_id

        inference_service_templates = sequential_inference_service
        if not inference_service_templates:
            raise ValueError(f"Error during AI service generation. Incorrect agent architecture was given: {agent}.")

        # Data source / retrieval
        self.retrievers = retrievers
        self.vector_stores = vector_stores
        self._ranker_config = ranker_config
        self.multi_index_reranking_number_chunks = multi_index_reranking_number_chunks
        # Index rebuilding
        self.chunker = chunker
        self.input_data_references = input_data_references

        # Inference
        self.model = model

        self.default_max_sequence_length = default_max_sequence_length
        self.word_to_token_ratio = word_to_token_ratio
        self.system_message_text = system_message_text
        self.user_message_text = user_message_text
        self.context_template_text = context_template_text
        self.agent = agent

        if self.context_template_text:
            self._validate_template_text(self.context_template_text, [self.DOCUMENT_PLACEHOLDER])

        if vector_stores and not retrievers:
            self.retrievers = [Retriever(vector_store=vs) for vs in vector_stores]
        elif retrievers and not vector_stores:
            self.vector_stores = [r.vector_store for r in retrievers]
        else:
            raise InvalidMultipleArguments(params_names_list=["vector_stores", "retrievers", "databases"])

        datasource_types = [getattr(vs, "_datasource_type", "") for vs in self.vector_stores]
        if isinstance(self._ranker_config, list) and any(
            config is not None and not isinstance(vs, MilvusVectorStore)
            for config, vs in zip(self._ranker_config, self.vector_stores)
        ):
            raise ValueError("`ranker_config` is used only for `MilvusVectorStore`")

        if (
            self._ranker_config is not None
            and not isinstance(self._ranker_config, list)
            and any("milvus" not in t for t in datasource_types)
        ):
            raise ValueError("`ranker_config` is used only for `MilvusVectorStore`")

        self.vector_store_initialization_function = VECTOR_STORE_INITIALIZATION_MAPPING.get(
            self.vector_stores[0].__class__.__name__
        )

        if datasource_types[0] == "chroma" and input_data_references is not None and self.chunker is not None:
            self.indexing_imports, self.indexing_function = INDEXING_CODE_MAPPING.get(self.chunker.__class__)

        self.retrieve_node = retrieve_node

        self.inference_service_template, self.inference_service_info = inference_service_templates.get("single_index")

        self.generate_node = chat_node

        self.code: str | None = None

        self.ai_service = self._populate_params(_copy_function(self.inference_service_template))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.ai_service(*args, **kwargs)

    def _populate_params(self, function: Callable[..., Any]) -> Callable[..., Any]:
        """Populate AI service params by updating and overwriting.
        Method populates in template `inference_service` the placeholders that starts
        with `REPLACE_THIS_CODE_WITH_` using default_params.

        :param function: AI service function which placeholders should be populated
        :type function: Callable

        :return: function with params populated if signature matches
        :rtype: Callable
        """
        args_spec = inspect.getfullargspec(function)
        defaults: tuple | list = args_spec.defaults or []
        args = args_spec.args or []

        if args and args[-1] == "vector_store_settings":
            vectorstore_params = self.vector_stores[0].to_dict() or {}
            vector_store_settings = {
                "connection_id": vectorstore_params.get("connection_id"),
            }
            vs_datasource_type = vectorstore_params.get("datasource_type")
            if isinstance(vs_datasource_type, str):
                if vs_datasource_type.startswith("milvus"):
                    vector_store_settings["collection_name"] = vectorstore_params.get("collection_name")
                elif "elasticsearch" in vs_datasource_type:
                    vector_store_settings["index_name"] = vectorstore_params.get("index_name")

            if space_id := self.space_id:
                vector_store_settings["space_id"] = space_id
            else:
                vector_store_settings["project_id"] = self.project_id
            function.__defaults__ = (
                *defaults[:-1],
                vector_store_settings,
            )

        source = AIServices._populate_default_params(function)  # pylint: disable=protected-access

        tree = ast.parse(source)
        visitor = FunctionVisitor()
        visitor.visit(tree)

        func_def = visitor.function

        # Credentials params
        credentials_params = _get_components_replace_data(
            self._swap_apikey_for_token(self.api_client.credentials.to_dict()),
            Credentials.__init__,
            suffix="credentials",
        )

        # Force credentials params to use URL that does not contain 'private' part
        credentials_params["REPLACE_THIS_CODE_WITH_CREDENTIALS_URL"] = credentials_params[
            "REPLACE_THIS_CODE_WITH_CREDENTIALS_URL"
        ].replace("private.", "")

        # APIClient params, for Chroma scenario
        api_client_params = _get_components_replace_data(
            {
                "space_id": self.space_id,
                "project_id": self.project_id,
            },
            APIClient.__init__,
            suffix="api_client",
        )

        common_replace_data = dict(  # pylint: disable=use-dict-literal
            # For AST Call node
            **credentials_params,
            **api_client_params,
            # For AST Assign Node
            REPLACE_THIS_CODE_WITH_SPACE_ID={
                "value": self.space_id,
                "replace": True,
            },
            REPLACE_THIS_CODE_WITH_PROJECT_ID={
                "value": self.project_id,
                "replace": True,
            },
        )
        template_specific_replace_data = self._replace_data()
        replace_data = common_replace_data | template_specific_replace_data

        replacer = FunctionTransformer(cast(ast.FunctionDef, func_def), **replace_data)
        new_tree = replacer.visit(tree)
        ast.fix_missing_locations(new_tree)

        self.code = ast.unparse(new_tree)

        # If default values in inference service make the line too long
        # use shorter signature
        code_lines = self.code.split("\n")
        tmp_signature = "def default_service(context):"
        formatted_code = black.format_str(
            tmp_signature + "\n" + "\n".join(code_lines[1:]),
            mode=black.FileMode(),
        ).rstrip()

        self.code = code_lines[0] + "\n" + "\n".join(formatted_code.split("\n")[1:])

        with tempfile.NamedTemporaryFile(suffix="inference_service.py", delete=True) as tmp_file:
            tmp_file.write(self.code.encode())
            compiled_code = compile(new_tree, filename=tmp_file.name, mode="exec")
            namespace: dict = {}
            exec(compiled_code, namespace)  # pylint: disable=exec-used
            function = namespace[function.__name__]
        return function

    def _replace_data(self):
        # pylint: disable=unnecessary-lambda
        replace_data_mapping = {
            sql_loop_checker_inference_service: lambda: self._sql_inference_service_shared_replace_data()
            | self._sql_loop_checker_consts(),
            sql_react_inference_service: lambda: self._sql_inference_service_shared_replace_data(),
            sequential_inference_service: lambda: self._sequential_inference_service_replace_data()
            | self._non_sql_common_replace_data(),
            multi_index_inference_service: lambda: self._multi_index_inference_service_replace_data()
            | self._non_sql_common_replace_data(),
        }
        return replace_data_mapping.get(self.inference_service_template)()

    def _non_sql_common_replace_data(self):
        """
        Common placeholders to replace in non-sql ai service templates.
        """
        common_params = {
            "REPLACE_THIS_CODE_WITH_WORD_TO_TOKEN_RATIO": self.word_to_token_ratio,
            "REPLACE_THIS_CODE_WITH_RETRIEVE_NODE": inspect.getsource(self.retrieve_node),
            "REPLACE_THIS_CODE_WITH_RETRIEVE_NODE_NAME": self.retrieve_node.__name__,
            "REPLACE_THIS_CODE_WITH_GENERATE_NODE_SOURCE": inspect.getsource(self.generate_node),
            "REPLACE_THIS_CODE_WITH_GENERATE_NODE_NAME": self.generate_node.__name__,
            "REPLACE_THIS_CODE_WITH_AUTOAIRAG_STATE": inspect.getsource(AI4RAGState),
        }

        build_prompt_additional_kwargs = {
            "context_template_text": self.context_template_text,
        }

        build_prompt_additional_kwargs |= {
            "system_message_text": self.system_message_text,
            "prompt_template_text": self.user_message_text,
        }

        if self.model.model_id:
            build_prompt_additional_kwargs |= {
                "model_max_input_tokens": get_max_input_tokens(
                    model=self.model,
                    default_max_sequence_length=self.default_max_sequence_length,
                ),
            }
        else:
            build_prompt_additional_kwargs |= {"model_max_input_tokens": self.default_max_sequence_length}
        common_params["REPLACE_THIS_CODE_WITH_BUILD_PROMPT_KWARGS"] = build_prompt_additional_kwargs

        model_init_params = self.model.get_identifying_params() or {}
        common_params |= _get_components_replace_data(model_init_params, ModelInference.__init__, "model")

        return common_params

    def _sequential_inference_service_replace_data(self):
        """
        Placeholders to replace in sequential inference service template.
        """
        replace_data = {
            "REPLACE_THIS_CODE_WITH_RETRIEVER": {
                "value": (
                    self.retrievers[0].to_dict() | {"window_size": self.retrievers[0].window_size}
                    if self.retrievers[0].method == "window"
                    else self.retrievers[0].to_dict()
                ),
                "replace": True,
            },
            **(
                {
                    "REPLACE_THIS_CODE_WITH_INDEXING_CODE": self._get_function_source_code(self.indexing_function),
                    "REPLACE_THIS_CODE_WITH_INDEXING_IMPORTS": self._get_function_source_code(self.indexing_imports),
                }
                if getattr(self, "indexing_imports", None) and getattr(self, "indexing_function", None)
                else {}
            ),
            "REPLACE_THIS_CODE_WITH_VECTOR_STORE_ASSIGN": self._prepare_vs_init_params()[0],
            "REPLACE_THIS_CODE_WITH_VECTOR_STORE_CLASS_NAME": self.vector_stores[0].__class__.__name__,
            "REPLACE_THIS_CODE_WITH_VECTOR_STORE_INITIALIZATION": self._get_function_source_code(
                self.vector_store_initialization_function
            ),
            "REPLACE_THIS_CODE_WITH_RETRIEVE_PARAMS": (
                {
                    "ranker_type": next(iter(self._ranker_config.keys())),
                    "ranker_params": next(iter(self._ranker_config.values())),
                }
                if self._ranker_config
                else {}
            ),
        }
        if self.input_data_references:
            replace_data["REPLACE_THIS_CODE_WITH_INPUT_DATA_REFERENCES"] = self.input_data_references
        if self.chunker:
            replace_data |= self._prepare_chunker_init_params()

        return replace_data

    def _multi_index_inference_service_replace_data(self):
        """
        Placeholders to replace in multi index inference service template.
        """
        return {
            "REPLACE_THIS_CODE_WITH_RETRIEVERS": [
                r.to_dict() | {"window_size": r.window_size} for r in self.retrievers
            ],
            "REPLACE_THIS_CODE_WITH_NUMBER_OF_RETRIEVED_CHUNKS": self.multi_index_reranking_number_chunks,
            "REPLACE_THIS_CODE_WITH_VECTOR_STORE_ASSIGN": self._prepare_vs_init_params(),
            "REPLACE_THIS_CODE_WITH_VECTOR_STORES_INITIALIZATION": self.vector_stores_initialization_params,
            "REPLACE_THIS_CODE_WITH_RETRIEVE_PARAMS": [
                (
                    {
                        "ranker_type": list(ranker_config.keys())[0],
                        "ranker_params": list(ranker_config.values())[0],
                    }
                    if (ranker_config := self._ranker_config[i]) is not None and isinstance(vs, MilvusVectorStore)
                    else {}
                )
                for i, vs in enumerate(self.vector_stores)
            ],
        }

    def _prepare_vs_init_params(self) -> list:
        """
        Prepares vector store parameters to be injected into the templates.
        """
        # pylint: disable=too-many-nested-blocks
        vector_stores_init_params_list = [
            vs_as_dict for vs in self.vector_stores if (vs_as_dict := vs.to_dict()) is not None
        ]

        # vector_stores_init_params_list = [vs.to_dict() for vs in self.vector_stores]
        for vector_store_init_params in vector_stores_init_params_list:
            ## Remove credential, project/space id and verify fields from wx embeddings
            ## since they will be restored from APIClient instance
            wx_embeddings_import_string = "ibm_watsonx_ai.foundation_models.embeddings.embeddings"
            embeddings_init_params = (
                vector_store_init_params.get("embedding") or vector_store_init_params.get("embeddings") or {}
            )
            if wx_embeddings_import_string in embeddings_init_params.get("__module__", ""):
                embeddings_init_params.pop("credentials", None)
                embeddings_init_params.pop("project_id", None)
                embeddings_init_params.pop("space_id", None)
                embeddings_init_params.pop("verify", None)

            embedding_function = vector_store_init_params.get("embedding_function")

            if not embedding_function:
                continue

            if not isinstance(embedding_function, list) and wx_embeddings_import_string in (
                embedding_function.get("__module__", "")
            ):
                embeddings_init_params.pop("credentials", None)
                embeddings_init_params.pop("project_id", None)
                embeddings_init_params.pop("space_id", None)
                embeddings_init_params.pop("verify", None)
                continue

            for embeddings_init_params in embedding_function:
                if wx_embeddings_import_string in (embeddings_init_params.get("__module__", "")):
                    embeddings_init_params.pop("credentials", None)
                    embeddings_init_params.pop("project_id", None)
                    embeddings_init_params.pop("space_id", None)
                    embeddings_init_params.pop("verify", None)

        return vector_stores_init_params_list

    def _prepare_chunker_init_params(self):
        """
        Prepares chunking parameters to be injected into the templates.
        """
        chunker_init_params = self.chunker.to_dict()
        if not isinstance(self.chunker, HybridSemanticChunker):
            return _get_components_replace_data(
                chunker_init_params,
                LangChainChunker.__init__,
                "langchain_chunker",
            )
        new_chunker_init_params = _get_components_replace_data(
            chunker_init_params,
            HybridSemanticChunker.__init__,
            "hybrid_semantic_chunker",
        )
        new_chunker_init_params["REPLACE_THIS_CODE_WITH_HYBRID_SEMANTIC_CHUNKER_EMBEDDING_MODEL_NAME"] = {
            "value": getattr(getattr(self.chunker, "embeddings"), "model_id"),
            "replace": True,
        }
        new_chunker_init_params["REPLACE_THIS_CODE_WITH_HYBRID_SEMANTIC_CHUNKER_EMBEDDING_MODEL_PARAMS"] = {
            "value": {
                "truncate_input_tokens": EmbeddingModels.get_max_tokens(
                    getattr(getattr(self.chunker, "embeddings"), "model_id")
                ),
            },
            "replace": True,
        }
        return new_chunker_init_params

    @staticmethod
    def _validate_template_text(template_text: str, required_input_variables: list[str]) -> None:
        """Check if template text has required input variables.

        Parameters
        ----------
        template_text : str
            Template as text with placeholders.

        required_input_variables : list[str]
            Input variables' names to check for.

        Raises
        ------
        ValidationError
            If any required input variable missing.
        """
        for key in required_input_variables:
            if key not in template_text:
                raise ValidationError(key)

    def _swap_apikey_for_token(self, credentials: dict) -> dict:
        """Remove api_key form credentials and add token. Used primarily
        to prevent api_key from displaying in stored function code preview.

        Parameters
        ----------
        credentials : dict
            Credentials to modify.

        Returns
        -------
        dict
            Credentials with api_key removed and token added.
        """
        result = credentials.copy()
        result.pop("api_key", None)
        result["token"] = self.api_client.token

        return result

    @staticmethod
    def _get_function_source_code(function: Callable) -> str:
        """
        Returns source code of a function, without the function definition itself.

        Example:
            Input: def fun():
                    x = 1

            Output: x = 1

        Parameters
        ----------
        function: Callable
            Function to extract source code from.

        Returns
        -------
        str
            Source code of the function.
        """
        function_source_code = inspect.getsource(function)
        indexing_code_source = textwrap.dedent(function_source_code)
        function_source = ast.parse(indexing_code_source).body[0]
        function_body_lines = [ast.unparse(stmt) for stmt in function_source.body]
        function_body_code = "\n".join(function_body_lines)

        return function_body_code
