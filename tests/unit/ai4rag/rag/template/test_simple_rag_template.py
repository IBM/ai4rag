# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import pytest
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.template.base_template import RAGTemplateError
from ai4rag.rag.template.simple_rag_template import SimpleRAG


def _make_docling_doc(name: str, text: str) -> DoclingDocument:
    """Create a minimal DoclingDocument for testing."""
    doc = DoclingDocument(name=name)
    doc.add_text(label=DocItemLabel.PARAGRAPH, text=text)
    return doc


class TestSimpleRAGInitialization:
    """Test suite for SimpleRAG initialization."""

    @pytest.fixture
    def mock_foundation_model(self, mocker):
        """Create a mock foundation model."""
        mock = mocker.MagicMock()
        mock.model_id = "test-model"
        mock.system_message_text = "You are a helpful assistant."
        mock.user_message_text = "Question: {question}\nReferences: {reference_documents}"
        mock.context_template_text = "Document: {document}"
        return mock

    @pytest.fixture
    def mock_retriever(self, mocker):
        """Create a mock retriever."""
        return mocker.MagicMock()

    @pytest.fixture
    def mock_chunker(self, mocker):
        """Create a mock LangChain chunker."""
        return mocker.MagicMock()

    @pytest.fixture
    def mock_embedding_model(self, mocker):
        """Create a mock embedding model."""
        return mocker.MagicMock()

    @pytest.fixture
    def mock_vector_store(self, mocker):
        """Create a mock vector store."""
        return mocker.MagicMock()

    def test_init_with_required_params_only(self, mock_foundation_model, mock_retriever):
        """Test initialization with only required parameters."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )
        assert rag.foundation_model == mock_foundation_model
        assert rag.retriever == mock_retriever
        assert rag.chunker is None
        assert rag.embedding_model is None
        assert rag.vector_store is None

    def test_init_with_all_params(
        self,
        mock_foundation_model,
        mock_retriever,
        mock_chunker,
        mock_embedding_model,
        mock_vector_store,
    ):
        """Test initialization with all parameters."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
            chunker=mock_chunker,
            embedding_model=mock_embedding_model,
            vector_store=mock_vector_store,
        )
        assert rag.foundation_model == mock_foundation_model
        assert rag.retriever == mock_retriever
        assert rag.chunker == mock_chunker
        assert rag.embedding_model == mock_embedding_model
        assert rag.vector_store == mock_vector_store

    def test_init_with_optional_chunker(self, mock_foundation_model, mock_retriever, mock_chunker):
        """Test initialization with optional chunker."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
            chunker=mock_chunker,
        )
        assert rag.chunker == mock_chunker
        assert rag.embedding_model is None
        assert rag.vector_store is None

    def test_init_with_optional_embedding_model(
        self,
        mock_foundation_model,
        mock_retriever,
        mock_embedding_model,
    ):
        """Test initialization with optional embedding model."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
            embedding_model=mock_embedding_model,
        )
        assert rag.embedding_model == mock_embedding_model
        assert rag.chunker is None
        assert rag.vector_store is None

    def test_init_with_optional_vector_store(
        self,
        mock_foundation_model,
        mock_retriever,
        mock_vector_store,
    ):
        """Test initialization with optional vector store."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
            vector_store=mock_vector_store,
        )
        assert rag.vector_store == mock_vector_store
        assert rag.chunker is None
        assert rag.embedding_model is None

    def test_init_inherits_from_base_template(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that SimpleRAG properly inherits from BaseRAGTemplate."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )
        # BaseRAGTemplate attributes should be accessible
        assert hasattr(rag, "foundation_model")
        assert hasattr(rag, "retriever")
        assert hasattr(rag, "embedding_model")
        assert hasattr(rag, "vector_store")


class TestSimpleRAGBuildIndex:
    """Test suite for SimpleRAG.build_index method."""

    @pytest.fixture
    def mock_foundation_model(self, mocker):
        """Create a mock foundation model."""
        return mocker.MagicMock()

    @pytest.fixture
    def mock_retriever(self, mocker):
        """Create a mock retriever."""
        return mocker.MagicMock()

    @pytest.fixture
    def mock_chunker(self, mocker):
        """Create a mock chunker."""
        mock = mocker.MagicMock()
        mock.split_documents.return_value = [
            AI4RAGChunk(text="chunk1", metadata={"document_id": "doc1", "sequence_number": 1}),
            AI4RAGChunk(text="chunk2", metadata={"document_id": "doc1", "sequence_number": 2}),
        ]
        return mock

    @pytest.fixture
    def mock_vector_store(self, mocker):
        """Create a mock vector store."""
        mock = mocker.MagicMock()
        mock.add_documents.return_value = None
        return mock

    @pytest.fixture
    def sample_documents(self):
        """Create sample DoclingDocuments for testing."""
        return [
            _make_docling_doc("doc1", "This is test document 1."),
            _make_docling_doc("doc2", "This is test document 2."),
        ]

    def test_build_index_chunks_and_adds_to_vector_store(
        self,
        mock_foundation_model,
        mock_retriever,
        mock_chunker,
        mock_vector_store,
        sample_documents,
    ):
        """Test that build_index chunks documents and adds them to vector store."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
            chunker=mock_chunker,
            vector_store=mock_vector_store,
        )

        rag.build_index(sample_documents)

        # Verify chunker was called with documents
        mock_chunker.split_documents.assert_called_once_with(sample_documents)

        # Verify vector store received the chunks
        mock_vector_store.add_documents.assert_called_once()
        added_chunks = mock_vector_store.add_documents.call_args[0][0]
        assert len(added_chunks) == 2
        assert added_chunks[0].text == "chunk1"
        assert added_chunks[1].text == "chunk2"

    def test_build_index_raises_error_when_chunker_is_none(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that build_index raises RAGTemplateError when chunker is None."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
            chunker=None,
        )

        with pytest.raises(RAGTemplateError):
            rag.build_index([_make_docling_doc("test", "test")])

    def test_build_index_fails_when_vector_store_is_none(
        self,
        mock_foundation_model,
        mock_retriever,
        mock_chunker,
        mocker,
    ):
        """Test that build_index fails with AttributeError when vector_store is None."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
            chunker=mock_chunker,
            embedding_model=mocker.MagicMock(),
            vector_store=None,
        )

        with pytest.raises(AttributeError):
            rag.build_index([_make_docling_doc("test", "test")])

    def test_build_index_works_with_embedding_model_none(
        self,
        mock_foundation_model,
        mock_retriever,
        mock_chunker,
        mock_vector_store,
    ):
        """Test that build_index works when embedding_model is None but other components are present."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
            chunker=mock_chunker,
            embedding_model=None,
            vector_store=mock_vector_store,
        )

        # Should not raise RAGTemplateError - embedding_model is not used in build_index
        rag.build_index([_make_docling_doc("test", "test")])
        mock_chunker.split_documents.assert_called_once()
        mock_vector_store.add_documents.assert_called_once()

    def test_build_index_raises_error_when_all_components_are_none(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that build_index raises RAGTemplateError when all components are None."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
            chunker=None,
            embedding_model=None,
            vector_store=None,
        )

        with pytest.raises(RAGTemplateError):
            rag.build_index([_make_docling_doc("test", "test")])

    def test_build_index_with_empty_document_list(
        self,
        mock_foundation_model,
        mock_retriever,
        mock_chunker,
        mock_vector_store,
    ):
        """Test build_index with empty document list."""
        mock_chunker.split_documents.return_value = []
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
            chunker=mock_chunker,
            vector_store=mock_vector_store,
        )

        rag.build_index([])

        mock_chunker.split_documents.assert_called_once_with([])
        mock_vector_store.add_documents.assert_called_once_with([])

    def test_build_index_with_large_document_list(
        self,
        mock_foundation_model,
        mock_retriever,
        mock_chunker,
        mock_vector_store,
    ):
        """Test build_index with large document list."""
        large_doc_list = [_make_docling_doc(f"doc{i}", f"Document {i}") for i in range(100)]
        large_chunk_list = [
            AI4RAGChunk(text=f"Chunk {i}", metadata={"document_id": f"doc{i % 100}", "sequence_number": i})
            for i in range(500)
        ]
        mock_chunker.split_documents.return_value = large_chunk_list

        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
            chunker=mock_chunker,
            vector_store=mock_vector_store,
        )

        rag.build_index(large_doc_list)

        mock_chunker.split_documents.assert_called_once_with(large_doc_list)
        mock_vector_store.add_documents.assert_called_once()
        added_chunks = mock_vector_store.add_documents.call_args[0][0]
        assert len(added_chunks) == 500


class TestSimpleRAGGenerate:
    """Test suite for SimpleRAG.generate method."""

    @pytest.fixture
    def mock_foundation_model(self, mocker):
        """Create a mock foundation model."""
        mock = mocker.MagicMock()
        mock.system_message_text = "You are a helpful assistant."
        mock.user_message_text = "Question: {question}\nReferences: {reference_documents}"
        mock.context_template_text = "Document: {document}"

        # Mock the chat response to match new API (returns list of choices)
        mock_message = mocker.MagicMock()
        mock_message.content = "This is the generated answer."
        mock_choice = mocker.MagicMock()
        mock_choice.message = mock_message
        mock.chat.return_value = [mock_choice]
        return mock

    @pytest.fixture
    def mock_retriever(self, mocker):
        """Create a mock retriever."""
        mock = mocker.MagicMock()
        mock.retrieve.return_value = [
            AI4RAGChunk(text="Relevant document 1", metadata={"document_id": "doc1"}),
            AI4RAGChunk(text="Relevant document 2", metadata={"document_id": "doc2"}),
        ]
        return mock

    def test_generate_returns_dict_with_correct_keys(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that generate returns a dict with answer, reference_documents, and question."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        result = rag.generate("What is AI?")

        assert isinstance(result, dict)
        assert "answer" in result
        assert "reference_documents" in result
        assert "question" in result

    def test_generate_retrieves_documents(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that generate calls retriever.retrieve."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        rag.generate("What is AI?")

        mock_retriever.retrieve.assert_called_once_with("What is AI?")

    def test_generate_retrieves_documents_with_kwargs(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that generate passes retrieval kwargs to retriever."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        rag.generate("What is AI?", number_of_chunks=5, window_size=3)

        mock_retriever.retrieve.assert_called_once_with(
            "What is AI?",
            number_of_chunks=5,
            window_size=3,
        )

    def test_generate_builds_context_from_retrieved_documents(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that generate builds context correctly from retrieved documents."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        rag.generate("What is AI?")

        # Verify chat was called
        mock_foundation_model.chat.assert_called_once()
        call_args = mock_foundation_model.chat.call_args

        # Verify messages list was passed correctly
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        # Verify user message contains formatted context
        user_message = messages[1]["content"]
        assert "Document: Relevant document 1" in user_message
        assert "Document: Relevant document 2" in user_message
        assert "What is AI?" in user_message

    def test_generate_numbers_documents_when_template_includes_doc_number(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that doc_number is passed when the context template includes it."""
        mock_foundation_model.context_template_text = "Document {doc_number}:\n{document}\n"
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        rag.generate("What is AI?")

        user_message = mock_foundation_model.chat.call_args.kwargs["messages"][1]["content"]
        assert "Document 1:\nRelevant document 1" in user_message
        assert "Document 2:\nRelevant document 2" in user_message

    def test_generate_calls_foundation_model_chat(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that generate calls foundation model's chat method."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        rag.generate("What is AI?")

        mock_foundation_model.chat.assert_called_once()
        call_args = mock_foundation_model.chat.call_args

        # Verify messages list was passed correctly
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert "What is AI?" in messages[1]["content"]

    def test_generate_returns_correct_answer(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that generate returns the answer from foundation model."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        result = rag.generate("What is AI?")

        assert result["answer"] == "This is the generated answer."

    def test_generate_returns_reference_documents(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that generate returns the reference documents."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        result = rag.generate("What is AI?")

        assert len(result["reference_documents"]) == 2
        assert result["reference_documents"][0].text == "Relevant document 1"
        assert result["reference_documents"][1].text == "Relevant document 2"

    def test_generate_returns_original_question(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that generate returns the original question."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        result = rag.generate("What is AI?")

        assert result["question"] == "What is AI?"

    def test_generate_with_no_retrieved_documents(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test generate when retriever returns no documents."""
        mock_retriever.retrieve.return_value = []

        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        result = rag.generate("What is AI?")

        assert result["reference_documents"] == []
        assert result["answer"] == "This is the generated answer."
        assert result["question"] == "What is AI?"

        # Verify chat was called with empty context
        call_args = mock_foundation_model.chat.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        user_message = messages[1]["content"]
        assert "What is AI?" in user_message

    def test_generate_with_single_retrieved_document(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test generate with single retrieved document."""
        mock_retriever.retrieve.return_value = [
            AI4RAGChunk(text="Single document", metadata={"document_id": "doc1"}),
        ]

        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        result = rag.generate("What is AI?")

        assert len(result["reference_documents"]) == 1
        assert result["reference_documents"][0].text == "Single document"

    def test_generate_handles_chunk_without_text_attribute(
        self,
        mock_foundation_model,
        mock_retriever,
        mocker,
    ):
        """Test that generate raises AttributeError for chunks without text attribute."""
        # Create a mock chunk-like object without text
        mock_chunk = mocker.MagicMock(spec=[])
        del mock_chunk.text  # Ensure text doesn't exist
        mock_retriever.retrieve.return_value = [mock_chunk]

        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        # Source code accesses chunk.text directly, so missing attribute raises
        with pytest.raises(AttributeError):
            rag.generate("What is AI?")

    def test_generate_with_different_questions(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test generate with various question types."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        questions = [
            "What is AI?",
            "How does machine learning work?",
            "Explain quantum computing",
            "What are the benefits of RAG systems?",
        ]

        for question in questions:
            result = rag.generate(question)
            assert result["question"] == question
            mock_retriever.retrieve.assert_called_with(question)

    @pytest.mark.parametrize(
        "retrieval_kwargs",
        [
            {"number_of_chunks": 5},
            {"window_size": 3},
            {"number_of_chunks": 10, "window_size": 5},
            {"custom_param": "value"},
        ],
    )
    def test_generate_with_various_retrieval_kwargs(
        self,
        mock_foundation_model,
        mock_retriever,
        retrieval_kwargs,
    ):
        """Parameterized test for generate with various retrieval kwargs."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        rag.generate("What is AI?", **retrieval_kwargs)

        mock_retriever.retrieve.assert_called_once_with("What is AI?", **retrieval_kwargs)


class TestSimpleRAGGenerateStream:
    """Test suite for SimpleRAG.generate_stream method."""

    @pytest.fixture
    def mock_foundation_model(self, mocker):
        """Create a mock foundation model."""
        mock = mocker.MagicMock()
        mock.system_message_text = "You are a helpful assistant."
        mock.user_message_text = "Question: {question}\nReferences: {reference_documents}"
        mock.context_template_text = "Document: {document}"

        # Mock the chat response to match new API (returns list of choices)
        mock_message = mocker.MagicMock()
        mock_message.content = "This is the generated answer."
        mock_choice = mocker.MagicMock()
        mock_choice.message = mock_message
        mock.chat.return_value = [mock_choice]
        return mock

    @pytest.fixture
    def mock_retriever(self, mocker):
        """Create a mock retriever."""
        mock = mocker.MagicMock()
        mock.retrieve.return_value = [
            AI4RAGChunk(text="Relevant document", metadata={"document_id": "doc1"}),
        ]
        return mock

    def test_generate_stream_is_generator(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that generate_stream returns a generator."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        result = rag.generate_stream("What is AI?")

        # Check that it's a generator
        import types

        assert isinstance(result, types.GeneratorType)

    def test_generate_stream_yields_answer(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that generate_stream yields the answer."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        result = list(rag.generate_stream("What is AI?"))

        assert len(result) == 1
        assert result[0] == "This is the generated answer."

    def test_generate_stream_calls_generate_internally(
        self,
        mock_foundation_model,
        mock_retriever,
        mocker,
    ):
        """Test that generate_stream calls generate method internally."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        # Spy on the generate method
        generate_spy = mocker.spy(rag, "generate")

        list(rag.generate_stream("What is AI?"))

        generate_spy.assert_called_once_with("What is AI?")

    def test_generate_stream_with_retrieval_kwargs(
        self,
        mock_foundation_model,
        mock_retriever,
        mocker,
    ):
        """Test that generate_stream passes kwargs to generate."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        generate_spy = mocker.spy(rag, "generate")

        list(rag.generate_stream("What is AI?", number_of_chunks=5, window_size=3))

        generate_spy.assert_called_once_with("What is AI?", number_of_chunks=5, window_size=3)

    def test_generate_stream_yields_complete_answer(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that generate_stream yields the complete answer in single chunk."""
        # Update the mock to return a longer answer
        mock_message = mock_foundation_model.chat.return_value[0].message
        mock_message.content = "This is a longer answer with multiple sentences."

        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        result = list(rag.generate_stream("What is AI?"))

        assert len(result) == 1
        assert result[0] == "This is a longer answer with multiple sentences."

    def test_generate_stream_with_different_questions(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test generate_stream with various questions."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        questions = ["What is AI?", "How does ML work?", "Explain quantum computing"]

        for question in questions:
            result = list(rag.generate_stream(question))
            assert len(result) == 1
            assert result[0] == "This is the generated answer."


class TestSimpleRAGIntegration:
    """Integration tests for SimpleRAG full workflow."""

    @pytest.fixture
    def complete_rag_system(self, mocker):
        """Create a complete RAG system with all components."""
        # Foundation model
        foundation_model = mocker.MagicMock()
        foundation_model.system_message_text = "You are a helpful assistant."
        foundation_model.user_message_text = "Question: {question}\nReferences: {reference_documents}"
        foundation_model.context_template_text = "Document: {document}"

        # Mock the chat response to match new API (returns list of choices)
        mock_message = mocker.MagicMock()
        mock_message.content = "The answer is 42."
        mock_choice = mocker.MagicMock()
        mock_choice.message = mock_message
        foundation_model.chat.return_value = [mock_choice]

        # Retriever
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = [
            AI4RAGChunk(
                text="The answer to everything is 42.",
                metadata={"document_id": "doc1", "sequence_number": 1},
            ),
        ]

        # Chunker
        chunker = mocker.MagicMock()
        chunker.split_documents.return_value = [
            AI4RAGChunk(
                text="The answer to everything is 42.",
                metadata={"document_id": "doc1", "sequence_number": 1},
            ),
        ]

        # Embedding model
        embedding_model = mocker.MagicMock()

        # Vector store
        vector_store = mocker.MagicMock()
        vector_store.add_documents.return_value = None

        return {
            "foundation_model": foundation_model,
            "retriever": retriever,
            "chunker": chunker,
            "embedding_model": embedding_model,
            "vector_store": vector_store,
        }

    def test_full_rag_workflow(self, complete_rag_system):
        """Test complete RAG workflow: build index then generate answer."""
        rag = SimpleRAG(
            foundation_model=complete_rag_system["foundation_model"],
            retriever=complete_rag_system["retriever"],
            chunker=complete_rag_system["chunker"],
            embedding_model=complete_rag_system["embedding_model"],
            vector_store=complete_rag_system["vector_store"],
        )

        # Step 1: Build index
        documents = [
            _make_docling_doc("doc1", "The answer to everything is 42."),
        ]
        rag.build_index(documents)

        # Verify build_index worked
        complete_rag_system["chunker"].split_documents.assert_called_once_with(documents)
        complete_rag_system["vector_store"].add_documents.assert_called_once()

        # Step 2: Generate answer
        result = rag.generate("What is the answer?")

        # Verify generate worked
        complete_rag_system["retriever"].retrieve.assert_called_once_with("What is the answer?")
        complete_rag_system["foundation_model"].chat.assert_called_once()

        assert result["answer"] == "The answer is 42."
        assert result["question"] == "What is the answer?"
        assert len(result["reference_documents"]) == 1

    def test_multiple_generate_calls_after_single_build_index(self, complete_rag_system):
        """Test multiple generate calls after a single build_index."""
        rag = SimpleRAG(
            foundation_model=complete_rag_system["foundation_model"],
            retriever=complete_rag_system["retriever"],
            chunker=complete_rag_system["chunker"],
            embedding_model=complete_rag_system["embedding_model"],
            vector_store=complete_rag_system["vector_store"],
        )

        # Build index once
        documents = [
            _make_docling_doc("doc1", "Test content"),
        ]
        rag.build_index(documents)

        # Generate multiple answers
        questions = ["Question 1?", "Question 2?", "Question 3?"]
        for question in questions:
            result = rag.generate(question)
            assert result["question"] == question
            assert result["answer"] == "The answer is 42."

        # Verify build_index was called only once
        assert complete_rag_system["chunker"].split_documents.call_count == 1
        assert complete_rag_system["vector_store"].add_documents.call_count == 1

        # Verify generate was called three times
        assert complete_rag_system["retriever"].retrieve.call_count == 3
        assert complete_rag_system["foundation_model"].chat.call_count == 3


class TestSimpleRAGEdgeCases:
    """Test suite for edge cases and error scenarios."""

    @pytest.fixture
    def mock_foundation_model(self, mocker):
        """Create a mock foundation model."""
        mock = mocker.MagicMock()
        mock.system_message_text = "System message"
        mock.user_message_text = "{question} {reference_documents}"
        mock.context_template_text = "{document}"

        # Mock the chat response to match new API (returns list of choices)
        mock_message = mocker.MagicMock()
        mock_message.content = "Answer"
        mock_choice = mocker.MagicMock()
        mock_choice.message = mock_message
        mock.chat.return_value = [mock_choice]
        return mock

    @pytest.fixture
    def mock_retriever(self, mocker):
        """Create a mock retriever."""
        return mocker.MagicMock()

    def test_generate_with_empty_question(self, mock_foundation_model, mock_retriever):
        """Test generate with empty question string."""
        mock_retriever.retrieve.return_value = []

        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        result = rag.generate("")

        assert result["question"] == ""
        assert result["answer"] == "Answer"
        mock_retriever.retrieve.assert_called_once_with("")

    def test_generate_with_very_long_question(self, mock_foundation_model, mock_retriever):
        """Test generate with very long question."""
        mock_retriever.retrieve.return_value = []

        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        long_question = "What is AI? " * 1000
        result = rag.generate(long_question)

        assert result["question"] == long_question
        assert result["answer"] == "Answer"

    def test_generate_with_special_characters_in_question(self, mock_foundation_model, mock_retriever):
        """Test generate with special characters in question."""
        mock_retriever.retrieve.return_value = []

        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        special_question = "What is AI? @#$%^&*(){}[]|\\:;\"'<>,.?/~`"
        result = rag.generate(special_question)

        assert result["question"] == special_question
        mock_retriever.retrieve.assert_called_once_with(special_question)

    def test_generate_with_unicode_question(self, mock_foundation_model, mock_retriever):
        """Test generate with Unicode characters in question."""
        mock_retriever.retrieve.return_value = []

        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        unicode_question = "What is AI? 你好 مرحبا שלום"
        result = rag.generate(unicode_question)

        assert result["question"] == unicode_question

    def test_generate_with_multiline_question(self, mock_foundation_model, mock_retriever):
        """Test generate with multiline question."""
        mock_retriever.retrieve.return_value = []

        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        multiline_question = "What is AI?\nHow does it work?\nWhat are its applications?"
        result = rag.generate(multiline_question)

        assert result["question"] == multiline_question
