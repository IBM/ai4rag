# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import pytest
from langchain_core.documents import Document

from ai4rag.rag.template.base_template import RAGTemplateError
from ai4rag.rag.template.simple_rag_template import SimpleRAG


class TestOGXRAGInitialization:
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


class TestOGXRAGBuildIndex:
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
            Document(page_content="chunk1", metadata={"document_id": "doc1", "sequence_number": 1}),
            Document(page_content="chunk2", metadata={"document_id": "doc1", "sequence_number": 2}),
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
        """Create sample documents for testing."""
        return [
            Document(page_content="This is test document 1.", metadata={"document_id": "doc1"}),
            Document(page_content="This is test document 2.", metadata={"document_id": "doc2"}),
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
        assert added_chunks[0].page_content == "chunk1"
        assert added_chunks[1].page_content == "chunk2"

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
            rag.build_index([Document(page_content="test", metadata={})])

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
            rag.build_index([Document(page_content="test", metadata={})])

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
        rag.build_index([Document(page_content="test", metadata={})])
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
            rag.build_index([Document(page_content="test", metadata={})])

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
        large_doc_list = [
            Document(page_content=f"Document {i}", metadata={"document_id": f"doc{i}"}) for i in range(100)
        ]
        large_chunk_list = [
            Document(page_content=f"Chunk {i}", metadata={"document_id": f"doc{i % 100}", "sequence_number": i})
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


class TestOGXRAGGenerate:
    """Test suite for SimpleRAG.generate method."""

    @pytest.fixture
    def mock_foundation_model(self, mocker):
        """Create a mock foundation model."""
        mock = mocker.MagicMock()
        mock.system_message_text = "You are a helpful assistant."
        mock.user_message_text = "Question: {question}\nReferences: {reference_documents}"
        mock.context_template_text = "Document: {document}"

        # Mock the create_response method to return a string
        mock.create_response.return_value = "This is the generated answer."
        return mock

    @pytest.fixture
    def mock_retriever(self, mocker):
        """Create a mock retriever."""
        mock = mocker.MagicMock()
        mock.collection_name = "test-collection"
        mock.retrieve.return_value = [
            Document(page_content="Relevant document 1", metadata={"document_id": "doc1"}),
            Document(page_content="Relevant document 2", metadata={"document_id": "doc2"}),
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

        # Verify create_response was called
        mock_foundation_model.create_response.assert_called_once()
        call_args = mock_foundation_model.create_response.call_args

        # Verify user_message was passed correctly with context
        user_message = call_args.kwargs["user_message"]
        assert "Question: What is AI?" in user_message
        assert "Document: Relevant document 1" in user_message
        assert "Document: Relevant document 2" in user_message

    def test_generate_calls_foundation_model_create_response(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test that generate calls foundation model's create_response method."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        rag.generate("What is AI?")

        mock_foundation_model.create_response.assert_called_once()
        call_args = mock_foundation_model.create_response.call_args

        # Verify user_message and vector_store_id were passed correctly
        user_message = call_args.kwargs["user_message"]
        assert "What is AI?" in user_message
        assert "Question:" in user_message
        assert call_args.kwargs["vector_store_id"] == "test-collection"  # From retriever.collection_name

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
        assert result["reference_documents"][0].page_content == "Relevant document 1"
        assert result["reference_documents"][1].page_content == "Relevant document 2"

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

        # Verify create_response was called with empty context
        call_args = mock_foundation_model.create_response.call_args
        user_message = call_args.kwargs["user_message"]
        assert "What is AI?" in user_message

    def test_generate_with_single_retrieved_document(
        self,
        mock_foundation_model,
        mock_retriever,
    ):
        """Test generate with single retrieved document."""
        mock_retriever.retrieve.return_value = [
            Document(page_content="Single document", metadata={"document_id": "doc1"}),
        ]

        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        result = rag.generate("What is AI?")

        assert len(result["reference_documents"]) == 1
        assert result["reference_documents"][0].page_content == "Single document"

    def test_generate_handles_document_without_page_content_attribute(
        self,
        mock_foundation_model,
        mock_retriever,
        mocker,
    ):
        """Test that generate handles documents without page_content attribute gracefully."""
        # Create a mock document-like object without page_content
        mock_doc = mocker.MagicMock(spec=[])
        del mock_doc.page_content  # Ensure page_content doesn't exist
        mock_retriever.retrieve.return_value = [mock_doc]

        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )

        result = rag.generate("What is AI?")

        # Should use empty string when page_content is missing
        call_args = mock_foundation_model.create_response.call_args
        user_message = call_args.kwargs["user_message"]
        assert "Document: " in user_message
        assert result["answer"] == "This is the generated answer."

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


class TestOGXRAGGenerateStream:
    """Test suite for SimpleRAG.generate_stream method."""

    @pytest.fixture
    def mock_foundation_model(self, mocker):
        """Create a mock foundation model."""
        mock = mocker.MagicMock()
        mock.system_message_text = "You are a helpful assistant."
        mock.user_message_text = "Question: {question}\nReferences: {reference_documents}"
        mock.context_template_text = "Document: {document}"

        # Mock the create_response method to return a string
        mock.create_response.return_value = "This is the generated answer."
        return mock

    @pytest.fixture
    def mock_retriever(self, mocker):
        """Create a mock retriever."""
        mock = mocker.MagicMock()
        mock.retrieve.return_value = [
            Document(page_content="Relevant document", metadata={"document_id": "doc1"}),
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
        mock_foundation_model.create_response.return_value = "This is a longer answer with multiple sentences."

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


class TestOGXRAGIntegration:
    """Integration tests for SimpleRAG full workflow."""

    @pytest.fixture
    def complete_rag_system(self, mocker):
        """Create a complete RAG system with all components."""
        # Foundation model
        foundation_model = mocker.MagicMock()
        foundation_model.system_message_text = "You are a helpful assistant."
        foundation_model.user_message_text = "Question: {question}\nReferences: {reference_documents}"
        foundation_model.context_template_text = "Document: {document}"

        # Mock the create_response method to return a string
        foundation_model.create_response.return_value = "The answer is 42."

        # Retriever
        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = [
            Document(
                page_content="The answer to everything is 42.",
                metadata={"document_id": "doc1", "sequence_number": 1},
            ),
        ]

        # Chunker
        chunker = mocker.MagicMock()
        chunker.split_documents.return_value = [
            Document(
                page_content="The answer to everything is 42.",
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
            Document(page_content="The answer to everything is 42.", metadata={"document_id": "doc1"}),
        ]
        rag.build_index(documents)

        # Verify build_index worked
        complete_rag_system["chunker"].split_documents.assert_called_once_with(documents)
        complete_rag_system["vector_store"].add_documents.assert_called_once()

        # Step 2: Generate answer
        result = rag.generate("What is the answer?")

        # Verify generate worked
        complete_rag_system["retriever"].retrieve.assert_called_once_with("What is the answer?")
        complete_rag_system["foundation_model"].create_response.assert_called_once()

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
            Document(page_content="Test content", metadata={"document_id": "doc1"}),
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
        assert complete_rag_system["foundation_model"].create_response.call_count == 3


class TestOGXRAGEdgeCases:
    """Test suite for edge cases and error scenarios."""

    @pytest.fixture
    def mock_foundation_model(self, mocker):
        """Create a mock foundation model."""
        mock = mocker.MagicMock()
        mock.system_message_text = "System message"
        mock.user_message_text = "{question} {reference_documents}"
        mock.context_template_text = "{document}"

        # Mock the create_response method to return a string
        mock.create_response.return_value = "Answer"
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
