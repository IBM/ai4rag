# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.template.simple_rag_template import SimpleRAG


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

    def test_init_with_required_params(self, mock_foundation_model, mock_retriever):
        """Test initialization with the only supported parameters."""
        rag = SimpleRAG(
            foundation_model=mock_foundation_model,
            retriever=mock_retriever,
        )
        assert rag.foundation_model == mock_foundation_model
        assert rag.retriever == mock_retriever

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
        assert hasattr(rag, "foundation_model")
        assert hasattr(rag, "retriever")
        assert not hasattr(rag, "vector_store")
        assert not hasattr(rag, "embedding_model")
        assert not hasattr(rag, "chunker")


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


class TestSimpleRAGChat:
    """Test suite for SimpleRAG.chat method."""

    @pytest.fixture
    def mock_foundation_model(self, mocker):
        """Create a mock foundation model."""
        mock = mocker.MagicMock()
        mock.system_message_text = "You are a helpful assistant."
        mock.user_message_text = "Question: {question}\nReferences: {reference_documents}"
        mock.context_template_text = "Document: {document}"

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

    def test_chat_raises_on_empty_messages(self, mock_foundation_model, mock_retriever):
        """Test that chat rejects an empty message list."""
        rag = SimpleRAG(foundation_model=mock_foundation_model, retriever=mock_retriever)

        with pytest.raises(ValueError):
            rag.chat([])

    def test_chat_retrieves_using_last_message_content(self, mock_foundation_model, mock_retriever):
        """Test that chat uses the last message's content as the retrieval query."""
        rag = SimpleRAG(foundation_model=mock_foundation_model, retriever=mock_retriever)

        rag.chat([{"role": "user", "content": "What is AI?"}])

        mock_retriever.retrieve.assert_called_once_with("What is AI?")

    def test_chat_passes_retrieval_kwargs(self, mock_foundation_model, mock_retriever):
        """Test that chat forwards kwargs to the retriever."""
        rag = SimpleRAG(foundation_model=mock_foundation_model, retriever=mock_retriever)

        rag.chat([{"role": "user", "content": "What is AI?"}], number_of_chunks=5)

        mock_retriever.retrieve.assert_called_once_with("What is AI?", number_of_chunks=5)

    def test_chat_prepends_system_message(self, mock_foundation_model, mock_retriever):
        """Test that chat always prepends the template's system message."""
        rag = SimpleRAG(foundation_model=mock_foundation_model, retriever=mock_retriever)

        rag.chat([{"role": "user", "content": "What is AI?"}])

        messages = mock_foundation_model.chat.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}

    def test_chat_preserves_earlier_history_untouched(self, mock_foundation_model, mock_retriever):
        """Test that only the last message is enriched; earlier turns pass through unchanged."""
        rag = SimpleRAG(foundation_model=mock_foundation_model, retriever=mock_retriever)

        rag.chat(
            [
                {"role": "user", "content": "Hi there"},
                {"role": "assistant", "content": "Hello! How can I help?"},
                {"role": "user", "content": "What is AI?"},
            ]
        )

        messages = mock_foundation_model.chat.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert messages[1] == {"role": "user", "content": "Hi there"}
        assert messages[2] == {"role": "assistant", "content": "Hello! How can I help?"}
        assert messages[3]["role"] == "user"
        assert "What is AI?" in messages[3]["content"]

    def test_chat_enriches_last_message_with_context(self, mock_foundation_model, mock_retriever):
        """Test that the last message's content is replaced with the RAG-enriched version."""
        rag = SimpleRAG(foundation_model=mock_foundation_model, retriever=mock_retriever)

        rag.chat([{"role": "user", "content": "What is AI?"}])

        messages = mock_foundation_model.chat.call_args.kwargs["messages"]
        enriched_content = messages[-1]["content"]
        assert "Document: Relevant document" in enriched_content
        assert "What is AI?" in enriched_content

    def test_chat_preserves_last_message_role(self, mock_foundation_model, mock_retriever):
        """Test that the role of the last message is preserved after enrichment."""
        rag = SimpleRAG(foundation_model=mock_foundation_model, retriever=mock_retriever)

        rag.chat([{"role": "user", "content": "What is AI?"}])

        messages = mock_foundation_model.chat.call_args.kwargs["messages"]
        assert messages[-1]["role"] == "user"

    def test_chat_returns_foundation_model_chat_response(self, mock_foundation_model, mock_retriever):
        """Test that chat returns the foundation model's chat response verbatim."""
        rag = SimpleRAG(foundation_model=mock_foundation_model, retriever=mock_retriever)

        result = rag.chat([{"role": "user", "content": "What is AI?"}])

        assert result == mock_foundation_model.chat.return_value

    def test_chat_with_no_retrieved_documents(self, mock_foundation_model, mock_retriever):
        """Test chat when the retriever returns no documents."""
        mock_retriever.retrieve.return_value = []
        rag = SimpleRAG(foundation_model=mock_foundation_model, retriever=mock_retriever)

        result = rag.chat([{"role": "user", "content": "What is AI?"}])

        assert result == mock_foundation_model.chat.return_value
        messages = mock_foundation_model.chat.call_args.kwargs["messages"]
        assert "What is AI?" in messages[-1]["content"]


class TestSimpleRAGIntegration:
    """Integration tests for SimpleRAG retrieval-and-generation workflow."""

    @pytest.fixture
    def rag_system(self, mocker):
        """Create a complete RAG system for retrieval and generation."""
        foundation_model = mocker.MagicMock()
        foundation_model.system_message_text = "You are a helpful assistant."
        foundation_model.user_message_text = "Question: {question}\nReferences: {reference_documents}"
        foundation_model.context_template_text = "Document: {document}"

        mock_message = mocker.MagicMock()
        mock_message.content = "The answer is 42."
        mock_choice = mocker.MagicMock()
        mock_choice.message = mock_message
        foundation_model.chat.return_value = [mock_choice]

        retriever = mocker.MagicMock()
        retriever.retrieve.return_value = [
            AI4RAGChunk(
                text="The answer to everything is 42.",
                metadata={"document_id": "doc1", "sequence_number": 1},
            ),
        ]

        return {"foundation_model": foundation_model, "retriever": retriever}

    def test_multiple_generate_calls(self, rag_system):
        """Test multiple generate calls against a single SimpleRAG instance."""
        rag = SimpleRAG(
            foundation_model=rag_system["foundation_model"],
            retriever=rag_system["retriever"],
        )

        questions = ["Question 1?", "Question 2?", "Question 3?"]
        for question in questions:
            result = rag.generate(question)
            assert result["question"] == question
            assert result["answer"] == "The answer is 42."

        assert rag_system["retriever"].retrieve.call_count == 3
        assert rag_system["foundation_model"].chat.call_count == 3

    def test_generate_then_chat_share_the_same_retrieval_and_context_logic(self, rag_system):
        """Test that generate and chat produce equivalent enriched content for the same question."""
        rag = SimpleRAG(
            foundation_model=rag_system["foundation_model"],
            retriever=rag_system["retriever"],
        )

        rag.generate("What is the answer?")
        generate_messages = rag_system["foundation_model"].chat.call_args.kwargs["messages"]

        rag.chat([{"role": "user", "content": "What is the answer?"}])
        chat_messages = rag_system["foundation_model"].chat.call_args.kwargs["messages"]

        assert generate_messages[1]["content"] == chat_messages[-1]["content"]


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
