# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import zipfile

from ai4rag.assets_generator.starter_kit import generate_starter_kit

_SAMPLE_PATTERN_DATA: dict = {
    "name": "pattern_001",
    "settings": {
        "generation": {
            "model_id": "publishers/ibm/models/granite-3.1-8b-instruct",
            "system_message_text": "Answer the question.",
            "user_message_text": "Question: {question}",
            "context_template_text": "Context: {context}",
            "language": {"code": "en", "name": "English"},
            "temperature": 0.2,
            "max_completion_tokens": 1024,
        },
        "embedding": {
            "model_id": "publishers/ibm/models/slate-125m-english-rtrvr",
            "embedding_params": {"embedding_dimension": 768},
        },
        "store_binding": {
            "provider_type": "milvus",
            "collection_name": "test_collection",
        },
        "retrieval": {
            "method": "simple",
            "number_of_chunks": 5,
            "search_mode": "hybrid",
            "ranker_strategy": "weighted",
            "ranker_alpha": 0.5,
        },
    },
    "indexing": {
        "pipeline_spec": {
            "parameters": {
                "maas_secret_name": "maas-connection",
                "db_secret_name": "vector-db-connection",
            }
        }
    },
}


class TestGenerateStarterKit:
    def test_generates_archive_with_runtime_files(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        assert zip_path.exists()
        assert zip_path.name == "starter_kit.zip"
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            assert {
                "starter_kit/main.py",
                "starter_kit/Makefile",
                "starter_kit/Containerfile.openshell",
                "starter_kit/agent_config.json",
                "starter_kit/src/agentic_rag/agent.py",
                "starter_kit/auth_wrapper.py",
            } <= names

    def test_agent_config_has_filled_values(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            agent_config = json.loads(zf.read("starter_kit/agent_config.json"))
            assert "runtime" not in agent_config
            assert agent_config["generation"]["model_id"] == "publishers/ibm/models/granite-3.1-8b-instruct"
            assert agent_config["generation"]["temperature"] == 0.2
            assert agent_config["embedding"] == {
                "model_id": "publishers/ibm/models/slate-125m-english-rtrvr",
                "dimension": 768,
            }
            assert agent_config["prompts"]["system_message"] == "Answer the question."
            assert agent_config["prompts"]["user_message_template"] == "Question: {question}"
            assert agent_config["prompts"]["context_template"] == "Context: {context}"
            assert agent_config["retrieval"] == {
                "method": "simple",
                "number_of_chunks": 5,
                "search_mode": "hybrid",
                "ranker_strategy": "weighted",
                "ranker_k": None,
                "ranker_alpha": 0.5,
            }
            assert agent_config["vector_store"] == {
                "provider_type": "milvus",
                "collection_name": "test_collection",
            }

    def test_pgvector_provider_is_written_to_config(self, tmp_path):
        data = {**_SAMPLE_PATTERN_DATA}
        data["settings"] = {**data["settings"]}
        data["settings"]["store_binding"] = {
            "provider_type": "pgvector",
            "collection_name": "pg_collection",
        }
        zip_path = generate_starter_kit(data, tmp_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            config = json.loads(zf.read("starter_kit/agent_config.json"))
            assert config["vector_store"]["provider_type"] == "pgvector"
            assert config["vector_store"]["collection_name"] == "pg_collection"

    def test_rrf_ranker_k_is_preserved(self, tmp_path):
        data = {**_SAMPLE_PATTERN_DATA, "settings": {**_SAMPLE_PATTERN_DATA["settings"]}}
        data["settings"]["retrieval"] = {
            **data["settings"]["retrieval"],
            "ranker_strategy": "rrf",
            "ranker_k": 42,
            "ranker_alpha": None,
        }

        with zipfile.ZipFile(generate_starter_kit(data, tmp_path)) as archive:
            config = json.loads(archive.read("starter_kit/agent_config.json"))

        assert config["retrieval"]["ranker_k"] == 42

    def test_values_yaml_has_secret_names(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            values_content = zf.read("starter_kit/values.yaml").decode("utf-8")
            assert 'maas_secret_name: "maas-connection"' in values_content
            assert 'db_secret_name: "vector-db-connection"' in values_content

    def test_creates_output_dir_if_needed(self, tmp_path):
        out = tmp_path / "nested" / "dir"
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, out)
        assert zip_path.exists()
