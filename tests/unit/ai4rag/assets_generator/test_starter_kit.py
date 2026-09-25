# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import zipfile

from ai4rag.assets_generator.starter_kit import (
    _apply_provider_conditionals,
    _create_starter_kit_mapping,
    generate_starter_kit,
)

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
        "rag_template": "AgenticRAG",
    },
    "indexing": {
        "pipeline_spec": {
            "parameters": {
                "maas_secret_name": "maas-connection",
                "vector_db_secret_name": "vector-db-connection",
            }
        }
    },
}


# ---------------------------------------------------------------------------
# _create_starter_kit_mapping
# ---------------------------------------------------------------------------


class TestCreateStarterKitMapping:
    def test_extracts_generation_fields(self):
        mapping = _create_starter_kit_mapping(_SAMPLE_PATTERN_DATA)
        assert mapping["__FM_MODEL_ID__"] == "publishers/ibm/models/granite-3.1-8b-instruct"
        assert mapping["__TEMPERATURE__"] == "0.2"
        assert mapping["__MAX_COMPLETION_TOKENS__"] == "1024"
        assert mapping["__SYSTEM_MESSAGE__"] == "Answer the question."
        assert mapping["__USER_MESSAGE__"] == "Question: {question}"
        assert mapping["__CONTEXT_TEMPLATE__"] == "Context: {context}"
        assert mapping["__LANGUAGE_CODE__"] == "en"
        assert mapping["__LANGUAGE_NAME__"] == "English"

    def test_extracts_embedding_fields(self):
        mapping = _create_starter_kit_mapping(_SAMPLE_PATTERN_DATA)
        assert mapping["__EMBEDDING_MODEL_ID__"] == "publishers/ibm/models/slate-125m-english-rtrvr"
        assert mapping["__EMBEDDING_DIMENSION__"] == "768"

    def test_extracts_retrieval_fields(self):
        mapping = _create_starter_kit_mapping(_SAMPLE_PATTERN_DATA)
        assert mapping["__RETRIEVAL_METHOD__"] == "simple"
        assert mapping["__NUMBER_OF_CHUNKS__"] == "5"
        assert mapping["__SEARCH_MODE__"] == "hybrid"
        assert mapping["__RANKER_STRATEGY__"] == "weighted"
        assert mapping["__RANKER_ALPHA__"] == "0.5"

    def test_extracts_vector_store_fields(self):
        mapping = _create_starter_kit_mapping(_SAMPLE_PATTERN_DATA)
        assert mapping["__PROVIDER_TYPE__"] == "milvus"
        assert mapping["__COLLECTION_NAME__"] == "test_collection"

    def test_extracts_indexing_secret_names(self):
        mapping = _create_starter_kit_mapping(_SAMPLE_PATTERN_DATA)
        assert mapping["__MAAS_SECRET_NAME__"] == "maas-connection"
        assert mapping["__VECTOR_DB_SECRET_NAME__"] == "vector-db-connection"

    def test_extracts_pattern_name(self):
        mapping = _create_starter_kit_mapping(_SAMPLE_PATTERN_DATA)
        assert mapping["__PATTERN_NAME__"] == "pattern_001"

    def test_defaults_on_empty_data(self):
        mapping = _create_starter_kit_mapping({})
        assert mapping["__FM_MODEL_ID__"] == ""
        assert mapping["__TEMPERATURE__"] == "0.0"
        assert mapping["__RETRIEVAL_METHOD__"] == "simple"
        assert mapping["__PROVIDER_TYPE__"] == "milvus"
        assert mapping["__EMBEDDING_DIMENSION__"] == "768"

    def test_all_values_are_strings(self):
        mapping = _create_starter_kit_mapping(_SAMPLE_PATTERN_DATA)
        for key, value in mapping.items():
            assert isinstance(value, str), f"{key} value is {type(value)}, not str"


# ---------------------------------------------------------------------------
# _apply_provider_conditionals
# ---------------------------------------------------------------------------


class TestApplyProviderConditionals:
    def test_keeps_milvus_strips_pgvector(self, tmp_path):
        content = (
            "# common\n"
            "# <<< BEGIN MILVUS >>>\n"
            "MILVUS_HOST=\n"
            "# <<< END MILVUS >>>\n"
            "# <<< BEGIN PGVECTOR >>>\n"
            "PGVECTOR_HOST=\n"
            "# <<< END PGVECTOR >>>\n"
            "# footer\n"
        )
        f = tmp_path / "test.env"
        f.write_text(content)
        _apply_provider_conditionals(f, "milvus")
        result = f.read_text()
        assert "MILVUS_HOST=" in result
        assert "PGVECTOR_HOST=" not in result
        assert "# common" in result
        assert "# footer" in result
        assert "<<< BEGIN" not in result

    def test_keeps_pgvector_strips_milvus(self, tmp_path):
        content = (
            "# <<< BEGIN MILVUS >>>\n"
            "MILVUS_HOST=\n"
            "# <<< END MILVUS >>>\n"
            "# <<< BEGIN PGVECTOR >>>\n"
            "PGVECTOR_HOST=\n"
            "# <<< END PGVECTOR >>>\n"
        )
        f = tmp_path / "test.env"
        f.write_text(content)
        _apply_provider_conditionals(f, "pgvector")
        result = f.read_text()
        assert "PGVECTOR_HOST=" in result
        assert "MILVUS_HOST=" not in result

    def test_case_insensitive(self, tmp_path):
        content = "# <<< BEGIN MILVUS >>>\n" "MILVUS_HOST=\n" "# <<< END MILVUS >>>\n"
        f = tmp_path / "test.env"
        f.write_text(content)
        _apply_provider_conditionals(f, "Milvus")
        result = f.read_text()
        assert "MILVUS_HOST=" in result


# ---------------------------------------------------------------------------
# generate_starter_kit
# ---------------------------------------------------------------------------


class TestGenerateStarterKit:
    def test_produces_zip(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        assert zip_path.exists()
        assert zip_path.name == "starter_kit.zip"

    def test_zip_has_top_level_directory(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            top_dirs = {name.split("/")[0] for name in zf.namelist()}
            assert top_dirs == {"starter_kit"}

    def test_zip_contains_expected_files(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            assert "starter_kit/main.py" in names
            assert "starter_kit/Makefile" in names
            assert "starter_kit/Containerfile.openshell" in names
            assert "starter_kit/values.yaml" in names
            assert "starter_kit/agent.yaml" not in names
            assert "starter_kit/agent_config.json" in names
            assert "starter_kit/src/agentic_rag/agent.py" in names
            assert "starter_kit/src/agentic_rag/tools.py" in names
            assert "starter_kit/src/agentic_rag/sqlite_shim.py" in names
            assert "starter_kit/sqlite_shim.py" not in names
            assert "starter_kit/src/agentic_rag/tracing.py" not in names
            assert "starter_kit/src/agentic_rag/sqlite_shim.py" in names
            assert "starter_kit/sqlite_shim.py" not in names
            assert "starter_kit/auth_wrapper.py" in names

    def test_agent_config_is_injected_at_deploy_time(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            containerfile = zf.read("starter_kit/Containerfile.openshell").decode("utf-8")
            makefile = zf.read("starter_kit/Makefile").decode("utf-8")

        assert "COPY --chown=1001:0 agent_config.json /sandbox/agent_config.json" not in containerfile
        assert "AGENT_CONFIG_B64" in makefile

    def test_env_example_has_filled_values(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            values_content = zf.read("starter_kit/values.yaml").decode("utf-8")
            assert 'MODEL_ID: "publishers/ibm/models/granite-3.1-8b-instruct"' in values_content
            agent_config = json.loads(zf.read("starter_kit/agent_config.json"))
            assert agent_config["runtime"]["port"] == 8000
            assert agent_config["generation"]["model_id"] == "publishers/ibm/models/granite-3.1-8b-instruct"
            assert agent_config["prompts"]["system_message"] == "Answer the question."
            assert agent_config["prompts"]["user_message_template"] == "Question: {question}"
            assert agent_config["prompts"]["context_template"] == "Context: {context}"
            assert agent_config["retrieval"]["number_of_chunks"] == 5
            assert 'RETRIEVAL_METHOD: "simple"' in values_content
            assert 'SEARCH_MODE: "hybrid"' in values_content
            assert 'EMBEDDING_DIMENSION: "768"' in values_content

    def test_no_placeholders_remain(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith((".py", ".json", ".yaml", ".yml", ".md", ".toml")):
                    content = zf.read(name).decode("utf-8")
                    assert "__FM_MODEL_ID__" not in content, f"Unreplaced placeholder in {name}"
                    assert "__PROVIDER_TYPE__" not in content, f"Unreplaced placeholder in {name}"

    def test_zip_uses_agentic_rag_template(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            agent_content = zf.read("starter_kit/src/agentic_rag/agent.py").decode("utf-8")

        assert "from ai4rag.rag.template.agentic_rag_template import AgenticRAG" in agent_content

    def test_milvus_provider_keeps_milvus_block(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            config = json.loads(zf.read("starter_kit/agent_config.json"))
            assert config["vector_store"]["provider_type"] == "milvus"
            assert config["vector_store"]["collection_name"] == "test_collection"

    def test_pgvector_provider_keeps_pgvector_block(self, tmp_path):
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

    def test_values_yaml_has_filled_values(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            values_content = zf.read("starter_kit/values.yaml").decode("utf-8")
            assert '"publishers/ibm/models/granite-3.1-8b-instruct"' in values_content
            assert '"milvus"' in values_content
            assert 'maas_secret_name: "maas-connection"' in values_content
            assert 'vector_db_secret_name: "vector-db-connection"' in values_content

    def test_creates_output_dir_if_needed(self, tmp_path):
        out = tmp_path / "nested" / "dir"
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, out)
        assert zip_path.exists()

    def test_collection_name_in_agent_config(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            config = json.loads(zf.read("starter_kit/agent_config.json"))
            assert config["vector_store"]["collection_name"] == "test_collection"

    def test_no_data_directory(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            data_files = [n for n in zf.namelist() if n.startswith("starter_kit/data/")]
            assert data_files == [], f"data/ directory should not exist: {data_files}"

    def test_excludes_local_development_artifacts(self, tmp_path):
        zip_path = generate_starter_kit(_SAMPLE_PATTERN_DATA, tmp_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert not any("__pycache__/" in name for name in names)
            assert not any(name.endswith(".pyc") for name in names)
            assert not any(name.startswith("starter_kit/.venv/") for name in names)
            assert "starter_kit/.env" not in names
