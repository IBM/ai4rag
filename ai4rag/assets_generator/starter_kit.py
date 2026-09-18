# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import base64
import importlib.resources
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

_CONFIGURABLE_FILES = {".env.example", "values.yaml", "agent.yaml"}
_IGNORED_TEMPLATE_NAMES = {".DS_Store", ".env", ".venv", "__pycache__"}

_PROVIDER_BLOCK_PATTERN = re.compile(
    r"^# <<< BEGIN (?P<provider>\w+) >>>\n(?P<body>.*?)^# <<< END \1 >>>\n",
    re.MULTILINE | re.DOTALL,
)


def _create_starter_kit_mapping(output_data: dict[str, Any]) -> dict[str, str]:
    """Build a flat placeholder mapping from pattern.json data.

    All values are stringified so they can be used with ``str.replace()``.
    """
    mapping: dict[str, str] = {}

    def _value(value: Any, default: str = "") -> str:
        return default if value is None else str(value)

    mapping["__PATTERN_NAME__"] = _value(output_data.get("name"))

    settings = output_data.get("settings", {})

    fm = settings.get("generation", {})
    mapping["__FM_MODEL_ID__"] = _value(fm.get("model_id"))
    mapping["__TEMPERATURE__"] = _value(fm.get("temperature"), "0.0")
    mapping["__MAX_COMPLETION_TOKENS__"] = _value(fm.get("max_completion_tokens"))
    mapping["__SYSTEM_MESSAGE__"] = _value(fm.get("system_message_text"))
    mapping["__USER_MESSAGE__"] = _value(fm.get("user_message_text"))
    mapping["__CONTEXT_TEMPLATE__"] = _value(fm.get("context_template_text"))
    mapping["__USER_MESSAGE_B64__"] = base64.b64encode(mapping["__USER_MESSAGE__"].encode()).decode()
    mapping["__CONTEXT_TEMPLATE_B64__"] = base64.b64encode(mapping["__CONTEXT_TEMPLATE__"].encode()).decode()
    mapping["__SYSTEM_MESSAGE_B64__"] = base64.b64encode(mapping["__SYSTEM_MESSAGE__"].encode()).decode()

    language = fm.get("language", {})
    if isinstance(language, dict):
        mapping["__LANGUAGE_CODE__"] = _value(language.get("code"))
        mapping["__LANGUAGE_NAME__"] = _value(language.get("name"), "auto")
    else:
        mapping["__LANGUAGE_CODE__"] = ""
        mapping["__LANGUAGE_NAME__"] = _value(language, "auto")

    em = settings.get("embedding", {})
    mapping["__EMBEDDING_MODEL_ID__"] = _value(em.get("model_id"))
    em_params = em.get("embedding_params", {})
    mapping["__EMBEDDING_DIMENSION__"] = _value(em_params.get("embedding_dimension"), "768")

    ret = settings.get("retrieval", {})
    mapping["__RETRIEVAL_METHOD__"] = _value(ret.get("method"), "simple")
    mapping["__NUMBER_OF_CHUNKS__"] = _value(ret.get("number_of_chunks"), "5")
    mapping["__SEARCH_MODE__"] = _value(ret.get("search_mode"))
    mapping["__RANKER_STRATEGY__"] = _value(ret.get("ranker_strategy"))
    mapping["__RANKER_ALPHA__"] = _value(ret.get("ranker_alpha"))

    vs = settings.get("vector_store_binding", {})
    mapping["__PROVIDER_TYPE__"] = _value(vs.get("provider_type"), "milvus")
    mapping["__COLLECTION_NAME__"] = _value(vs.get("collection_name"))

    indexing_params = output_data.get("indexing", {}).get("pipeline_spec", {}).get("parameters", {})
    mapping["__MAAS_SECRET_NAME__"] = _value(indexing_params.get("maas_secret_name"))
    mapping["__VECTOR_DB_SECRET_NAME__"] = _value(indexing_params.get("vector_db_secret_name"))

    return mapping


def _copy_template_tree(src: Path, dst: Path) -> None:
    """Recursively copy a template directory tree to *dst*."""
    if src.name in _IGNORED_TEMPLATE_NAMES or src.suffix == ".pyc":
        return

    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return

    for child in src.iterdir():
        if child.name in _IGNORED_TEMPLATE_NAMES or child.suffix == ".pyc":
            continue
        _copy_template_tree(child, dst / child.name)


def _replace_placeholders(file_path: Path, mapping: dict[str, str]) -> None:
    """Replace ``__PLACEHOLDER__`` markers in a text file."""
    content = file_path.read_text(encoding="utf-8")
    for marker, value in mapping.items():
        content = content.replace(marker, value)
    file_path.write_text(content, encoding="utf-8")


def _apply_provider_conditionals(file_path: Path, active_provider: str) -> None:
    """Keep blocks for *active_provider* and strip blocks for others."""
    content = file_path.read_text(encoding="utf-8")

    def _replacer(match: re.Match) -> str:
        provider = match.group("provider").lower()
        body = match.group("body")
        if provider == active_provider.lower():
            return body
        return ""

    content = _PROVIDER_BLOCK_PATTERN.sub(_replacer, content)
    file_path.write_text(content, encoding="utf-8")


def generate_starter_kit(
    output_data: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Generate a ``starter_kit.zip`` from pattern optimisation data.

    Parameters
    ----------
    output_data : dict[str, Any]
        The parsed ``pattern.json`` data produced by the optimisation pipeline.
    output_dir : str | Path
        Directory where ``starter_kit.zip`` will be written (typically the
        pattern directory alongside ``pattern.json``).

    Returns
    -------
    Path
        Absolute path to the generated ``starter_kit.zip``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = _create_starter_kit_mapping(output_data)
    active_provider = mapping.get("__PROVIDER_TYPE__", "milvus")

    template_root = importlib.resources.files("ai4rag.assets_generator").joinpath(
        "starter_kit_templates", "agentic_rag"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        kit_dir = Path(tmpdir) / "starter_kit"

        with importlib.resources.as_file(template_root) as resolved_root:
            _copy_template_tree(resolved_root, kit_dir)

        for fname in _CONFIGURABLE_FILES:
            target = kit_dir / fname
            if target.exists():
                _replace_placeholders(target, mapping)
                _apply_provider_conditionals(target, active_provider)

        zip_path = output_dir / "starter_kit.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(kit_dir.rglob("*")):
                if file_path.is_file():
                    arcname = str(Path("starter_kit") / file_path.relative_to(kit_dir))
                    zf.write(file_path, arcname)

    return zip_path
