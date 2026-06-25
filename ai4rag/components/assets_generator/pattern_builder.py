# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
def build_pattern_json(
    pattern: dict,
    detected_language: dict | None = None,
) -> dict:
    """Update pattern information with detected language and responses template.

    Parameters
    ----------
    pattern : dict
        A single evaluation result object carrying ``indexing_params``,
        ``rag_params``, ``pattern_name``, ``collection``, etc.
    detected_language : dict | None, default=None
        Language detection result (``{"code": "...", "name": "..."}``).

    Returns
    -------
    dict
        Pattern definition suitable for JSON serialisation.
    """
    if detected_language:
        pattern["settings"]["generation"]["detected_language"] = detected_language

    pattern["settings"]["responses_template"] = {
        "model": pattern["settings"]["generation"]["model_id"],
        "stream": False,
        "store": False,
        "input": "<user_query_placeholder>",
        "instructions": pattern["settings"]["generation"]["system_message_text"],
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": [pattern["settings"]["vector_store_binding"]["vector_store_id"]],
                "ranking_options": {
                    "max_num_results": pattern["settings"]["retrieval"]["number_of_chunks"],
                },
            },
        ],
        "include": ["file_search_call.results"],
    }

    retrieval_settings = pattern["settings"]["retrieval"]
    search_mode = retrieval_settings.get("search_mode")
    ranker_strategy = retrieval_settings.get("ranker_strategy")
    ranker_k = retrieval_settings.get("ranker_k")
    ranker_alpha = retrieval_settings.get("ranker_alpha")

    if search_mode == "hybrid" and ranker_strategy == "rrf" and ranker_k is not None and ranker_k > 0:
        pattern["settings"]["responses_template"]["tools"][0]["ranking_options"]["impact_factor"] = ranker_k
    elif search_mode == "hybrid" and ranker_strategy == "weighted" and ranker_alpha is not None and ranker_alpha != 1:
        pattern["settings"]["responses_template"]["tools"][0]["ranking_options"]["alpha"] = ranker_alpha

    return pattern
