# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
def build_pattern_json(
    pattern: dict,
) -> dict:
    """Update pattern information with detected language and responses template.

    Parameters
    ----------
    pattern : dict
        A single evaluation result object carrying ``indexing_params``,
        ``rag_params``, ``pattern_name``, ``collection``, etc.

    Returns
    -------
    dict
        Pattern definition suitable for JSON serialisation.
    """
    pattern["settings"]["responses_template"] = {
        "model": pattern["settings"]["generation"]["model_id"],
        "stream": False,
        "store": False,
        "input": [
            {
                "content": [{"text": pattern["settings"]["generation"]["system_message_text"], "type": "input_text"}],
                "role": "system",
            },
            {"content": [{"text": "<user_query_placeholder>", "type": "input_text"}], "role": "user"},
        ],
        "max_output_tokens": pattern["settings"]["generation"]["max_completion_tokens"],
        "temperature": pattern["settings"]["generation"]["temperature"],
        "tool_choice": {"mode": "required", "tools": [{}], "type": "file_search"},
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": [pattern["settings"]["vector_store_binding"]["vector_store_id"]],
                "max_num_results": pattern["settings"]["retrieval"]["number_of_chunks"],
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
        pattern["settings"]["responses_template"]["tools"][0]["ranking_options"] = {
            "ranker": "rrf",
            "impact_factor": ranker_k,
        }
    elif search_mode == "hybrid" and ranker_strategy == "weighted" and ranker_alpha is not None and ranker_alpha != 1:
        pattern["settings"]["responses_template"]["tools"][0]["ranking_options"] = {
            "ranker": "weighted",
            "alpha": ranker_alpha,
        }
    else:
        pattern["settings"]["responses_template"]["tools"][0]["ranking_options"] = {
            # simulate semantic-only search
            "ranker": "weighted",
            "alpha": 1.0,
        }

    return pattern
