# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import html
import json
from pathlib import Path

# Fallback keys when ai4rag uses a different schema (embedding vs embeddings, flat keys)
_CONFIG_COLUMN_FALLBACKS = {
    "embeddings.model_id": ("embedding_model", "embedding.model_id"),
    "generation.model_id": ("foundation_model",),
}

_LEADERBOARD_METRIC_COLUMNS = [
    "mean_answer_correctness",
    "mean_faithfulness",
    "mean_context_correctness",
]

_LEADERBOARD_CONFIG_COLUMNS = [
    "chunking.method",
    "chunking.chunk_size",
    "chunking.chunk_overlap",
    "embeddings.model_id",
    "retrieval.method",
    "retrieval.number_of_chunks",
    "retrieval.search_mode",
    "retrieval.ranker_strategy",
    "generation.model_id",
]


def _get_nested(params: dict, key: str):
    """Resolve a dotted key from a flat or nested dict (e.g. ``chunking.method``)."""
    if not params:
        return None
    if key in params:
        return params[key]
    parts = key.split(".", 1)
    if len(parts) == 2:
        outer = params.get(parts[0])
        if isinstance(outer, dict):
            return outer.get(parts[1])
    return None


def _get_config_value(merged: dict, col: str):
    """Get config column value with fallbacks for ai4rag schema differences."""
    val = _get_nested(merged, col)
    if val is not None:
        return val
    fallbacks = _CONFIG_COLUMN_FALLBACKS.get(col)
    if not fallbacks:
        return None
    for fallback in fallbacks:
        if "." in fallback:
            val = _get_nested(merged, fallback)
        else:
            val = merged.get(fallback) if merged else None
        if val is not None:
            return val
    return None


def _merge_params(indexing_params: dict, rag_params: dict) -> dict:
    """Merge indexing and RAG params into a single flat dict."""
    merged = dict(indexing_params or {})
    merged.update(rag_params or {})
    return merged


def _settings_from_rag_pattern(e: dict) -> dict | None:
    """Build leaderboard config dict from ``rag_pattern.settings`` (legacy nested schema)."""
    rp = (e.get("rag_pattern") or {}).get("settings")
    if not rp:
        return None
    return {
        "chunking": rp.get("chunking") or {},
        "embeddings": {"model_id": (rp.get("embedding") or {}).get("model_id")},
        "retrieval": {
            "method": rp.get("method"),
            "number_of_chunks": rp.get("number_of_chunks"),
        },
        "generation": {"model_id": (rp.get("generation") or {}).get("model_id")},
    }


def _normalize_flat_settings(settings: dict | None) -> dict | None:
    """Build leaderboard config dict from flat ``pattern.json`` settings."""
    if not settings:
        return None
    emb = settings.get("embedding") or settings.get("embeddings") or {}
    gen = settings.get("generation") or {}
    emb_model_id = emb.get("model_id") if isinstance(emb, dict) else None
    if emb_model_id is None and isinstance(settings.get("embedding_model"), str):
        emb_model_id = settings.get("embedding_model")
    gen_model_id = gen.get("model_id") if isinstance(gen, dict) else None
    if gen_model_id is None and isinstance(settings.get("foundation_model"), str):
        gen_model_id = settings.get("foundation_model")
    return {
        "chunking": settings.get("chunking") or {},
        "embeddings": {"model_id": emb_model_id},
        "retrieval": settings.get("retrieval") or {},
        "generation": {"model_id": gen_model_id},
    }


def _metric_key(name: str, evaluator: str | None) -> str:
    """Build a collision-free metric key, disambiguating shared names by evaluator.

    A metric name is not unique across evaluators (e.g. both the unitxt and RAGAS
    evaluators emit ``"faithfulness"``). The unitxt/custom metrics keep their bare
    name so the primary columns stay stable; any other evaluator's metrics are
    prefixed with the evaluator (e.g. ``"ragas_faithfulness"``) so they do not
    overwrite each other.
    """
    if evaluator and evaluator not in ("unitxt", "custom"):
        return f"{evaluator}_{name}"
    return name


def _get_aggregate_scores(e: dict) -> dict[str, dict]:
    """Extract aggregate metric scores as ``{key: {mean, ci_low, ci_high}}`` from either format.

    Keys are disambiguated by evaluator via :func:`_metric_key` so a metric name
    shared across evaluators (e.g. unitxt vs RAGAS ``faithfulness``) yields
    separate columns instead of one silently overwriting the other.
    """
    evaluation = e.get("evaluation")
    if isinstance(evaluation, dict):
        return {
            _metric_key(m["name"], m.get("evaluator")): m["scores"]
            for m in evaluation.get("metrics", [])
            if isinstance(m, dict)
        }
    raw = e.get("scores") or {}
    if isinstance(raw, list):
        return {_metric_key(m["name"], m.get("evaluator")): m["scores"] for m in raw if isinstance(m, dict)}
    aggregate = raw.get("scores") if isinstance(raw.get("scores"), dict) else raw
    return aggregate or {}


def _get_final_score(e: dict) -> float | None:
    """Extract the mean score of the metric marked with ``optimization_metric: True``."""
    metrics = e.get("evaluation", {}).get("metrics", [])
    for m in metrics:
        if isinstance(m, dict) and m.get("optimization_metric"):
            return m.get("scores", {}).get("mean")
    return None


def _metric_to_mean_key(metric: str) -> str:
    return "mean_" + metric


def _metric_display_name(metric: str) -> str:
    """Format metric name for display (e.g. ``answer_correctness`` to ``answer correctness``)."""
    if not metric:
        return "optimization metric"
    return metric.replace("_", " ").strip()


def _header_two_lines(label: str) -> str:
    """Split a column header by ``"."`` to produce a multi-line ``<th>`` cell."""
    if "." in label:
        parts = label.split(".", 1)
        line1 = parts[0]
        line2 = parts[1].replace("_", " ") if len(parts) > 1 else ""
        if line2:
            if " " in line2:
                last_space = line2.rfind(" ")
                line2a = line2[:last_space]
                line2b = line2[last_space + 1 :]
                return html.escape(line1) + "<br>" + html.escape(line2a) + "<br>" + html.escape(line2b)
            return html.escape(line1) + "<br>" + html.escape(line2)
    return html.escape(label.replace("_", " "))


def _build_leaderboard_html(
    header_row: str,
    table_body: str,
    best_pattern_name: str,
    num_patterns: int,
    eval_metric: str,
    colgroup_html: str = "",
) -> str:
    """Build a styled HTML document for the RAG leaderboard."""
    metric_label = html.escape(_metric_display_name(eval_metric))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Patterns Leaderboard</title>
  <style>
    :root {{
      --bg: #0f1419;
      --surface: #1a2332;
      --surface-hover: #243044;
      --border: #2d3a4f;
      --text: #e6edf3;
      --text-muted: #8b949e;
      --accent: #58a6ff;
      --accent-dim: #388bfd66;
      --gold: #f0b429;
      --silver: #a8b2c1;
      --bronze: #c9a227;
      --success: #3fb950;
      --radius: 12px;
      --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 2rem;
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      min-height: 100vh;
    }}
    .container {{
      margin: 0 auto;
      width: 100%;
      max-width: 960px;
    }}
    @media (min-width: 1200px) {{
      .container {{ max-width: 1100px; }}
    }}
    @media (min-width: 1400px) {{
      .container {{ max-width: 1300px; }}
    }}
    @media (min-width: 1600px) {{
      .container {{ max-width: 1500px; }}
    }}
    @media (min-width: 1920px) {{
      .container {{ max-width: min(1800px, 92vw); }}
    }}
    header {{
      margin-bottom: 2rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{
      margin: 0 0 0.25rem 0;
      font-size: 1.75rem;
      font-weight: 600;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 0.9rem;
    }}
    .badge {{
      display: inline-block;
      margin-left: 0.5rem;
      padding: 0.2rem 0.5rem;
      font-size: 0.75rem;
      font-weight: 600;
      border-radius: 6px;
      background: var(--accent-dim);
      color: var(--accent);
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }}
    .leaderboard-wrap {{
      overflow-x: auto;
    }}
    .leaderboard-wrap table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      table-layout: auto;
    }}
    .leaderboard-wrap th {{
      text-align: left;
      padding: 0.5rem 0.5rem;
      background: var(--surface-hover);
      color: var(--text-muted);
      font-weight: 600;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      border-bottom: 1px solid var(--border);
      line-height: 1.25;
    }}
    .leaderboard-wrap td {{
      padding: 0.5rem 0.5rem;
      border-bottom: 1px solid var(--border);
    }}
    .leaderboard-wrap th,
    .leaderboard-wrap td {{
      overflow-wrap: break-word;
      word-break: break-word;
    }}
    .leaderboard-wrap tr:last-child td {{
      border-bottom: none;
    }}
    .leaderboard-wrap tbody tr:hover {{
      background: var(--surface-hover);
    }}
    .leaderboard-wrap tbody tr.rank-1 {{
      background: linear-gradient(90deg, rgba(240,180,41,0.12) 0%, transparent 100%);
    }}
    .leaderboard-wrap tbody tr.rank-1 td:first-child {{
      color: var(--gold);
      font-weight: 700;
    }}
    .leaderboard-wrap .metric-cell {{
      font-variant-numeric: tabular-nums;
      color: var(--success);
    }}
    .best-model-footer {{
      margin-top: 1.5rem;
      padding: 1rem 1.25rem;
      background: var(--surface-hover);
      border-radius: 8px;
      font-size: 0.9rem;
      color: var(--text-muted);
    }}
    .best-model-footer strong {{
      color: var(--gold);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>RAG Patterns Leaderboard</h1>
      <p class="subtitle">
        Ranked by <span class="badge">{metric_label}</span> (best first) · {num_patterns} pattern(s)
      </p>
    </header>
    <div class="card">
      <div class="leaderboard-wrap">
        <table>
{colgroup_html}
          <thead>
            <tr>{header_row}</tr>
          </thead>
          <tbody>
            {table_body}
          </tbody>
        </table>
      </div>
    </div>
    <div class="best-model-footer">
      Best pattern: <strong>{html.escape(best_pattern_name)}</strong>
    </div>
  </div>
</body>
</html>"""


def build_leaderboard_html(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    patterns_dir: str | Path,
    optimization_metric: str = "faithfulness",
    optimization_metric_evaluator: str | None = None,
) -> str:
    """Build a styled HTML leaderboard from RAG pattern evaluation results.

    Scans each subdirectory of *patterns_dir* for a ``pattern.json`` file,
    ranks patterns by the specified optimisation metric, and produces a
    self-contained HTML document with a responsive leaderboard table.

    Parameters
    ----------
    patterns_dir : str | Path
        Directory of RAG patterns; each subdirectory must contain a
        ``pattern.json`` file.
    optimization_metric : str, default="faithfulness"
        Name of the metric used to rank patterns (e.g.
        ``"faithfulness"``, ``"answer_correctness"``).
    optimization_metric_evaluator : str | None, default=None
        Evaluator that produced ``optimization_metric`` (e.g. ``"ragas"``).
        Required to disambiguate metric names shared across evaluators (see
        :func:`_metric_key`); omit it only when the name is unambiguous
        (unitxt/custom metrics, whose columns already use the bare name).

    Returns
    -------
    str
        Self-contained HTML document with the leaderboard table.

    Raises
    ------
    FileNotFoundError
        If *patterns_dir* does not exist or is not a directory.
    """
    rag_patterns_dir = Path(patterns_dir)
    if not rag_patterns_dir.is_dir():
        raise FileNotFoundError(f"rag_patterns path is not a directory: {rag_patterns_dir}")

    evaluations: list[dict] = []
    for subdir in sorted(rag_patterns_dir.iterdir()):
        if not subdir.is_dir():
            continue
        pattern_file = subdir / "pattern.json"
        if not pattern_file.is_file():
            continue
        with pattern_file.open("r", encoding="utf-8") as f:
            evaluations.append(json.load(f))

    # Sort by optimization metric score descending; missing scores last
    def _optimization_score(e: dict) -> tuple[bool, float]:
        v = _get_final_score(e)
        if v is not None:
            try:
                return (False, -float(v))
            except (TypeError, ValueError):
                pass
        for info in _get_aggregate_scores(e).values():
            if isinstance(info, dict):
                mean = info.get("mean")
                if mean is not None:
                    try:
                        return (False, -float(mean))
                    except (TypeError, ValueError):
                        pass
        return (True, 0)

    evaluations.sort(key=_optimization_score)

    # Discover metric columns present in data
    all_metric_names: list[str] = []
    for e in evaluations:
        for m in _get_aggregate_scores(e):
            if m not in all_metric_names:
                all_metric_names.append(m)
    metric_columns = [c for c in _LEADERBOARD_METRIC_COLUMNS if c.replace("mean_", "", 1) in all_metric_names]
    for m in all_metric_names:
        col = _metric_to_mean_key(m)
        if col not in metric_columns:
            metric_columns.append(col)

    # Place the optimisation metric column right after Pattern_Name. The key must go
    # through `_metric_key` too, or an evaluator-shared name (e.g. "faithfulness")
    # would resolve to the unitxt/custom column even when RAGAS drove optimization.
    opt_metric_key = _metric_key(optimization_metric or "faithfulness", optimization_metric_evaluator)
    opt_metric_col = _metric_to_mean_key(opt_metric_key)
    if opt_metric_col in metric_columns:
        other_metrics = [c for c in metric_columns if c != opt_metric_col]
        metric_columns = [opt_metric_col] + other_metrics

    config_columns = list(_LEADERBOARD_CONFIG_COLUMNS)
    headers = ["Pattern_Name"] + metric_columns + config_columns
    header_row = "".join(f"<th>{_header_two_lines(h)}</th>" for h in headers)

    # Build rows and collect cell values for dynamic column width computation
    rows_cells: list[list[str]] = []
    rows: list[str] = []
    for i, e in enumerate(evaluations):
        pattern_name = e.get("name") or e.get("pattern_name") or (e.get("rag_pattern") or {}).get("name", "—")
        scores = _get_aggregate_scores(e)
        merged = (
            _settings_from_rag_pattern(e)
            or _normalize_flat_settings(e.get("settings"))
            or _merge_params(e.get("indexing_params") or {}, e.get("rag_params") or {})
        )

        cells: list[str] = [str(pattern_name)]
        for col in metric_columns:
            metric_name = col.replace("mean_", "", 1)
            info = scores.get(metric_name) or {}
            mean = info.get("mean")
            if mean is not None:
                cell = f"{mean:.4f}" if isinstance(mean, (int, float)) else str(mean)
            else:
                cell = ""
            cells.append(cell)
        for col in config_columns:
            val = _get_config_value(merged, col) if merged else None
            if val is not None and (val != "" or col != "retrieval.ranker_strategy"):
                if isinstance(val, dict):
                    cells.append(json.dumps(val, sort_keys=True, ensure_ascii=False))
                else:
                    cells.append(str(val))
            elif col == "retrieval.ranker_strategy":
                cells.append("-")
            else:
                cells.append("")
        rows_cells.append(cells)
        tr_class = ' class="rank-1"' if i == 0 else ""
        rows.append("<tr" + tr_class + ">" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")

    table_body = "".join(rows)

    # Dynamic column widths from content
    ncols = len(headers)
    column_max_len = [
        max(
            len(headers[i]),
            max((len(rows_cells[r][i]) for r in range(len(rows_cells)))) if rows_cells else 0,
        )
        for i in range(ncols)
    ]
    width_rem: list[float] = []
    for i in range(ncols):
        if i in (7, 12):
            min_rem = 18  # embeddings.model_id, generation.model_id
        elif i == 0:
            min_rem = 10
        elif i == 1:
            min_rem = 12
        else:
            min_rem = 4
        if "." in headers[i]:
            parts = headers[i].split(".", 1)
            line1_len = len(parts[0])
            line2 = parts[1].replace("_", " ") if len(parts) > 1 else ""
            if " " in line2:
                last_space = line2.rfind(" ")
                seg_lens = [line1_len, len(line2[:last_space]), len(line2[last_space + 1 :])]
            else:
                seg_lens = [line1_len, len(line2)] if line2 else [line1_len]
            min_for_two_line = max(seg_lens) * 1.0
            min_rem = max(min_rem, min_for_two_line)
        w = max(min_rem, min(32, 0.6 * column_max_len[i]))
        width_rem.append(w)
    colgroup_html = (
        "          <colgroup>\n"
        + "\n".join(f'            <col style="width: {width_rem[i]:.1f}rem">' for i in range(ncols))
        + "\n          </colgroup>"
    )

    best_pattern_name = "—"
    if evaluations:
        best_pattern_name = (
            evaluations[0].get("name")
            or evaluations[0].get("pattern_name")
            or (evaluations[0].get("rag_pattern") or {}).get("name", "—")
        )
        best_pattern_name = str(best_pattern_name)

    return _build_leaderboard_html(
        header_row=header_row,
        table_body=table_body,
        best_pattern_name=best_pattern_name,
        num_patterns=len(evaluations),
        eval_metric=opt_metric_key,
        colgroup_html=colgroup_html,
    )
