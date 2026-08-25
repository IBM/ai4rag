Review and update project documentation to reflect code changes since the latest release.

## Steps

### 1. Gather context

Run the following in parallel:
- Find the latest tag: `git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1`
- Get the list of changed files since that tag: `git diff <latest-tag>..HEAD --name-only`
- Get the diff summary: `git diff <latest-tag>..HEAD --stat`
- Get the commit log: `git log <latest-tag>..HEAD --oneline`

### 2. Map changed source files to documentation pages

Using the changed-files list from Step 1, identify which documentation pages are potentially affected. Use the following mapping — a changed source file implies the corresponding documentation page(s) should be reviewed:

| Source path pattern | Documentation pages to review |
|---|---|
| `ai4rag/core/experiment*` | `docs/api-reference/core/experiment.md`, `docs/user-guide/overview.md` |
| `ai4rag/core/hpo/` | `docs/api-reference/core/hpo.md`, `docs/user-guide/optimizers.md` |
| `ai4rag/core/pre_selector*` | `docs/api-reference/core/hpo.md`, `docs/architecture/core-components.md` |
| `ai4rag/search_space/` | `docs/api-reference/search-space/search-space.md`, `docs/user-guide/search-space.md` |
| `ai4rag/rag/chunking/` | `docs/api-reference/rag/chunking.md` |
| `ai4rag/rag/embedding/` | `docs/api-reference/rag/embedding.md` |
| `ai4rag/rag/vector_store/` | `docs/api-reference/rag/vector-stores.md`, `docs/user-guide/hybrid-search.md` |
| `ai4rag/rag/retrieval/` | `docs/api-reference/rag/retrieval.md` |
| `ai4rag/rag/foundation_model/` | `docs/api-reference/rag/foundation-models.md` |
| `ai4rag/rag/template/` | `docs/architecture/rag-components.md`, `docs/architecture/data-flow.md` |
| `ai4rag/evaluator/` | `docs/api-reference/evaluator/evaluator.md`, `docs/user-guide/evaluation.md` |
| `ai4rag/utils/event_handler*` | `docs/user-guide/event-handlers.md` |
| `pyproject.toml` | `docs/getting-started/installation.md`, `README.md` |
| New or removed modules | `docs/architecture/overview.md`, `docs/architecture/core-components.md`, `docs/architecture/rag-components.md` |

Also always include `README.md` and `docs/getting-started/quick-start.md` in the review when any public API signature changed (new parameters, renamed classes, changed defaults).

If no source files under `ai4rag/` changed, report that no documentation review is needed and stop here.

### 3. Review affected documentation pages

For each documentation page identified in Step 2, read both the documentation page and the corresponding source file(s). Check for each of the following issues:

- **Stale references**: class names, function names, parameter names, or module paths mentioned in the docs that no longer exist in the source code
- **Undocumented additions**: new public classes, functions, parameters, or configuration options added since the last tag that are not yet covered in docs
- **Outdated code examples**: snippets in the docs that use old API signatures, removed parameters, or renamed imports
- **Dependency drift**: version requirements in `pyproject.toml` that changed but are not reflected in `docs/getting-started/installation.md` or `README.md`
- **Architecture gaps**: structural changes (new modules, moved files, changed inheritance) not reflected in architecture docs
- **Incorrect descriptions**: behavioural changes that make existing prose inaccurate (e.g., a default value changed, a validation rule added or removed)

### 4. Present findings and apply updates

Produce a summary table of findings before making any edits:

```
| Documentation page | Issue | Action |
|---|---|---|
| docs/api-reference/rag/vector-stores.md | `LSVectorStore` renamed to `MilvusVectorStore` | Update all references |
| docs/getting-started/installation.md | openai bumped to ~=2.1.0 | Update version requirement |
| ... | ... | ... |
```

If the table is empty, state "No documentation updates required — all pages are consistent with the current code." and skip to Step 5.

Otherwise, apply the updates to each affected documentation page. When editing:
- Preserve the existing formatting, heading structure, and admonition style
- Keep code examples syntactically correct and runnable
- Use the same terminology conventions as the rest of the docs
- Do not add or remove sections unless the code changes warrant it
- For API reference pages that use `mkdocstrings` directives (`::: module.path`), verify the module path is still valid rather than rewriting prose

### 5. Verify documentation build

Run `mkdocs build --strict` to confirm:
- No broken cross-references or missing pages
- No warnings about undefined references
- Build completes successfully

If the build fails, fix the issues and re-run until it passes.

### 6. Report

Tell the user:
- How many documentation pages were reviewed
- How many required updates (with the summary table from Step 4)
- Whether the docs build passed
- Any pages that may need deeper manual review (e.g., user-guide pages where behavioural nuance is hard to verify mechanically)
