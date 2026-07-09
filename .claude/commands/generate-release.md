Generate a changelog and release notes for the current branch.

## Steps

### 1. Gather context

Run the following in parallel:
- Find the latest tag matching `vX.Y.Z` pattern: `git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1`
- Read the current version from `ai4rag/__init__.py`
- Get commits between that tag and HEAD: `git log <latest-tag>..HEAD --oneline`
- Get the full diff summary: `git diff <latest-tag>..HEAD --stat`

### 2. Assess the version

Read the current `__version__` from `ai4rag/__init__.py`. Then reason explicitly about whether the version bump already applied is appropriate given the nature of the commits since the latest tag:

- **Patch** (X.Y.Z+1): only bug fixes, doc updates, minor internal changes — no new public API
- **Minor** (X.Y+1.0): new features or capabilities added, backward compatible
- **Major** (X+1.0.0): breaking changes to public API or behaviour

State your conclusion clearly:
- If the version in `__init__.py` already reflects the right bump relative to the latest tag, confirm it and proceed.
- If the version is insufficient (e.g., new features added but only patch was bumped), suggest the correct version and explain why. Ask the user to confirm before proceeding.

Use the confirmed version (from `__init__.py` or the suggested one) as `RELEASE_VERSION` for all subsequent steps.

### 3. Categorise the changes

Analyse every commit since the latest tag (read relevant changed files if needed for clarity) and categorise changes into:
- **Added**: new features, new parameters, new modules, new integrations
- **Changed**: updates to existing behaviour, refactoring, dependency bumps, renamed things
- **Fixed**: bug fixes
- **Removed**: deleted features or deprecated items
- **Deprecated**: features marked for future removal

Omit categories that have no entries. Each bullet should be concrete and user-facing (what changed, not the internal implementation detail).

**Every bullet must start with a bold component label** that tells the reader *where* in the project the change lives. Use logical component names (not file paths) — the label should match how a developer thinks about the area. Examples:

- `**Search space preparation** — added chunk_overlaps parameter on prepare_search_space_report()...`
- `**RAG optimization component** — renamed max_threads to inference_max_threads...`
- `**Dependencies** — removed the langchain meta-package...`
- `**Chunker** — DoclingChunker now supports...`
- `**Vector store** — OGXVectorStore.search() returns...`

If a single change spans multiple components, pick the one most relevant to the user; if genuinely cross-cutting, use a broad label like **Core** or **API**. Use existing labels from prior changelog entries when applicable for consistency. Review older entries in `docs/about/changelog.md` to align with established conventions.

### 4. Update `docs/about/changelog.md`

Read the existing `docs/about/changelog.md` to understand the format and conventions (Keep a Changelog style, links to GitHub releases). Insert a new section for `RELEASE_VERSION` immediately after the `---` separator that follows the file header (before the current first release entry). Follow the exact same markdown structure used by existing entries:

```markdown
## [X.Y.Z](https://github.com/IBM/ai4rag/releases/tag/vX.Y.Z)

### Added
- **Component name** — description of what was added...

### Changed
- **Component name** — description of what changed...

### Fixed
- **Component name** — description of what was fixed...

---
```

Each bullet follows the pattern: `**Component label** — concise description`. The component label is a logical area name (e.g. "Search space preparation", "RAG optimization component", "Dependencies"), not a file path.

Do not touch the version numbering explanation or the "Release Process" / "Stay Updated" sections at the bottom.

### 5. Generate the release description

Create the file `local/releases/release-X.Y.Z.md` (create `local/releases/` directory if it does not exist). This file is a PR / release description in GitHub-flavoured markdown. It should include:

```markdown
# Release vX.Y.Z

## Summary

<2-4 sentence plain-English summary of the overall theme of this release — what it enables for users.>

## Changes

### Added
- ...

### Changed
- ...

### Fixed
- ...

## Migration notes

<Any steps a user upgrading from the previous version needs to take. If there are no breaking changes or config changes, write "No breaking changes.">

## Checklist

- [ ] `__version__` in `ai4rag/__init__.py` updated to `X.Y.Z`
- [ ] `docs/about/changelog.md` updated
- [ ] All tests pass (`pytest`)
- [ ] Docs build successfully
```

After writing the file, tell the user:
- The release version used and whether the version in `__init__.py` was confirmed or needs updating.
- The paths of the two files written.
- If the version in `__init__.py` needs to be changed, remind the user to update it.
- Remind the user to run `/update-docs` to review and update project documentation before tagging the release.
