# Changelog

All notable changes to ai4rag will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.1](https://github.com/IBM/ai4rag/releases/tag/v0.2.1)

### Changed
- Default optimizer is now `GAMOptimizer`
- Default retrieval methods no longer contain `window` method, as this is not supported for `ls_milvus` at the moment
- `Parameter` no longer requires to specify `param_type`. `C` type is used as default

### Fixed
- Bug in `GAMOptimizer` that unabled its usage (failing during deepcopy)

## [0.2.0](https://github.com/IBM/ai4rag/releases/tag/v0.2.0)

### Added
- Support for `LocalEventHandler`
- Support for external models introduced via `OpenAI` client
- CI/CD tooling
- Added RAG pattern object streaming with and added it to results, so that pattern can be reused post experiment


### Fixed
- Documentation and `README.md` update
- Updated samples
- Updated docstrings

### Changed
- Loose required parameters for `AI4RAGExperiment`
- Change "Optimiser" to "Optimizer" in all references

## [0.1.0](https://github.com/IBM/ai4rag/releases/tag/v0.1.0)

### Added
- Initial working implementation of `ai4rag` that can be used with `llama-stack` for RAG Template optimization

---

### Version Numbering

ai4rag follows [Semantic Versioning](https://semver.org/):

- **Major.Minor.Patch** (e.g., 1.2.3)
- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

---

## Release Process

Releases are created by merging the `dev` branch into `main` with a merge commit and git tag.

See [Development Workflow](../development/workflow.md#creating-a-release) for detailed release procedures.

---

## Stay Updated

- Watch the [GitHub repository](https://github.com/IBM/ai4rag) for releases
- Subscribe to release notifications
- Check the [releases page](https://github.com/IBM/ai4rag/releases) for version history
