# ai4rag Documentation

This directory contains the source files for the ai4rag documentation site, built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## Building Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

### Development Server

Start a live-reloading development server:

```bash
mkdocs serve
```

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) to view the documentation.

Changes to source files will automatically trigger a rebuild.

### Build Static Site

Build the documentation site:

```bash
mkdocs build
```

The built site will be in the `site/` directory.

### Strict Mode

Build with strict mode to catch warnings as errors:

```bash
mkdocs build --strict
```

Use this before committing to ensure documentation quality.

## Documentation Structure

```
docs/
├── index.md                    # Homepage
├── getting-started/            # Installation, quick start, configuration
├── user-guide/                 # Search space, optimizers, evaluation
├── architecture/               # System architecture documentation
├── api-reference/              # Auto-generated API documentation
├── development/                # Contributing, workflow, testing
├── about/                      # License, changelog
├── stylesheets/                # Custom CSS
├── javascripts/                # Custom JavaScript
└── includes/                   # Reusable content (abbreviations)
```

## Writing Documentation

### Markdown Files

- Use standard Markdown syntax
- Follow the Material for MkDocs [reference guide](https://squidfunk.github.io/mkdocs-material/reference/)
- Include license header in all files

### Code Examples

Use fenced code blocks with language specifiers:

````markdown
```python
from ai4rag.core.experiment import AI4RAGExperiment

experiment = AI4RAGExperiment(...)
```
````

### Admonitions

Use admonitions for notes, tips, warnings:

```markdown
!!! note
    This is a note.

!!! tip
    This is a helpful tip.

!!! warning
    This is a warning.

!!! important
    This is important information.
```

### API Documentation

API reference is auto-generated from docstrings using mkdocstrings:

```markdown
::: ai4rag.core.experiment.experiment.AI4RAGExperiment
    options:
      show_root_heading: true
      show_source: true
```

Ensure all public classes and functions have complete Google-style docstrings.

## Deployment

Documentation is automatically deployed to GitHub Pages via GitHub Actions:

- **main branch**: Deployed as `latest` (default version)
- **dev branch**: Deployed as `dev` version
- **tags (vX.Y.Z)**: Deployed as versioned documentation

See `.github/workflows/docs.yml` for the deployment workflow.

## Versioning

Documentation versioning is handled by [mike](https://github.com/jimporter/mike):

### Available Versions

List deployed versions:

```bash
mike list
```

### Deploy a New Version

```bash
# Deploy version 0.2.0 and alias as latest
mike deploy --push --update-aliases 0.2.0 latest

# Set latest as default
mike set-default --push latest
```

### View Versions Locally

```bash
mike serve
```

## Configuration

Main configuration file: `mkdocs.yml` (project root)

Key settings:

- **Theme**: Material for MkDocs with dark/light toggle
- **Plugins**: search, minify, git-revision-date, mkdocstrings
- **Extensions**: Admonitions, code highlighting, tables, etc.
- **Navigation**: Page tree structure

## Style Guide

- Use clear, concise language
- Provide code examples
- Link to related documentation
- Use consistent terminology (see `includes/abbreviations.md`)
- Follow the [Google Developer Documentation Style Guide](https://developers.google.com/style)

## Getting Help

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs Documentation](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings Documentation](https://mkdocstrings.github.io/)

## License

Copyright © 2025-2026 IBM Corp.
SPDX-License-Identifier: Apache-2.0
