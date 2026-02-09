# Development Workflow

This document describes the development workflow for contributing to ai4rag.

---

## Branch Strategy

ai4rag uses a **two-branch workflow**:

- **`main`**: Production-ready releases only
- **`dev`**: Active development and integration

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'git0':'#0f62fe', 'git1':'#08bdba', 'gitBranchLabel0':'#ffffff', 'gitBranchLabel1':'#ffffff', 'commitLabelColor':'#ffffff', 'commitLabelBackground':'#525252', 'tagLabelColor':'#ffffff', 'tagLabelBackground':'#0f62fe'}}}%%
gitGraph
    commit id: "Initial"
    branch dev
    checkout dev
    commit id: "Feature A"
    commit id: "Feature B"
    commit id: "Feature C"
    checkout main
    merge dev id: "Release v0.2.0" tag: "v0.2.0"
    checkout dev
    commit id: "Feature D"
    commit id: "Feature E"
    checkout main
    merge dev id: "Release v0.3.0" tag: "v0.3.0"
```

---

## Pull Request Workflow

### 1. Create a Feature Branch

All development starts from `dev`:

```bash
# Ensure dev is up to date
git checkout dev
git pull origin dev

# Create a feature branch
git checkout -b feature/your-feature-name
```

**Branch Naming Conventions:**

- `feature/` - New features (e.g., `feature/add-hybrid-retrieval`)
- `fix/` - Bug fixes (e.g., `fix/chunk-overlap-validation`)
- `docs/` - Documentation updates (e.g., `docs/improve-quickstart`)
- `refactor/` - Code refactoring (e.g., `refactor/search-space-validation`)
- `test/` - Test improvements (e.g., `test/add-chunking-tests`)

### 2. Make Changes

Develop your feature following our [code style guidelines](code-style.md):

```bash
# Make changes to files
# ...

# Run tests
pytest

# Check code quality
black ai4rag/
pylint ai4rag/

# Commit with DCO sign-off
git commit -s -m "Add hybrid retrieval method

Implements hybrid retrieval combining dense and sparse search.

Signed-off-by: Your Name <your.email@example.com>"
```

!!! warning "Developer Certificate of Origin"
    All commits must include a sign-off (`git commit -s`) indicating acceptance of the [DCO](https://developercertificate.org/).

### 3. Push and Create PR

```bash
# Push your branch
git push origin feature/your-feature-name

# Create a PR on GitHub targeting the `dev` branch
```

**PR Guidelines:**

- **Title**: Clear, concise description (e.g., "Add hybrid retrieval method")
- **Description**:
    - What: Summary of changes
    - Why: Motivation and context
    - How: Approach taken
    - Testing: How you verified it works
- **Link Issues**: Reference related issues (e.g., "Closes #123")
- **Request Reviews**: Tag relevant maintainers

### 4. Code Review Process

Maintainers will review your PR and may request changes:

```bash
# Make requested changes
# ...

# Commit and push updates
git commit -s -m "Address review comments"
git push origin feature/your-feature-name
```

**Review Requirements:**

- At least **1 LGTM** (Looks Good To Me) from maintainers
- All CI checks passing (tests, linters)
- No merge conflicts with `dev`

### 5. PR Merge (Squash)

Once approved, maintainers will **squash and merge** your PR into `dev`:

- All commits are squashed into a single commit
- Commit message is the PR title and description
- Feature branch is deleted after merge

```
feature/your-feature-name → dev (squash merge)
```

!!! note "Why Squash?"
    Squashing keeps the `dev` branch history clean with one commit per feature/fix, making it easier to track changes.

---

## Release Workflow

### Creating a Release

Releases are created by merging `dev` into `main` with a **merge commit** (not squash):

```bash
# 1. Ensure dev is ready for release
git checkout dev
git pull origin dev

# 2. Run full test suite
pytest

# 3. Update version in ai4rag/version.py
# Edit file: __version__ = "0.3.0"

# 4. Update CHANGELOG.md
# Add release notes

# 5. Commit version bump
git commit -s -m "Bump version to 0.3.0"
git push origin dev

# 6. Merge dev into main (merge commit, not squash)
git checkout main
git pull origin main
git merge --no-ff dev -m "Release v0.3.0

Merge dev branch for version 0.3.0 release.

Changes:
- Feature A
- Feature B
- Bug fix C
"

# 7. Tag the release
git tag -a v0.3.0 -m "Release version 0.3.0"

# 8. Push main and tags
git push origin main
git push origin v0.3.0

# 9. Create GitHub Release
# Use GitHub UI to create release from tag with changelog
```

**Release Commit Structure:**

```
main branch:
  - Release v0.1.0 (merge commit from dev)
  - Release v0.2.0 (merge commit from dev)
  - Release v0.3.0 (merge commit from dev)

dev branch:
  - Feature A (squashed PR)
  - Feature B (squashed PR)
  - Bug fix C (squashed PR)
  - Feature D (squashed PR)
```

!!! warning "Merge Commits for Releases"
    Always use `--no-ff` when merging `dev` into `main` to preserve the development history and make releases clearly identifiable.

---

## Workflow Summary

```mermaid
graph TB
    A[Create feature branch from dev] --> B[Develop and commit]
    B --> C[Push and open PR to dev]
    C --> D[Code review]
    D --> E{Approved?}
    E -->|No| B
    E -->|Yes| F[Squash merge to dev]
    F --> G{Ready for release?}
    G -->|No| A
    G -->|Yes| H[Merge dev to main]
    H --> I[Tag release]
```

---

## Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) style:

```
<type>: <subject>

<body>

<footer>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**

```
feat: add hybrid retrieval method

Implements a hybrid retrieval strategy combining dense vector
search and sparse keyword search for improved recall.

- Add HybridRetriever class
- Update search space to support hybrid mode
- Add tests for hybrid retrieval

Closes #42

Signed-off-by: John Doe <john.doe@example.com>
```

---

## CI/CD Pipeline

### Continuous Integration

On every PR to `dev`: Coming soon...

### Continuous Deployment

On merge to `main`: Coming soon...

---

## Local Development Setup

### Initial Setup

```bash
# Clone repository
git clone https://github.com/IBM/ai4rag.git
cd ai4rag

# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks (optional but recommended)
# pre-commit install
```

### Development Cycle

```bash
# 1. Create feature branch
git checkout dev
git pull origin dev
git checkout -b feature/my-feature

# 2. Make changes
# ... edit files ...

# 3. Format code
black ai4rag/
isort ai4rag/

# 4. Run linter
pylint ai4rag/

# 5. Run tests
pytest

# 6. Run tests with coverage
pytest --cov=ai4rag --cov-report=html

# 7. Build docs locally
mkdocs serve  # Visit http://127.0.0.1:8000

# 8. Commit with sign-off
git add .
git commit -s

# 9. Push and create PR
git push origin feature/my-feature
```

---

## Testing Requirements

All PRs must include tests:

- **Unit Tests**: For new functions/classes
- **Integration Tests**: For component interactions
- **Coverage**: Maintain or improve coverage (target: >80%)

```bash
# Run specific test file
pytest tests/ai4rag/core/test_experiment.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=ai4rag --cov-report=term-missing
```

See [Testing Guide](testing.md) for detailed testing practices.

---

## Documentation Requirements

PRs with new features must include documentation:

- **Docstrings**: All public classes and functions
- **User Guide**: Update relevant user guide pages
- **API Reference**: Auto-generated from docstrings
- **Examples**: Add examples for new features

```bash
# Build docs locally
mkdocs serve

# Check for broken links
mkdocs build --strict
```

---

## Version Numbering

ai4rag follows [Semantic Versioning](https://semver.org/):

- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

**Examples:**

- `0.1.0 → 0.2.0`: New feature added
- `0.2.0 → 0.2.1`: Bug fix
- `0.9.0 → 1.0.0`: Breaking API change

---

## Getting Help

- **Questions**: Open a [discussion](https://github.com/IBM/ai4rag/discussions)
- **Bugs**: Open an [issue](https://github.com/IBM/ai4rag/issues)

---

## Next Steps

- [Contributing Guide](contributing.md) - Detailed contribution guidelines
- [Testing Guide](testing.md) - Testing best practices
- [Code Style](code-style.md) - Code formatting and style rules
