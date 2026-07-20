# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".pptx",
        ".md",
        ".html",
        ".txt",
        # New formats
        ".odt",
        ".odp",
        ".adoc",
        ".tex",
        ".epub",
        ".eml",
        ".qmd",
        ".Rmd",
        ".xhtml",
    }
)
