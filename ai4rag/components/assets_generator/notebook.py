# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import importlib.resources
from json import dump as json_dump
from json import load as json_load
from pathlib import Path
from string import Formatter
from typing import Literal, Self


class NotebookCell:
    """Represents a single cell in a Jupyter notebook.

    Parameters
    ----------
    cell_type : {"code", "markdown"}
        The type of cell.
    source : str | list[str]
        The cell content. Can be a string or list of strings.
    metadata : dict, optional
        Cell metadata.
    """

    def __init__(
        self,
        cell_type: Literal["code", "markdown"],
        source: str | list[str],
        metadata: dict | None = None,
    ):
        self.cell_type = cell_type
        self.metadata = metadata or {}
        self.source = source

        if cell_type == "code":
            self.execution_count = None
            self.outputs: list = []

    def to_dict(self) -> dict:
        """Convert cell to notebook JSON format.

        Returns
        -------
        dict
            Cell in Jupyter notebook JSON format.
        """
        cell_dict = {
            "cell_type": self.cell_type,
            "metadata": self.metadata,
            "source": self.source,
        }

        if self.cell_type == "code":
            cell_dict["execution_count"] = self.execution_count
            cell_dict["outputs"] = self.outputs

        return cell_dict

    def format_source(self, placeholders_mapping: dict) -> Self:
        """Format cell source by substituting placeholders.

        Performs ``str.format``-style substitution on each line of the cell
        source.  Placeholders not present in *placeholders_mapping* are
        replaced with empty strings so that missing keys never raise.

        Parameters
        ----------
        placeholders_mapping : dict
            Mapping from placeholder names to replacement values.

        Returns
        -------
        Self
            This cell instance (mutated in-place) for method chaining.
        """
        if isinstance(self.source, list):
            new_source = []
            for line in self.source:
                line_mapping = {}
                for _, field_name, _, _ in Formatter().parse(line):
                    if field_name is None:
                        continue
                    line_mapping[field_name] = placeholders_mapping.get(field_name, "")

                new_source.append(line.format(**line_mapping))
            self.source = new_source

            return self

        self.source = self.source.format(**placeholders_mapping)

        return self


class Notebook:
    """Builder for programmatically creating and manipulating Jupyter notebooks.

    Provides a fluent API for building notebooks by adding cells, formatting
    content with placeholder substitution, and saving to disk.

    Parameters
    ----------
    kernel_name : str, default="python3"
        Kernel name for the notebook.
    kernel_display_name : str, default="Python 3"
        Display name for the kernel.
    language : str, default="python"
        Programming language.
    language_version : str, default="3.13.11"
        Language version.
    cells : list[NotebookCell] | None, default=None
        Notebook cells to build the notebook from.

    Examples
    --------
    >>> nb = Notebook(
    ...     cells=[
    ...         NotebookCell(
    ...             cell_type="markdown",
    ...             source="### Hello world!",
    ...         )
    ...     ]
    ... )
    >>> nb.save("output.ipynb")
    """

    def __init__(
        self,
        kernel_name: str = "python3",
        kernel_display_name: str = "Python 3",
        language: str = "python",
        language_version: str = "3.13.11",
        cells: list[NotebookCell] | None = None,
    ):
        self.cells: list[NotebookCell] = cells if cells else []
        self.metadata = {
            "kernelspec": {
                "display_name": kernel_display_name,
                "language": language,
                "name": kernel_name,
            },
            "language_info": {"name": language, "version": language_version},
        }
        self.nbformat = 4
        self.nbformat_minor = 4

    def to_dict(self) -> dict:
        """Convert notebook to dictionary format.

        Returns
        -------
        dict
            Notebook in Jupyter JSON format.
        """
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "metadata": self.metadata,
            "nbformat": self.nbformat,
            "nbformat_minor": self.nbformat_minor,
        }

    def save(self, path: str | Path, indent: int = 2) -> "Notebook":
        """Save notebook to a file.

        Parameters
        ----------
        path : str | Path
            Output file path.
        indent : int, default=2
            JSON indentation level.

        Returns
        -------
        Notebook
            Self for method chaining.

        Examples
        --------
        >>> nb = Notebook()
        >>> nb.save("output.ipynb")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w+", encoding="utf-8") as f:
            json_dump(self.to_dict(), f, indent=indent)

        return self

    @classmethod
    def load(
        cls,
        notebook_name: str,
        templates_dir: str | Path | None = None,
    ) -> "Notebook":
        """Load a Jupyter notebook template from bundled package data or a custom directory.

        Parameters
        ----------
        notebook_name : str
            Name of the template file (e.g. ``"ogx_indexing_template.ipynb"``).
        templates_dir : str | Path | None, default=None
            Directory containing the template notebooks.  When *None*,
            templates are loaded from the ``notebook_templates/`` sub-package
            bundled with ``ai4rag.assets_generator``.

        Returns
        -------
        Notebook
            A new Notebook instance populated with the loaded cells and metadata.

        Examples
        --------
        >>> nb = Notebook.load("ogx_indexing_template.ipynb")
        >>> nb = Notebook.load("custom.ipynb", templates_dir="/data/templates")
        """
        if templates_dir is not None:
            resolved_path = Path(templates_dir) / notebook_name
            with resolved_path.open("r", encoding="utf-8") as f:
                nb_dict = json_load(f)
        else:
            template_path = importlib.resources.files("ai4rag.components.assets_generator").joinpath(
                "notebook_templates", notebook_name
            )
            with importlib.resources.as_file(template_path) as resolved_path:
                with resolved_path.open("r", encoding="utf-8") as f:
                    nb_dict = json_load(f)

        loaded_cells = []
        for cell_data in nb_dict.get("cells", []):
            cell = NotebookCell(
                cell_type=cell_data.get("cell_type", "code"),
                source=cell_data.get("source", ""),
                metadata=cell_data.get("metadata", {}),
            )

            if cell.cell_type == "code":
                cell.execution_count = cell_data.get("execution_count")
                cell.outputs = cell_data.get("outputs", [])

            loaded_cells.append(cell)

        metadata = nb_dict.get("metadata", {})
        kernelspec = metadata.get("kernelspec", {})
        language_info = metadata.get("language_info", {})

        notebook = cls(
            kernel_name=kernelspec.get("name", "python3"),
            kernel_display_name=kernelspec.get("display_name", "Python 3"),
            language=language_info.get("name", "python"),
            language_version=language_info.get("version", "3.13.11"),
            cells=loaded_cells,
        )

        # Preserve exact original metadata and notebook format versions
        notebook.metadata = metadata
        notebook.nbformat = nb_dict.get("nbformat", 4)
        notebook.nbformat_minor = nb_dict.get("nbformat_minor", 4)

        return notebook
