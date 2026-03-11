# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from json import dump as json_dump
from string import Formatter
from typing import Literal, Self
from pathlib import Path


class AssetGenerationError(Exception):
    """Error raised when error will occur during asseet generation."""

    pass


class NotebookCell:
    """
    Represents a single cell in a Jupyter notebook.

    Parameters
    ----------
    cell_type : Literal["code", "markdown"]
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
            self.outputs = []

    def to_dict(self) -> dict:
        """
        Convert cell to notebook JSON format.

        Returns
        -------
        dict
            Cell in notebook format.
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

    def format_source(
        self,
        placeholders_mapping: dict,
    ) -> Self:
        """
        Formats cell source based on provided placeholders_mapping.

        Returns
        -------
        Self
            Instance of NotebookCell.
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
    """
    Builder class for creating and manipulating Jupyter notebooks.

    This class provides a fluent API for programmatically building notebooks
    by adding code and markdown cells, formatting content, and saving to disk.

    Parameters
    ----------
    kernel_name : str, default="python3"
        Kernel name for the notebook.
    kernel_display_name : str, default="Python 3"
        Display name for the kernel.
    language : str, default="python"
        Programming language.
    language_version : str, default="3.11.0"
        Language version.
    cells : list[NotebookCell] | None, default=None
        Notebook cells to build the notebook from.

    Examples
    --------
    >>> nb = Notebook(
        cells=[
            NotebookCell(
                cell_type="markdown",
                source="### Hello world!",
            )
        ])
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
        """
        Convert notebook to dictionary format.

        Returns
        -------
        dict
            Notebook in JSON format.
        """
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "metadata": self.metadata,
            "nbformat": self.nbformat,
            "nbformat_minor": self.nbformat_minor,
        }

    def save(self, path: str | Path, indent: int = 2) -> "Notebook":
        """
        Save notebook to a file.

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

        with path.open("w+") as f:
            json_dump(self.to_dict(), f, indent=indent)

        return self
