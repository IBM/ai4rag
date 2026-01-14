# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Iterator

import pandas as pd


class BenchmarkDataValueError(ValueError):
    """Error representing incorrect value given in the benchmark dataset"""


class BenchmarkData:
    """
    Class representing benchmarking dataset given to the AI4RAGExperiment
    introducing user-friendly, specified interface.

    Parameters
    ----------
    benchmark_data : pandas.DataFrame
        Benchmark Data given as pd.df. It should be tabular data that contains
        questions, answers and documents_ids columns.

    Attributes
    ----------
    questions : list[str]
        Validated questions from the benchmark dataset.

    answers : list[str]
        Validated answers from the benchmark dataset.

    document_ids : list[str]
        Validated IDs of documents with correct context for given answers.

    Raises
    ------
    BenchmarkValueError
        Raised when any of the arguments in the dataset is considered invalid.
    """

    QUESTION = "question"
    ANSWERS = "correct_answers"
    DOC_IDS = "correct_answer_document_ids"

    def __init__(self, benchmark_data: pd.DataFrame):
        self._benchmark_data = benchmark_data

        self.questions: list[str] = list(self._benchmark_data[self.QUESTION])
        self.answers: list[list[str]] = list(self._benchmark_data[self.ANSWERS])
        self.document_ids: list[list[str]] = list(self._benchmark_data[self.DOC_IDS])
        self._questions_ids = [f"q{idx}" for idx in range(len(self.questions))]

    def __iter__(self) -> Iterator[tuple[str, list[str], list[str] | None]]:
        for q, a, id_ in zip(self.questions, self.answers, self.document_ids):
            yield q, a, id_

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> tuple[str, list[str], list[str] | None]:
        return self.questions[idx], self.answers[idx], self.document_ids[idx]

    def get_random_sample(self, n_records: int = 10, random_seed: int = 17) -> "BenchmarkData":
        """
        Create sample of the original BenchmarkData. If number of desired records
        is bigger than actual size of the data, create new instance based on
        all samples.

        Parameters
        ----------
        n_records : int, default=10
            Number of records to be included in the newly created instance.

        random_seed : int, default=17
            Seed to make data sampling deterministic.

        Returns
        -------
        BenchmarkData
            New instance of BenchmarkData.
        """
        if n_records > len(self):
            sample = self._benchmark_data.copy()
        else:
            sample = self._benchmark_data.sample(n=n_records, random_state=random_seed)
        return self.__class__(benchmark_data=sample)

    @property
    def questions(self) -> list[str]:
        """Get all questions from benchmark data."""
        return self._questions

    @questions.setter
    def questions(self, val: list[str]) -> None:
        """Validate whether questions is a list of not empty strings"""
        _validate_list_of_strings(val, self.QUESTION)

        self._questions = val

    @property
    def answers(self) -> list[list[str]]:
        """get all answers from benchmark data."""
        return self._answers

    @answers.setter
    def answers(self, val: list[list[str]]) -> None:
        """Validate whether each element is a list of not empty strings"""
        for el in val:
            _validate_list_of_strings(el, self.ANSWERS)
        self._answers = val

    @property
    def document_ids(self) -> list[list[str]]:
        """Get all document ids from benchmark data."""
        return self._document_ids

    @document_ids.setter
    def document_ids(self, val: list[list[str]] | None) -> None:
        """Validate whether each element is a list of not empty strings"""
        if val is None:
            self._document_ids = val
        else:
            for el in val:
                _validate_list_of_strings(el, self.DOC_IDS)
            self._document_ids = val

    @property
    def questions_ids(self) -> list[str]:
        """Get all questions ids from benchmark data."""
        return self._questions_ids


def _validate_list_of_strings(elements: list[str], key: str) -> None:
    """
    Validate whether list of values is actually list of not-empty strings.

    Parameters
    ----------
    elements : list[str]
        List to be validated

    key : str
        What attribute we are validating. It is used to create proper message

    Raises
    ------
    BenchmarkDataValueError
        When some element is invalid
    """

    for element in elements:
        if not isinstance(element, str) or not element:
            raise BenchmarkDataValueError(f"Incorrect '{key}' value: '{element}'.")
