# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
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
        Benchmark Data given as DataFrame.It should be tabular data that contains
        questions, answers and correct_answer_document_keys columns.

    Attributes
    ----------
    questions : list[str]
        Validated questions from the benchmark dataset.

    correct_answers : list[str]
        Validated answers from the benchmark dataset.

    document_keys : list[list[str]]
        Validated S3 object keys of documents with correct context for given answers.

    Raises
    ------
    BenchmarkValueError
        Raised when any of the arguments in the dataset is considered invalid.
    """

    QUESTION = "question"
    CORRECT_ANSWERS = "correct_answers"
    DOC_KEYS = "correct_answer_document_keys"

    def __init__(self, benchmark_data: pd.DataFrame):
        if len(benchmark_data) == 0:
            raise BenchmarkDataValueError("There are no records in the benchmark data.")
        self._benchmark_data = benchmark_data

        self.questions: list[str] = list(self._benchmark_data[self.QUESTION])
        self.correct_answers: list[list[str]] = list(self._benchmark_data[self.CORRECT_ANSWERS])

        if self.DOC_KEYS not in self._benchmark_data.columns:
            raise BenchmarkDataValueError(
                f"Benchmark data must contain '{self.DOC_KEYS}'. "
                "Each record needs a list of S3 object keys identifying the ground-truth documents."
            )
        self.document_keys: list[list[str]] = list(self._benchmark_data[self.DOC_KEYS])

        self._questions_ids = [f"q{idx}" for idx in range(len(self.questions))]

    def __iter__(self) -> Iterator[tuple[str, list[str], list[str] | None]]:
        for q, a, keys in zip(self.questions, self.correct_answers, self.document_keys):
            yield q, a, keys

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> tuple[str, list[str], list[str] | None]:
        return self.questions[idx], self.correct_answers[idx], self.document_keys[idx]

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
        if n_records <= 0:
            raise BenchmarkDataValueError("Cannot make empty sample. Select 'n_records' to be an int greater than 0.")
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
    def correct_answers(self) -> list[list[str]]:
        """get all answers from benchmark data."""
        return self._correct_answers

    @correct_answers.setter
    def correct_answers(self, val: list[list[str]]) -> None:
        """Validate whether each element is a non-empty list of not empty strings"""
        for el in val:
            if not el:
                raise BenchmarkDataValueError(
                    f"Incorrect '{self.CORRECT_ANSWERS}' value: each question must have at least one "
                    "correct answer, got an empty list."
                )
            _validate_list_of_strings(el, self.CORRECT_ANSWERS)
        self._correct_answers = val

    @property
    def document_keys(self) -> list[list[str]]:
        """Get all document keys from benchmark data."""
        return self._document_keys

    @document_keys.setter
    def document_keys(self, val: list[list[str]]) -> None:
        """Validate whether each element is a non-empty list of not empty strings"""
        for el in val:
            if not el:
                raise BenchmarkDataValueError(
                    f"Incorrect '{self.DOC_KEYS}' value: each question must have at least one "
                    "document key, got an empty list."
                )
            _validate_list_of_strings(el, self.DOC_KEYS)
        self._document_keys = val

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
