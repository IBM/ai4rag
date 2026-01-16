# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np

from ai4rag.core.hpo.utils import handle_missing_values_in_combinations_being_explored


class TestHandleMissingValues:
    """Test the handle_missing_values_in_combinations_being_explored function."""

    def test_handle_missing_ranker_strategy_and_sparse_vectors(self):
        """Test handling of missing values in ranker_strategy and ranker_sparse_vectors columns."""
        df = pd.DataFrame({
            "ranker_strategy": ["strategy1", np.nan, "strategy2"],
            "ranker_sparse_vectors": [["vec1", "vec2"], np.nan, ["vec3"]],
            "other_column": [1, 2, 3],
        })

        result = handle_missing_values_in_combinations_being_explored(df)

        assert result["ranker_strategy"].tolist() == ["strategy1", "", "strategy2"]
        assert result["ranker_sparse_vectors"].tolist() == ["['vec1', 'vec2']", "", "['vec3']"]
        assert result["other_column"].tolist() == [1, 2, 3]

    def test_handle_ranker_sparse_vectors_with_non_list(self):
        """Test handling of ranker_sparse_vectors when it contains non-list values."""
        df = pd.DataFrame({
            "ranker_strategy": ["strategy1", "strategy2", "strategy3"],
            "ranker_sparse_vectors": ["not_a_list", np.nan, "another_string"],
        })

        result = handle_missing_values_in_combinations_being_explored(df)

        assert result["ranker_sparse_vectors"].tolist() == ["not_a_list", "", "another_string"]
        assert result["ranker_strategy"].tolist() == ["strategy1", "strategy2", "strategy3"]

    def test_handle_missing_ranker_k(self):
        """Test handling of missing values in ranker_k column."""
        df = pd.DataFrame({
            "ranker_k": [10, np.nan, 20, 30],
        })

        result = handle_missing_values_in_combinations_being_explored(df)

        assert result["ranker_k"].tolist() == [10, -1, 20, 30]

    def test_handle_missing_ranker_alpha(self):
        """Test handling of missing values in ranker_alpha column."""
        df = pd.DataFrame({
            "ranker_alpha": [0.5, np.nan, 0.8, np.nan],
        })

        result = handle_missing_values_in_combinations_being_explored(df)

        assert result["ranker_alpha"].tolist() == [0.5, -1, 0.8, -1]

    def test_handle_all_special_columns_together(self):
        """Test handling of all special columns together."""
        df = pd.DataFrame({
            "ranker_strategy": ["strategy1", np.nan, "strategy2"],
            "ranker_sparse_vectors": [["vec1"], np.nan, "not_a_list"],
            "ranker_k": [10, np.nan, 30],
            "ranker_alpha": [0.5, np.nan, 0.9],
            "regular_column": ["a", "b", "c"],
        })

        result = handle_missing_values_in_combinations_being_explored(df)

        assert result["ranker_strategy"].tolist() == ["strategy1", "", "strategy2"]
        assert result["ranker_sparse_vectors"].tolist() == ["['vec1']", "", "not_a_list"]
        assert result["ranker_k"].tolist() == [10, -1, 30]
        assert result["ranker_alpha"].tolist() == [0.5, -1, 0.9]
        assert result["regular_column"].tolist() == ["a", "b", "c"]

    def test_handle_dataframe_without_special_columns(self):
        """Test that dataframes without special columns are returned unchanged."""
        df = pd.DataFrame({
            "column1": [1, 2, 3],
            "column2": ["a", "b", "c"],
            "column3": [np.nan, 4.5, 6.7],
        })

        result = handle_missing_values_in_combinations_being_explored(df)

        # The result should be the same since no special columns are present
        pd.testing.assert_frame_equal(result, df)

    def test_handle_empty_dataframe(self):
        """Test handling of an empty dataframe."""
        df = pd.DataFrame()

        result = handle_missing_values_in_combinations_being_explored(df)

        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_handle_dataframe_with_only_nan_values(self):
        """Test handling of dataframe with only NaN values in special columns."""
        df = pd.DataFrame({
            "ranker_strategy": [np.nan, np.nan],
            "ranker_sparse_vectors": [np.nan, np.nan],
            "ranker_k": [np.nan, np.nan],
        })

        result = handle_missing_values_in_combinations_being_explored(df)

        assert result["ranker_strategy"].tolist() == ["", ""]
        assert result["ranker_sparse_vectors"].tolist() == ["", ""]
        assert result["ranker_k"].tolist() == [-1, -1]

    def test_handle_dataframe_with_no_nan_values(self):
        """Test handling of dataframe with no NaN values in special columns."""
        df = pd.DataFrame({
            "ranker_strategy": ["strategy1", "strategy2"],
            "ranker_k": [10, 20],
            "ranker_alpha": [0.5, 0.8],
            "ranker_sparse_vectors": [["vec1"], ["vec2"]],
        })

        result = handle_missing_values_in_combinations_being_explored(df)

        assert result["ranker_strategy"].tolist() == ["strategy1", "strategy2"]
        assert result["ranker_k"].tolist() == [10, 20]
        assert result["ranker_alpha"].tolist() == [0.5, 0.8]
        assert result["ranker_sparse_vectors"].tolist() == ["['vec1']", "['vec2']"]

    def test_returns_same_dataframe_object(self):
        """Test that the function modifies and returns the same dataframe object."""
        df = pd.DataFrame({
            "ranker_strategy": ["strategy1", np.nan],
            "ranker_sparse_vectors": [["vec1"], np.nan],
        })

        result = handle_missing_values_in_combinations_being_explored(df)

        # The function should return the modified dataframe
        assert result is df
