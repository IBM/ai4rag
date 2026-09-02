# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import random
from unittest.mock import MagicMock

import pytest

from ai4rag.utils.data.test_data_loader import (
    TestDataLoaderError,
    TestDataResult,
    load_test_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_record(question: str = "q", answer: str = "a", doc_id: str = "d1") -> dict:
    """Build a benchmark record with the required keys."""
    return {
        "question": question,
        "correct_answers": [answer],
        "correct_answer_document_ids": [doc_id],
    }


def _make_s3_body(payload: str) -> dict:
    """Simulate the ``get_object`` response shape used by ``_download_object``."""
    body = MagicMock()
    body.read.return_value = payload.encode("utf-8")
    return {"Body": body}


def _make_mock_s3_client(mocker, response: dict):
    """Return a mock S3 client with a preset ``get_object`` response."""
    mock = mocker.MagicMock()
    mock.get_object.return_value = response
    return mock


# ---------------------------------------------------------------------------
# TestDataResult
# ---------------------------------------------------------------------------


class TestTestDataResult:
    """Tests for the ``TestDataResult`` frozen dataclass."""

    def test_attributes(self):
        """Verify field values are stored correctly."""
        result = TestDataResult(data=[{"a": 1}], record_count=1, sampled=False)
        assert result.data == [{"a": 1}]
        assert result.record_count == 1
        assert result.sampled is False

    def test_frozen(self):
        """Frozen dataclass must reject attribute mutation."""
        result = TestDataResult(data=[], record_count=0, sampled=False)
        with pytest.raises(AttributeError):
            result.data = [{"new": True}]


# ---------------------------------------------------------------------------
# load_test_data -- happy path
# ---------------------------------------------------------------------------


class TestLoadTestDataHappyPath:
    """Happy-path tests for ``load_test_data``."""

    def test_returns_all_records(self, mocker):
        """All valid records are returned when count <= sample size."""
        records = [_valid_record(question=f"q{i}") for i in range(5)]
        payload = json.dumps(records)
        mock_client = _make_mock_s3_client(mocker, _make_s3_body(payload))

        result = load_test_data(
            bucket_name="bucket",
            key="data.json",
            benchmark_sample_size=25,
            s3_client=mock_client,
        )

        assert result.record_count == 5
        assert result.sampled is False
        assert result.data == records

    def test_get_object_called_correctly(self, mocker):
        """``get_object`` must receive bucket and key arguments."""
        records = [_valid_record()]
        mock_client = _make_mock_s3_client(mocker, _make_s3_body(json.dumps(records)))

        load_test_data(
            bucket_name="my-bucket",
            key="tests/data.json",
            s3_client=mock_client,
        )

        mock_client.get_object.assert_called_once_with(Bucket="my-bucket", Key="tests/data.json")


# ---------------------------------------------------------------------------
# load_test_data -- sampling
# ---------------------------------------------------------------------------


class TestLoadTestDataSampling:
    """Sampling behaviour of ``load_test_data``."""

    def test_large_dataset_is_sampled(self, mocker):
        """Datasets exceeding ``benchmark_sample_size`` must be sampled."""
        records = [_valid_record(question=f"q{i}") for i in range(50)]
        mock_client = _make_mock_s3_client(mocker, _make_s3_body(json.dumps(records)))

        result = load_test_data(
            bucket_name="bucket",
            key="data.json",
            benchmark_sample_size=25,
            s3_client=mock_client,
        )

        assert result.record_count == 25
        assert result.sampled is True

    def test_sampling_is_reproducible_with_seed_42(self, mocker):
        """Two calls with identical data must produce the same sample (seed 42)."""
        records = [_valid_record(question=f"q{i}") for i in range(50)]
        payload = json.dumps(records)

        result_a = load_test_data(
            bucket_name="b",
            key="k",
            benchmark_sample_size=10,
            s3_client=_make_mock_s3_client(mocker, _make_s3_body(payload)),
        )
        result_b = load_test_data(
            bucket_name="b",
            key="k",
            benchmark_sample_size=10,
            s3_client=_make_mock_s3_client(mocker, _make_s3_body(payload)),
        )

        assert result_a.data == result_b.data

    def test_sampling_matches_stdlib_random(self, mocker):
        """Sampled output should match ``random.Random(42).sample(...)``."""
        records = [_valid_record(question=f"q{i}") for i in range(40)]
        mock_client = _make_mock_s3_client(mocker, _make_s3_body(json.dumps(records)))

        result = load_test_data(
            bucket_name="b",
            key="k",
            benchmark_sample_size=10,
            s3_client=mock_client,
        )

        expected = random.Random(42).sample(records, 10)
        assert result.data == expected

    def test_sampling_disabled_when_zero(self, mocker):
        """``benchmark_sample_size=0`` disables sampling entirely."""
        records = [_valid_record(question=f"q{i}") for i in range(50)]
        mock_client = _make_mock_s3_client(mocker, _make_s3_body(json.dumps(records)))

        result = load_test_data(
            bucket_name="b",
            key="k",
            benchmark_sample_size=0,
            s3_client=mock_client,
        )

        assert result.record_count == 50
        assert result.sampled is False

    def test_no_sampling_when_count_equals_limit(self, mocker):
        """No sampling when the dataset size exactly matches the limit."""
        records = [_valid_record(question=f"q{i}") for i in range(25)]
        mock_client = _make_mock_s3_client(mocker, _make_s3_body(json.dumps(records)))

        result = load_test_data(
            bucket_name="b",
            key="k",
            benchmark_sample_size=25,
            s3_client=mock_client,
        )

        assert result.record_count == 25
        assert result.sampled is False


# ---------------------------------------------------------------------------
# load_test_data -- error paths
# ---------------------------------------------------------------------------


class TestLoadTestDataLoaderErrors:
    """Error-path tests for ``load_test_data``."""

    def test_invalid_json_raises_test_data_error(self, mocker):
        """Non-JSON content must raise ``TestDataLoaderError``."""
        mock_client = _make_mock_s3_client(mocker, _make_s3_body("not json {{{"))

        with pytest.raises(TestDataLoaderError, match="valid JSON"):
            load_test_data(bucket_name="b", key="k", s3_client=mock_client)

    def test_non_list_json_raises_test_data_error(self, mocker):
        """A JSON object (not a list) must raise ``TestDataLoaderError``."""
        mock_client = _make_mock_s3_client(mocker, _make_s3_body(json.dumps({"key": "value"})))

        with pytest.raises(TestDataLoaderError, match="list of benchmark records"):
            load_test_data(bucket_name="b", key="k", s3_client=mock_client)

    def test_missing_keys_raises_test_data_error(self, mocker):
        """Records missing required keys must raise ``TestDataLoaderError``."""
        bad_records = [{"question": "q1"}]  # missing two required keys
        mock_client = _make_mock_s3_client(mocker, _make_s3_body(json.dumps(bad_records)))

        with pytest.raises(TestDataLoaderError, match="Incorrect or incomplete keys"):
            load_test_data(bucket_name="b", key="k", s3_client=mock_client)

    def test_extra_keys_raises_test_data_error(self, mocker):
        """Records with extra keys beyond the required set must raise ``TestDataLoaderError``."""
        record = _valid_record()
        record["extra_field"] = "unexpected"
        mock_client = _make_mock_s3_client(mocker, _make_s3_body(json.dumps([record])))

        with pytest.raises(TestDataLoaderError, match="Incorrect or incomplete keys"):
            load_test_data(bucket_name="b", key="k", s3_client=mock_client)

    def test_non_dict_record_raises_test_data_error(self, mocker):
        """A list element that is not a dict must raise ``TestDataLoaderError``."""
        bad_records = ["not a dict"]
        mock_client = _make_mock_s3_client(mocker, _make_s3_body(json.dumps(bad_records)))

        with pytest.raises(TestDataLoaderError, match="Expected a dict"):
            load_test_data(bucket_name="b", key="k", s3_client=mock_client)

    def test_s3_404_raises_file_not_found(self, mocker):
        """A 404 from S3 must surface as ``FileNotFoundError``."""
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
        mock_client = mocker.MagicMock()
        mock_client.get_object.side_effect = ClientError(error_response, "GetObject")

        with pytest.raises(FileNotFoundError, match="not found in S3"):
            load_test_data(bucket_name="b", key="missing.json", s3_client=mock_client)

    def test_s3_no_such_key_raises_file_not_found(self, mocker):
        """``NoSuchKey`` error code must also surface as ``FileNotFoundError``."""
        from botocore.exceptions import ClientError

        error_response = {"Error": {"Code": "NoSuchKey", "Message": "No Such Key"}}
        mock_client = mocker.MagicMock()
        mock_client.get_object.side_effect = ClientError(error_response, "GetObject")

        with pytest.raises(FileNotFoundError, match="not found in S3"):
            load_test_data(bucket_name="b", key="gone.json", s3_client=mock_client)

    def test_empty_bucket_name_raises_type_error(self):
        """An empty ``bucket_name`` must raise ``TypeError``."""
        with pytest.raises(TypeError, match="non-empty string"):
            load_test_data(bucket_name="", key="k")

    def test_none_bucket_name_raises_type_error(self):
        """``None`` as ``bucket_name`` must raise ``TypeError``."""
        with pytest.raises(TypeError, match="non-empty string"):
            load_test_data(bucket_name=None, key="k")
