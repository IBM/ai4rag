# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import json
import time
from pathlib import Path

import pytest

from ai4rag.utils.experiment_monitor import ExperimentMonitor


class TestExperimentMonitorInitialization:
    """Test suite for ExperimentMonitor initialization."""

    def test_init_with_none_output_path(self):
        """Test initialization with None output_path."""
        monitor = ExperimentMonitor(output_path=None)
        assert monitor.output_path is None
        assert monitor.total_time == 0
        assert monitor.total_event_times == {}
        assert monitor.rag_patterns == {}
        assert isinstance(monitor.last_event, float)
        assert isinstance(monitor.last_pattern, float)

    def test_init_with_string_json_file_path(self, tmp_path):
        """Test initialization with string path to json file."""
        output_file = tmp_path / "test_monitor.json"
        monitor = ExperimentMonitor(output_path=str(output_file))
        assert monitor.output_path == output_file
        assert monitor.total_time == 0
        assert monitor.total_event_times == {}

    def test_init_with_path_json_file(self, tmp_path):
        """Test initialization with Path object pointing to json file."""
        output_file = tmp_path / "test_monitor.json"
        monitor = ExperimentMonitor(output_path=output_file)
        assert monitor.output_path == output_file

    def test_init_with_directory_path(self, tmp_path):
        """Test initialization with directory path uses default filename."""
        monitor = ExperimentMonitor(output_path=tmp_path)
        expected_path = tmp_path / ExperimentMonitor._default_output_file
        assert monitor.output_path == expected_path

    def test_init_with_string_directory_path(self, tmp_path):
        """Test initialization with string directory path."""
        monitor = ExperimentMonitor(output_path=str(tmp_path))
        expected_path = tmp_path / ExperimentMonitor._default_output_file
        assert monitor.output_path == expected_path

    def test_init_with_non_json_file_extension_warns_and_uses_default(self, tmp_path, mocker):
        """Test initialization with non-json file extension logs warning and uses default."""
        mock_logger = mocker.patch("ai4rag.utils.experiment_monitor.logger")
        output_file = tmp_path / "test_monitor.txt"
        monitor = ExperimentMonitor(output_path=output_file)

        # Should use default json filename in parent directory
        expected_path = tmp_path / ExperimentMonitor._default_output_file
        assert monitor.output_path == expected_path
        mock_logger.warning.assert_called_once()
        assert "should be json" in mock_logger.warning.call_args[0][0]

    def test_init_with_nested_directory_path(self, tmp_path):
        """Test initialization with nested directory path."""
        nested_dir = tmp_path / "nested" / "dir"
        monitor = ExperimentMonitor(output_path=nested_dir)
        expected_path = nested_dir / ExperimentMonitor._default_output_file
        assert monitor.output_path == expected_path

    def test_init_sets_timestamps(self):
        """Test that initialization sets last_event and last_pattern timestamps."""
        before_time = time.time()
        monitor = ExperimentMonitor(output_path=None)
        after_time = time.time()

        assert before_time <= monitor.last_event <= after_time
        assert before_time <= monitor.last_pattern <= after_time

    def test_default_output_file_constant(self):
        """Test that default output file constant is correct."""
        assert ExperimentMonitor._default_output_file == "experiment_monitor.json"


class TestExperimentMonitorPatternTracking:
    """Test suite for pattern tracking methods."""

    @pytest.fixture
    def monitor(self):
        """Create a monitor without output path for testing."""
        return ExperimentMonitor(output_path=None)

    def test_on_pattern_start_updates_timestamp(self, monitor):
        """Test that on_pattern_start updates last_pattern timestamp."""
        original_time = monitor.last_pattern
        time.sleep(0.01)  # Small delay to ensure time difference
        monitor.on_pattern_start()
        assert monitor.last_pattern > original_time

    def test_on_pattern_finish_records_pattern_time(self, monitor, mocker):
        """Test that on_pattern_finish records pattern execution time."""
        mock_logger = mocker.patch("ai4rag.utils.experiment_monitor.logger")
        monitor.on_pattern_start()
        time.sleep(0.01)  # Small delay
        monitor.on_pattern_finish("test_pattern")

        assert "test_pattern" in monitor.rag_patterns
        assert "total_time" in monitor.rag_patterns["test_pattern"]
        assert isinstance(monitor.rag_patterns["test_pattern"]["total_time"], str)
        mock_logger.debug.assert_called_once()

    def test_on_pattern_finish_logs_execution_time(self, monitor, mocker):
        """Test that on_pattern_finish logs the execution time."""
        mock_logger = mocker.patch("ai4rag.utils.experiment_monitor.logger")
        monitor.on_pattern_start()
        monitor.on_pattern_finish("pattern_1")

        mock_logger.debug.assert_called_once()
        log_message = mock_logger.debug.call_args[0][0]
        assert "Total execution time" in log_message
        assert "pattern_1" in mock_logger.debug.call_args[0][1]

    def test_on_pattern_finish_multiple_patterns(self, monitor):
        """Test tracking multiple patterns."""
        monitor.on_pattern_start()
        time.sleep(0.01)
        monitor.on_pattern_finish("pattern_1")

        monitor.on_pattern_start()
        time.sleep(0.01)
        monitor.on_pattern_finish("pattern_2")

        assert "pattern_1" in monitor.rag_patterns
        assert "pattern_2" in monitor.rag_patterns
        assert len(monitor.rag_patterns) == 2

    def test_on_pattern_finish_updates_last_pattern_timestamp(self, monitor):
        """Test that on_pattern_finish updates last_pattern timestamp."""
        monitor.on_pattern_start()
        original_time = monitor.last_pattern
        time.sleep(0.01)
        monitor.on_pattern_finish("test_pattern")
        assert monitor.last_pattern > original_time


class TestExperimentMonitorEventTracking:
    """Test suite for event tracking methods."""

    @pytest.fixture
    def monitor(self):
        """Create a monitor without output path for testing."""
        return ExperimentMonitor(output_path=None)

    def test_on_start_event_info_updates_timestamp(self, monitor):
        """Test that on_start_event_info updates last_event timestamp."""
        original_time = monitor.last_event
        time.sleep(0.01)
        monitor.on_start_event_info()
        assert monitor.last_event > original_time

    def test_on_finish_event_info_logs_event(self, monitor, mocker):
        """Test that on_finish_event_info logs event information."""
        mock_logger = mocker.patch("ai4rag.utils.experiment_monitor.logger")
        monitor.on_start_event_info()
        time.sleep(0.01)
        monitor.on_finish_event_info(event="indexing", step="optimization")

        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "indexing" in log_message
        assert "optimization" in log_message
        assert "took:" in log_message

    def test_on_finish_event_info_with_kwargs(self, monitor, mocker):
        """Test on_finish_event_info with additional kwargs."""
        mock_logger = mocker.patch("ai4rag.utils.experiment_monitor.logger")
        monitor.on_start_event_info()
        monitor.on_finish_event_info(
            event="chunking",
            step="preprocessing",
            chunk_size=512,
            model="test-model",
        )

        log_message = mock_logger.info.call_args[0][0]
        assert "chunking" in log_message
        assert "preprocessing" in log_message
        assert "chunk_size" in log_message
        assert "512" in log_message

    def test_on_finish_event_info_updates_total_event_times(self, monitor):
        """Test that on_finish_event_info updates total_event_times."""
        monitor.on_start_event_info()
        time.sleep(0.01)
        monitor.on_finish_event_info(event="embedding", step="indexing")

        assert "indexing" in monitor.total_event_times
        assert "embedding" in monitor.total_event_times["indexing"]
        assert monitor.total_event_times["indexing"]["embedding"] > 0

    def test_on_finish_event_info_accumulates_event_times(self, monitor):
        """Test that multiple calls to same event accumulate times."""
        monitor.on_start_event_info()
        time.sleep(0.01)
        monitor.on_finish_event_info(event="retrieval", step="generation")

        monitor.on_start_event_info()
        time.sleep(0.01)
        monitor.on_finish_event_info(event="retrieval", step="generation")

        # Time should be accumulated
        assert monitor.total_event_times["generation"]["retrieval"] > 0.01

    def test_on_finish_event_info_multiple_events_same_step(self, monitor):
        """Test tracking multiple events in the same step."""
        monitor.on_start_event_info()
        monitor.on_finish_event_info(event="event1", step="step1")

        monitor.on_start_event_info()
        monitor.on_finish_event_info(event="event2", step="step1")

        assert "step1" in monitor.total_event_times
        assert "event1" in monitor.total_event_times["step1"]
        assert "event2" in monitor.total_event_times["step1"]

    def test_on_finish_event_info_multiple_steps(self, monitor):
        """Test tracking events across multiple steps."""
        monitor.on_start_event_info()
        monitor.on_finish_event_info(event="event1", step="step1")

        monitor.on_start_event_info()
        monitor.on_finish_event_info(event="event2", step="step2")

        assert "step1" in monitor.total_event_times
        assert "step2" in monitor.total_event_times
        assert len(monitor.total_event_times) == 2


class TestExperimentMonitorFormatTime:
    """Test suite for _format_time static method."""

    def test_format_time_zero_seconds(self):
        """Test formatting zero seconds."""
        result = ExperimentMonitor._format_time(0)
        assert result == "0.00 s"

    def test_format_time_less_than_one_second(self):
        """Test formatting time less than one second."""
        result = ExperimentMonitor._format_time(0.5)
        assert result == "0.50 s"

    def test_format_time_exact_seconds(self):
        """Test formatting exact seconds."""
        result = ExperimentMonitor._format_time(5.0)
        assert result == "5.00 s"

    def test_format_time_seconds_with_decimals(self):
        """Test formatting seconds with decimal places."""
        result = ExperimentMonitor._format_time(45.678)
        assert result == "45.68 s"

    def test_format_time_one_minute(self):
        """Test formatting exactly one minute."""
        result = ExperimentMonitor._format_time(60)
        assert result == "1 min 0.00 s"

    def test_format_time_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        result = ExperimentMonitor._format_time(125.5)
        assert result == "2 min 5.50 s"

    def test_format_time_multiple_minutes(self):
        """Test formatting multiple minutes."""
        result = ExperimentMonitor._format_time(245.25)
        assert result == "4 min 5.25 s"

    def test_format_time_large_value(self):
        """Test formatting large time values."""
        result = ExperimentMonitor._format_time(3661.75)  # 1 hour, 1 minute, 1.75 seconds
        assert result == "61 min 1.75 s"

    def test_format_time_rounds_to_two_decimals(self):
        """Test that time is rounded to two decimal places."""
        result = ExperimentMonitor._format_time(1.23456)
        assert result == "1.23 s"

    @pytest.mark.parametrize(
        "time_value,expected",
        [
            (0, "0.00 s"),
            (1, "1.00 s"),
            (59.99, "59.99 s"),
            (60, "1 min 0.00 s"),
            (90, "1 min 30.00 s"),
            (120, "2 min 0.00 s"),
            (3600, "60 min 0.00 s"),
        ],
    )
    def test_format_time_parameterized(self, time_value, expected):
        """Parameterized test for various time values."""
        result = ExperimentMonitor._format_time(time_value)
        assert result == expected


class TestExperimentMonitorClose:
    """Test suite for close method and related functionality."""

    @pytest.fixture
    def monitor_with_output(self, tmp_path):
        """Create a monitor with output path."""
        output_file = tmp_path / "test_monitor.json"
        return ExperimentMonitor(output_path=output_file)

    @pytest.fixture
    def monitor_without_output(self):
        """Create a monitor without output path."""
        return ExperimentMonitor(output_path=None)

    def test_close_calls_summarize(self, monitor_without_output, mocker):
        """Test that close calls _summarize method."""
        mock_summarize = mocker.patch.object(monitor_without_output, "_summarize")
        monitor_without_output.close()
        mock_summarize.assert_called_once()

    def test_close_calls_to_json_when_output_path_set(self, monitor_with_output, mocker):
        """Test that close calls _to_json when output_path is set."""
        mock_to_json = mocker.patch.object(monitor_with_output, "_to_json")
        monitor_with_output.close()
        mock_to_json.assert_called_once()

    def test_close_does_not_call_to_json_when_output_path_none(self, monitor_without_output, mocker):
        """Test that close does not call _to_json when output_path is None."""
        mock_to_json = mocker.patch.object(monitor_without_output, "_to_json")
        monitor_without_output.close()
        mock_to_json.assert_not_called()

    def test_summarize_logs_event_times(self, monitor_without_output, mocker):
        """Test that _summarize logs event times."""
        mock_logger = mocker.patch("ai4rag.utils.experiment_monitor.logger")

        # Add some test data
        monitor_without_output.on_start_event_info()
        time.sleep(0.01)
        monitor_without_output.on_finish_event_info(event="test_event", step="test_step")

        monitor_without_output._summarize()

        # Should log event time
        assert mock_logger.info.call_count >= 1
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("test_event" in call and "test_step" in call for call in log_calls)

    def test_summarize_logs_pattern_times(self, monitor_without_output, mocker):
        """Test that _summarize logs pattern times."""
        mock_logger = mocker.patch("ai4rag.utils.experiment_monitor.logger")

        # Add pattern data
        monitor_without_output.on_pattern_start()
        time.sleep(0.01)
        monitor_without_output.on_pattern_finish("test_pattern")

        # Clear previous logger calls
        mock_logger.reset_mock()

        monitor_without_output._summarize()

        # Should log pattern time
        assert mock_logger.info.called
        # Check all arguments in logger calls, not just the format string
        log_calls = [call[0] for call in mock_logger.info.call_args_list]
        assert any("test_pattern" in str(call) for call in log_calls)

    def test_to_json_creates_file(self, monitor_with_output):
        """Test that _to_json creates JSON file."""
        monitor_with_output.on_pattern_start()
        time.sleep(0.01)
        monitor_with_output.on_pattern_finish("test_pattern")

        monitor_with_output._to_json()

        assert monitor_with_output.output_path.exists()
        assert monitor_with_output.output_path.is_file()

    def test_to_json_creates_parent_directories(self, tmp_path):
        """Test that _to_json creates parent directories if they don't exist."""
        nested_path = tmp_path / "nested" / "dir" / "monitor.json"
        monitor = ExperimentMonitor(output_path=nested_path)

        monitor.on_pattern_start()
        monitor.on_pattern_finish("test_pattern")
        monitor._to_json()

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_to_json_writes_correct_structure(self, monitor_with_output):
        """Test that _to_json writes correct JSON structure."""
        monitor_with_output.on_pattern_start()
        time.sleep(0.01)
        monitor_with_output.on_pattern_finish("pattern_1")

        monitor_with_output._to_json()

        with open(monitor_with_output.output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "data" in data
        assert "pattern_1" in data["data"]
        assert "total_time" in data["data"]["pattern_1"]

    def test_to_json_writes_multiple_patterns(self, monitor_with_output):
        """Test _to_json with multiple patterns."""
        monitor_with_output.on_pattern_start()
        monitor_with_output.on_pattern_finish("pattern_1")

        monitor_with_output.on_pattern_start()
        monitor_with_output.on_pattern_finish("pattern_2")

        monitor_with_output._to_json()

        with open(monitor_with_output.output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["data"]) == 2
        assert "pattern_1" in data["data"]
        assert "pattern_2" in data["data"]

    def test_to_json_logs_output_path(self, monitor_with_output, mocker):
        """Test that _to_json logs the output path."""
        mock_logger = mocker.patch("ai4rag.utils.experiment_monitor.logger")
        monitor_with_output._to_json()

        mock_logger.debug.assert_called_once()
        assert "Writing monitoring results" in mock_logger.debug.call_args[0][0]


class TestExperimentMonitorIntegration:
    """Integration tests for ExperimentMonitor complete workflow."""

    def test_complete_workflow_with_json_output(self, tmp_path):
        """Test complete workflow from initialization to close with JSON output."""
        output_file = tmp_path / "complete_test.json"
        monitor = ExperimentMonitor(output_path=output_file)

        # Track events
        monitor.on_start_event_info()
        time.sleep(0.01)
        monitor.on_finish_event_info(event="embedding", step="indexing")

        monitor.on_start_event_info()
        time.sleep(0.01)
        monitor.on_finish_event_info(event="vectorstore", step="indexing")

        # Track patterns
        monitor.on_pattern_start()
        time.sleep(0.01)
        monitor.on_pattern_finish("pattern_1")

        monitor.on_pattern_start()
        time.sleep(0.01)
        monitor.on_pattern_finish("pattern_2")

        # Close and verify
        monitor.close()

        assert output_file.exists()
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "data" in data
        assert "pattern_1" in data["data"]
        assert "pattern_2" in data["data"]

    def test_complete_workflow_without_json_output(self, mocker):
        """Test complete workflow without JSON output."""
        mock_logger = mocker.patch("ai4rag.utils.experiment_monitor.logger")
        monitor = ExperimentMonitor(output_path=None)

        monitor.on_start_event_info()
        time.sleep(0.01)
        monitor.on_finish_event_info(event="test", step="testing")

        monitor.on_pattern_start()
        time.sleep(0.01)
        monitor.on_pattern_finish("test_pattern")

        monitor.close()

        # Should have logged but not saved to file
        assert mock_logger.info.called

    def test_multiple_events_and_patterns(self, tmp_path):
        """Test tracking multiple events and patterns."""
        output_file = tmp_path / "multi_test.json"
        monitor = ExperimentMonitor(output_path=output_file)

        # Multiple events in same step
        for i in range(3):
            monitor.on_start_event_info()
            time.sleep(0.01)
            monitor.on_finish_event_info(event=f"event_{i}", step="step_1")

        # Multiple patterns
        for i in range(3):
            monitor.on_pattern_start()
            time.sleep(0.01)
            monitor.on_pattern_finish(f"pattern_{i}")

        monitor.close()

        assert output_file.exists()
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["data"]) == 3

    def test_events_across_multiple_steps(self, tmp_path):
        """Test tracking events across different steps."""
        output_file = tmp_path / "multi_step_test.json"
        monitor = ExperimentMonitor(output_path=output_file)

        # Events in different steps
        for step in ["preprocessing", "optimization", "evaluation"]:
            monitor.on_start_event_info()
            time.sleep(0.01)
            monitor.on_finish_event_info(event="test_event", step=step)

        monitor.close()

        # Verify all steps were tracked
        assert "preprocessing" in monitor.total_event_times
        assert "optimization" in monitor.total_event_times
        assert "evaluation" in monitor.total_event_times


class TestExperimentMonitorEdgeCases:
    """Test suite for edge cases and error scenarios."""

    def test_pattern_finish_without_pattern_start(self):
        """Test calling on_pattern_finish without on_pattern_start."""
        monitor = ExperimentMonitor(output_path=None)
        # Should not raise error, just record from initialization time
        monitor.on_pattern_finish("test_pattern")
        assert "test_pattern" in monitor.rag_patterns

    def test_event_finish_without_event_start(self):
        """Test calling on_finish_event_info without on_start_event_info."""
        monitor = ExperimentMonitor(output_path=None)
        # Should not raise error, just record from initialization time
        monitor.on_finish_event_info(event="test", step="testing")
        assert "testing" in monitor.total_event_times

    def test_close_with_empty_data(self, tmp_path):
        """Test closing monitor with no tracked data."""
        output_file = tmp_path / "empty_test.json"
        monitor = ExperimentMonitor(output_path=output_file)
        monitor.close()

        assert output_file.exists()
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)

        assert data["data"] == {}

    def test_close_multiple_times(self, tmp_path):
        """Test calling close multiple times."""
        output_file = tmp_path / "multi_close.json"
        monitor = ExperimentMonitor(output_path=output_file)

        monitor.on_pattern_start()
        monitor.on_pattern_finish("pattern_1")

        # Close multiple times should not raise error
        monitor.close()
        monitor.close()

        assert output_file.exists()

    def test_very_long_pattern_name(self):
        """Test with very long pattern name."""
        monitor = ExperimentMonitor(output_path=None)
        long_name = "pattern_" + "x" * 1000

        monitor.on_pattern_start()
        monitor.on_pattern_finish(long_name)

        assert long_name in monitor.rag_patterns

    def test_special_characters_in_names(self):
        """Test with special characters in event and pattern names."""
        monitor = ExperimentMonitor(output_path=None)

        special_chars = "test_@#$%^&*()_pattern"
        monitor.on_pattern_start()
        monitor.on_pattern_finish(special_chars)

        assert special_chars in monitor.rag_patterns

    def test_unicode_in_names(self, tmp_path):
        """Test with Unicode characters in names."""
        output_file = tmp_path / "unicode_test.json"
        monitor = ExperimentMonitor(output_path=output_file)

        unicode_name = "pattern_测试_テスト_🚀"
        monitor.on_pattern_start()
        monitor.on_pattern_finish(unicode_name)

        monitor.close()

        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)

        assert unicode_name in data["data"]

    def test_empty_string_names(self):
        """Test with empty string as pattern/event name."""
        monitor = ExperimentMonitor(output_path=None)

        monitor.on_pattern_start()
        monitor.on_pattern_finish("")

        assert "" in monitor.rag_patterns

    def test_event_with_empty_kwargs(self, mocker):
        """Test on_finish_event_info with empty kwargs."""
        mock_logger = mocker.patch("ai4rag.utils.experiment_monitor.logger")
        monitor = ExperimentMonitor(output_path=None)

        monitor.on_start_event_info()
        monitor.on_finish_event_info(event="test", step="testing")

        # Should log without kwargs section
        log_message = mock_logger.info.call_args[0][0]
        assert "test" in log_message
        assert "testing" in log_message

    def test_json_output_with_existing_file_overwrites(self, tmp_path):
        """Test that saving JSON overwrites existing file."""
        output_file = tmp_path / "overwrite_test.json"

        # Create first monitor and save
        monitor1 = ExperimentMonitor(output_path=output_file)
        monitor1.on_pattern_start()
        monitor1.on_pattern_finish("pattern_1")
        monitor1.close()

        # Create second monitor and save
        monitor2 = ExperimentMonitor(output_path=output_file)
        monitor2.on_pattern_start()
        monitor2.on_pattern_finish("pattern_2")
        monitor2.close()

        # Second save should overwrite
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "pattern_2" in data["data"]
        assert "pattern_1" not in data["data"]
