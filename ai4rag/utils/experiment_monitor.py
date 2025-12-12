#
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
#
import time
import json
import os
from pathlib import Path
from collections import defaultdict

from ai4rag import logger


class ExperimentMonitor:
    """
    Class responsible for monitoring execution times for each step of AutoRAG experiment

    Parameters
    ----------
    output_path : str | Path | None, default=None
        Path to which json file with results is saved. If None, results are not saved.
        Can point to directory or file, if not json file is provided file_directory / _default_output_file will be used.

    Attributes
    ----------
    output_path : str | Path | None
        Path for json with monitoring results

    rag_patterns : dict
        Dictionary containing results of experiment monitoring

    total_time : int
        Total time of the experiment.

    Methods
    -------
    on_start_event_info()
        Saves events start time

    on_start_pattern()
        Saves pattern execution start time

    on_finish_event_info()
        Logs information about tracked events and updates total_event_times dictionary

    on_pattern_finish()
        Logs monitoring results and updates rag_patterns dict on experiment pattern finish

    close()
        Closes experiment monitoring, invokes summarization() and _to_json() methods
    """

    _default_output_file = "experiment_monitor.json"

    def __init__(self, output_path: Path | str | None):
        self.output_path = output_path
        self.total_time = 0
        self.total_event_times = {}
        self.last_event = time.time()
        self.last_pattern = time.time()
        self.rag_patterns = {}

        if not output_path:
            return

        if isinstance(output_path, str):
            output_path = Path(output_path)

        suffix = output_path.suffix
        if suffix != "":
            if suffix != ".json":
                self.output_path = output_path.parent / ExperimentMonitor._default_output_file
                logger.warning(
                    "Output file for experiment monitor should be json, using %s as output file", self.output_path
                )
            else:
                self.output_path = output_path
        else:
            self.output_path = output_path / ExperimentMonitor._default_output_file

    def on_pattern_start(self):
        """
        Saves pattern execution start time
        """
        self.last_pattern = time.time()

    def on_pattern_finish(self, pattern_name: str):
        """
        Records and logs total pattern execution time

        Parameters
        ----------
        pattern_name : str
            name of finished pattern
        """

        finish_time = time.time()
        self.rag_patterns[pattern_name] = {"total_time": self._format_time(finish_time - self.last_pattern)}
        self.last_pattern = finish_time

        logger.debug(
            "Total execution time for %s: %s",
            pattern_name,
            self.rag_patterns[pattern_name]["total_time"],
        )

    def on_start_event_info(self):
        """
        Saves event start time
        """
        self.last_event = time.time()

    def on_finish_event_info(self, event: str, step: str, **kwargs):
        """
        Records and logs information about time of execution on INFO level

        Parameters
        ----------
        event: str
            name of the event.
        step: str
            current step of experiment - model preselection or actual optimization
        kwargs : dict[str, str]
            optional arguments to pass to get_message() function.
        """
        duration = time.time() - self.last_event

        process_info = f"{event} process during {step} step "
        description_info = ""
        if kwargs:
            description_info += str(kwargs)
        duration_info = f" took: {self._format_time(duration)}"

        message = process_info + description_info + duration_info
        logger.info(message)

        if step not in self.total_event_times:
            self.total_event_times[step] = defaultdict(int)
        self.total_event_times[step][event] += duration

    def close(self):
        """
        At the end of experiment performs rag_patterns summarization and saves to json if save_to_json is set to True
        """
        self._summarize()
        if self.output_path:
            self._to_json()

    def _to_json(self):
        """
        Saves experiment monitoring results to json file in location specified in self.output_path
        """
        to_save = {"data": self.rag_patterns}
        logger.debug("Writing monitoring results json in location: %s", self.output_path)

        os.makedirs(self.output_path.parent, exist_ok=True)
        with open(self.output_path, mode="w", encoding="utf-8") as file:
            json.dump(to_save, file, indent=4)

    def _summarize(self):
        """
        Sums all steps execution times and logs results
        """
        for key_step, dictionary in self.total_event_times.items():
            for key_event, value in dictionary.items():
                logger.info("%s total time during %s: %s", key_event, key_step, self._format_time(value))

        for key, value in self.rag_patterns.items():
            logger.info("%s execution total time %s", key, value["total_time"])

    @staticmethod
    def _format_time(time_: float) -> str:
        """
        Formats time period in seconds to minutes and seconds.

        Parameters
        ----------
        time_ : float
            Time period to be formatted.

        Returns
        -------
        str
            Time given in minutes and seconds as a string.
        """
        minutes = int(time_ / 60)
        seconds = time_ - minutes * 60
        return f"{f'{minutes} min ' if minutes else ''}{seconds:.2f} s"
