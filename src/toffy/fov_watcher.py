import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple, Union

from matplotlib import pyplot as plt
from watchdog.events import (
    DirCreatedEvent,
    FileCreatedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from toffy.json_utils import read_json_file


class RunStructure:
    """Expected bin and json files

    Attributes:
        fov_progress (dict): Whether or not an expected file has been created
    """

    def __init__(self, run_folder: str, timeout: int = 10 * 60):
        """initializes RunStructure by parsing run json within provided run folder

        Args:
            run_folder (str):
                path to run folder
            timeout (int):
                number of seconds to wait for non-null filesize before raising an error
        """
        self.timeout = timeout
        self.fov_progress = {}
        self.processed_fovs = []
        self.moly_points = []

        # find run .json and get parameters
        run_name = Path(run_folder).parts[-1]
        run_metadata = read_json_file(os.path.join(run_folder, f"{run_name}.json"))

        # parse run_metadata and populate expected structure
        for fov in run_metadata.get("fovs", ()):
            run_order = fov.get("runOrder", -1)
            scan = fov.get("scanCount", -1)
            if run_order * scan < 0:
                raise KeyError(f"Could not locate keys in {run_folder}.json")

            fov_names = [f"fov-{run_order}-scan-{s + 1}" for s in range(scan)]

            # identify moly points
            if fov.get("standardTarget", "") == "Molybdenum Foil":
                for fov_name in fov_names:
                    self.moly_points.append(fov_name)

            for fov_name in fov_names:
                self.fov_progress[fov_name] = {
                    "json": False,
                    "bin": False,
                }

    def check_run_condition(self, path: str) -> Tuple[bool, str]:
        """Checks if all requisite files exist and are complete

        Args:
            path (str):
                path to expected file

        Raises:
            TimeoutError

        Returns:
            (bool, str):
                whether or not both json and bin files exist, as well as the name of the point
        """

        filename = Path(path).parts[-1]

        # if filename starts with a '.' (temp file), it should be ignored
        if filename[0] == ".":
            return False, ""

        # filename is not corrct format of fov.bin or fov.json
        if len(filename.split(".")) != 2:
            warnings.warn(
                f"The file {filename} is not a valid FOV file and will be skipped from processing.",
                Warning,
            )
            return False, ""

        fov_name, extension = filename.split(".")

        # path no longer valid
        if not os.path.exists(path):
            warnings.warn(
                f"{path} doesn't exist but was recently created. This should be unreachable...",
                Warning,
            )
            return False, ""

        # avoids repeated processing in case of duplicated events
        if fov_name in self.processed_fovs:
            return False, fov_name

        # does not process moly points
        if fov_name in self.moly_points:
            return False, fov_name

        wait_time = 0
        if fov_name in self.fov_progress:
            if extension in self.fov_progress[fov_name]:
                while os.path.getsize(path) == 0:
                    # consider timed out fovs complete
                    if wait_time >= self.timeout:
                        del self.fov_progress[fov_name]
                        raise TimeoutError(f"timed out waiting for {path}...")

                    time.sleep(self.timeout / 10)
                    wait_time += self.timeout / 10

                self.fov_progress[fov_name][extension] = True

            if all(self.fov_progress[fov_name].values()):
                return True, fov_name

        elif extension == "bin":
            warnings.warn(f"Found unexpected bin file, {path}...", Warning)
            return False, ""

        return False, fov_name

    def processed(self, fov_name: str):
        """Notifies run structure that fov has been processed

        Args:
            fov_name (str):
                Name of FoV
        """
        self.processed_fovs.append(fov_name)

    def check_fov_progress(self) -> dict:
        """Condenses internal dictionary to show which fovs have finished

        Returns:
            dict
        """
        all_fovs = self.fov_progress.keys()
        moly_fovs = self.moly_points
        necessary_fovs = list(set(all_fovs).difference(moly_fovs))

        return {k: all(self.fov_progress[k].values()) for k in necessary_fovs}


class FOV_EventHandler(FileSystemEventHandler):
    """File event handler for FOV files

    Attributes:
        run_folder (str):
            path to run folder
        watcher_out (str):
            folder to save all callback results + log file
        run_structure (RunStructure):
            expected run file structure + fov_progress status
        fov_callback (Callable[[str, str], None]):
            callback to run on each fov
        run_callback (Callable[[None], None]):
            callback to run over the entire run
    """

    def __init__(
        self,
        run_folder: str,
        log_folder: str,
        fov_callback: Callable[[str, str], None],
        run_callback: Callable[[str], None],
        intermediate_callback: Callable[[str], None] = None,
        timeout: int = 1.03 * 60 * 60,
    ):
        """Initializes FOV_EventHandler

        Args:
            run_folder (str):
                path to run folder
            log_folder (str):
                path to save outputs to
            fov_callback (Callable[[str, str], None]):
                callback to run on each fov
            run_callback (Callable[[None], None]):
                callback to run over the entire run
            intermediate_callback (Callable[[None], None]):
                run callback overriden to run on each fov
            timeout (int):
                number of seconds to wait for non-null filesize before raising an error
        """
        super().__init__()
        self.run_folder = run_folder

        self.log_path = os.path.join(log_folder, f"{Path(run_folder).parts[-1]}_log.txt")
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        # create run structure
        self.run_structure = RunStructure(run_folder, timeout=timeout)

        self.fov_func = fov_callback
        self.run_func = run_callback
        self.inter_func = intermediate_callback
        self.inter_return_vals = None

        for root, dirs, files in os.walk(run_folder):
            for name in files:
                self.on_created(FileCreatedEvent(os.path.join(root, name)))

    def _run_callbacks(self, event: Union[DirCreatedEvent, FileCreatedEvent, FileMovedEvent]):
        # check if what's created is in the run structure
        try:
            if type(event) in [DirCreatedEvent, FileCreatedEvent]:
                file_trigger = event.src_path
            else:
                file_trigger = event.dest_path
            fov_ready, point_name = self.run_structure.check_run_condition(file_trigger)
        except TimeoutError as timeout_error:
            print(f"Encountered TimeoutError error: {timeout_error}")
            logf = open(self.log_path, "a")
            logf.write(
                f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- '
                f"{event.src_path} never reached non-zero file size...\n"
            )
            self.check_complete()
            return

        if fov_ready:
            print(f"Discovered {point_name}, beginning per-fov callbacks...")
            logf = open(self.log_path, "a")

            logf.write(
                f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- Extracting {point_name}\n'
            )

            # run per_fov callbacks
            logf.write(
                f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- '
                f"Running {self.fov_func.__name__} on {point_name}\n"
            )

            self.fov_func(self.run_folder, point_name)
            self.run_structure.processed(point_name)

            if self.inter_func:
                # clear plots contained in intermediate return values if set
                if self.inter_return_vals:
                    qc_plots = self.inter_return_vals.get("plot_qc_metrics", None)
                    mph_plot = self.inter_return_vals.get("plot_mph_metrics", None)

                    if qc_plots or mph_plot:
                        plt.cla()
                        plt.clf()
                        plt.close("all")

                self.inter_return_vals = self.inter_func(self.run_folder)

            logf.close()
            self.check_complete()

    def on_created(self, event: FileCreatedEvent):
        """Handles file creation events

        If FOV structure is completed, the fov callback, `self.fov_func` will be run over the data.
        This function is automatically called; users generally shouldn't call this function

        Args:
            event (FileCreatedEvent):
                file creation event
        """
        super().on_created(event)
        self._run_callbacks(event)

    def on_moved(self, event: FileMovedEvent):
        """Handles file renaming events

        If FOV structure is completed, the fov callback, `self.fov_func` will be run over the data.
        This function is automatically called; users generally shouldn't call this function

        Args:
            event (FileMovedEvent):
                file moved event
        """
        super().on_moved(event)
        self._run_callbacks(event)

    def check_complete(self):
        """Checks run structure fov_progress status

        If run is complete, all calbacks in `per_run` will be run over the whole run.
        """
        if all(self.run_structure.check_fov_progress().values()):
            logf = open(self.log_path, "a")

            logf.write(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- All FOVs finished\n')

            # run per_runs
            logf.write(
                f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- '
                f"Running {self.run_func.__name__} on whole run\n"
            )

            self.run_func(self.run_folder)


def start_watcher(
    run_folder: str,
    log_folder: str,
    fov_callback: Callable[[str, str], None],
    run_callback: Callable[[None], None],
    intermediate_callback: Callable[[str, str], None] = None,
    completion_check_time: int = 30,
    zero_size_timeout: int = 1.03 * 60 * 60,
):
    """Passes bin files to provided callback functions as they're created

    Args:
        run_folder (str):
            path to run folder
        log_folder (str):
            where to create log file
        fov_callback (Callable[[str, str], None]):
            function to run on each completed fov. assemble this using
            `watcher_callbacks.build_callbacks`
        run_callback (Callable[[None], None]):
            function ran once the run has completed. assemble this using
            `watcher_callbacks.build_callbacks`
        intermediate_callback (Callable[[None], None]):
            function defined as run callback overriden as fov callback. assemble this using
            `watcher_callbacks.build_callbacks`
        completion_check_time (int):
            how long to wait before checking watcher completion, in seconds.
            note, this doesn't effect the watcher itself, just when this wrapper function exits.
        zero_size_timeout (int):
            number of seconds to wait for non-zero file size
    """
    observer = Observer()
    event_handler = FOV_EventHandler(
        run_folder, log_folder, fov_callback, run_callback, intermediate_callback, zero_size_timeout
    )
    observer.schedule(event_handler, run_folder, recursive=True)
    observer.start()

    try:
        while not all(event_handler.run_structure.check_fov_progress().values()):
            time.sleep(completion_check_time)
    except KeyboardInterrupt:
        observer.stop()

    observer.stop()
    observer.join()
