import os
import time
import json
from datetime import datetime
from typing import Callable, List, Tuple
from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer


class RunStructure:
    """Expected bin and json files

    Attributes:
        completion (dict): Whether or not an expected file has been created
    """
    def __init__(self, run_folder: str, timeout: int = 10 * 60):
        """ initializes RunStructure by parsing run json within provided run folder

        Args:
            run_folder (str):
                path to run folder
            timeout (int):
                number of seconds to wait for non-null filesize before raising an error
        """
        self.timeout = timeout
        self.completion = {}

        # find run .json and get parameters
        with open(f'{run_folder}.json', 'r') as f:
            run_metadata = json.load(f)

        # parse run_metadata and populate expected structure
        for fov in run_metadata.get('fovs', ()):
            run_order = fov.get('runOrder', -1)
            scan = fov.get('scanCount', -1)
            if run_order * scan < 0:
                raise KeyError(f"Could not locate keys in {run_folder}.json")

            fov_name = f'fov-{run_order}-scan-{scan}'
            self.completion[fov_name] = {
                'json': False,
                'bin': False,
            }

    def check_run_condition(self, path: str, check_interval: int = 10) -> Tuple[bool, str]:
        """Checks if all requisite files exist and are complete

        Args:
            path (str):
                path to expected file
            check_interval (int):
                number of seconds to wait before re-checking filesize

        Raises:
            TimeoutError

        Returns:
            (bool, str):
                whether or not both json and bin files exist, as well as the name of the point
        """

        # TODO: check watchdog path depth

        fov_name, extension = path.split('.')[0:2]
        wait_time = 0
        if fov_name in self.completion:
            if extension in self.completion[fov_name]:
                while os.path.getsize(path) == 0:
                    # consider timed out fovs complete
                    if wait_time >= self.timeout:
                        for ext in self.completion[fov_name].keys():
                            self.completion[fov_name][ext] = True
                        raise TimeoutError(f'timed out waiting for {path}...')

                    time.sleep(check_interval)
                    wait_time += check_interval

                self.completion[fov_name][extension] = True

            if all(self.completion[fov_name].values):
                return True, fov_name

        elif extension == 'bin':
            raise KeyError(f'Found unexpected file, {path}...')

        return False, fov_name

    def check_completion(self) -> dict:
        """Condenses internal dictionary to show which fovs have finished

        Returns:
            dict
        """
        return {k: all(self.completion[k].values) for k in self.completion}


class FOV_EventHandler(FileSystemEventHandler):
    """File event handler for FOV files

    Attributes:
        run_folder (str):
            path to run folder
        watcher_out (str):
            folder to save all callback results + log file
        run_structure (RunStructure):
            expected run file structure + completion status
        per_fov (list):
            callbacks to run on each fov
        per_run (list):
            callbacks to run over the entire run
    """
    def __init__(self, run_folder: str, per_fov: List[Callable[[str, str, str], None]],
                 per_run: List[Callable[[str, str], None]], timeout: int = 10 * 60):
        """Initializes FOV_EventHandler

        Args:
            run_folder (str):
                path to run folder
            per_fov (list):
                callbacks to run on each fov
            per_run (list):
                callbacks to run over the entire run
            timeout (int):
                number of seconds to wait for non-null filesize before raising an error
        """
        super().__init__()
        self.run_folder = run_folder

        self.watcher_out = os.path.join(run_folder, 'watcher_outs')

        if not os.path.exists(self.watcher_out):
            os.makedirs(self.watcher_out)

        # create run structure
        self.run_structure = RunStructure(run_folder, timeout=timeout)

        self.per_fov = per_fov
        self.per_run = per_run

        for root, dirs, files in os.walk(run_folder):
            for name in files:
                self.on_created(FileCreatedEvent(os.path.join(root, name)))

    def on_created(self, event: FileCreatedEvent):
        """Handles file creation events

        If FOV structure is completed, all callbacks in `per_fov` will be run over the data.
        This function is automatically called; users generally shouldn't call this function

        Args:
            event (FileCreatedEvent):
                file creation event
        """
        super().on_created(event)

        log_file_path = os.path.join(self.watcher_out, 'log.txt')

        # check if what's created is in the run structure
        try:
            fov_ready, point_name = self.run_structure.check_run_condition(event.src_path)
        except TimeoutError as timeout_error:
            print('Encountered TimeoutError error: ' + timeout_error)
            logf = open(log_file_path, 'a')
            logf.write(
                f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- '
                f'{event.src_path} never reached non-zero file size...'
            )
            self.check_complete()

        if fov_ready:
            logf = open(log_file_path, 'a')

            logf.write(
                f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- '
                f'Extracting {point_name}'
            )

            # run per_fov callbacks
            for fov_func in self.per_fov:
                logf.write(
                    f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- '
                    f'Running {fov_func.__name__} on {point_name}'
                )
                fov_func(self.run_folder, point_name, self.watcher_out)

            logf.close()
            self.check_complete()

    def check_complete(self):
        """Checks run structure completion status

        If run is complete, all calbacks in `per_run` will be run over the whole run.
        """
        if all(self.run_structure.check_completion().values()):
            log_file_path = os.path.join(self.watcher_out, 'log.txt')
            logf = open(log_file_path, 'a')

            logf.write(
                f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- '
                f'All FOVs finished'
            )

            # run per_runs
            for run_func in self.per_run:
                logf.write(
                    f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} -- '
                    f'Running {run_func.__name__} on whole run'
                )
                run_func(self.run_folder, self.watcher_out)


def start_watcher(run_folder: str, per_fov: List[Callable[[str, str, str], None]],
                  per_run: List[Callable[[str, str], None]],
                  completion_check_time: int = 30):
    """ Passes bin files to provided callback functions as they're created

    Args:
        run_folder (str):
            path to run folder
        per_fov (list):
            list of functions to pass bin files
        per_run (list):
            list of functions to pass whole run
        completion_check_time (int):
            how long to wait before checking watcher completion, in seconds.
            note, this doesn't effect the watcher itself, just when this wrapper function exits.
    """
    observer = Observer()
    event_handler = FOV_EventHandler(run_folder, per_fov, per_run)
    observer.schedule(event_handler, run_folder, recursive=True)
    observer.start()

    try:
        while not all(event_handler.run_structure.check_completion().values()):
            time.sleep(completion_check_time)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()