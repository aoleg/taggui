"""
The connection to the `pipeline` tool of dataset_tools: where its interpreter
is, and a process wrapper that runs a job and turns the JSON lines it prints
into signals. Nothing from dataset_tools is imported; the tool runs in its
own virtual environment as a subprocess.
"""
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from utils.settings import DEFAULT_SETTINGS, get_settings

# The dataset tools keep the originals and their run folders in
# `<folder>_backup` next to the dataset folder, outside it, so a trainer that
# walks every subfolder never sees them. Until 2026-10-11 they were in
# `_backup` inside the folder; the next run of a tool moves that folder out.
BACKUP_SUFFIX = '_backup'
OLD_BACKUP_DIRNAME = '_backup'
# The file that marks a folder as the backup folder of the dataset tools.
BACKUP_MARK = '_dataset_tools.json'
RUN_DIRNAME = '_pipeline'
PREVIEWS_DIRNAME = 'previews'
MASKS_DIRNAME = 'masks'
JOB_FILE_NAME = 'job.json'
PLAN_FILE_NAME = 'plan.json'
REPORT_FILE_NAME = 'report.jsonl'
OVERRIDES_FILE_NAME = 'overrides.txt'
# The passes of the pipeline, in order, for the progress bar text.
PASSES = ['borders', 'watermark', 'reframe', 'faces', 'quality', 'compose',
          'benefit', 'apply', 'previews']
# Seconds to wait for the process to end on its own after a cancel request
# before it is killed.
CANCEL_GRACE_SECONDS = 5


def get_taggui_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).parent.parent.parent


@dataclass
class DatasetToolsEnvironment:
    """Where dataset_tools and its interpreter are, or why they are not."""
    root: Path | None = None
    python: Path | None = None
    script: Path | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def get_venv_python(venv_path: Path) -> Path:
    if sys.platform == 'win32':
        return venv_path / 'Scripts' / 'python.exe'
    return venv_path / 'bin' / 'python'


def resolve_dataset_tools(
        path_setting: str | None = None) -> DatasetToolsEnvironment:
    """
    Find the dataset_tools checkout from the setting, which may name the
    checkout or its `venv` folder. An empty setting means the sibling folder
    `../dataset_tools` of this checkout.
    """
    if path_setting is None:
        path_setting = get_settings().value(
            'dataset_tools_path',
            defaultValue=DEFAULT_SETTINGS['dataset_tools_path'], type=str)
    path_setting = (path_setting or '').strip()
    if path_setting:
        path = Path(path_setting).expanduser()
    else:
        path = get_taggui_root().parent / 'dataset_tools'
    if not path.is_dir():
        return DatasetToolsEnvironment(
            error=f'The folder {path} does not exist.')
    # The setting may point at the `venv` folder instead of the checkout.
    if get_venv_python(path).is_file() and not (path / 'venv').is_dir():
        root = path.parent
        python = get_venv_python(path)
    else:
        root = path
        python = get_venv_python(path / 'venv')
    script = root / 'pipeline' / 'pipeline.py'
    if not python.is_file():
        return DatasetToolsEnvironment(
            root=root, error=f'No virtual environment at {python.parent.parent}.'
                             f' Run the install.bat of dataset_tools first.')
    if not script.is_file():
        return DatasetToolsEnvironment(
            root=root, python=python,
            error=f'{script} is missing. The dataset_tools checkout needs '
                  f'the pipeline tool.')
    return DatasetToolsEnvironment(root=root, python=python, script=script)


def get_backup_directories(folder: Path) -> list[Path]:
    """
    Where the originals of the folder can be: `<folder>_backup` next to it
    first, then the old `_backup` inside it, for a folder no tool has run on
    since the backup moved out.
    """
    return [folder.with_name(folder.name + BACKUP_SUFFIX),
            folder / OLD_BACKUP_DIRNAME]


def get_pipeline_directory(folder: Path) -> Path:
    """
    The run folder of the pipeline: in `<folder>_backup`, or in the old
    `_backup` while only that one exists.
    """
    backup_directory, old_backup_directory = get_backup_directories(folder)
    if not backup_directory.is_dir() and old_backup_directory.is_dir():
        return old_backup_directory / RUN_DIRNAME
    return backup_directory / RUN_DIRNAME


def get_backup_path(folder: Path, image_path: Path) -> Path | None:
    """
    The backup of an image, if there is one: the untouched original in the
    backup folder, under the same relative path. An image the pipeline wrote
    as PNG keeps its stem, so the backup is looked for with the JPEG
    suffixes as well.
    """
    try:
        relative_path = image_path.relative_to(folder)
    except ValueError:
        return None
    for backup_directory in get_backup_directories(folder):
        backup_path = backup_directory / relative_path
        if backup_path.is_file():
            return backup_path
        if image_path.suffix.lower() == '.png':
            for suffix in ('.jpg', '.jpeg', '.jpe', '.jfif', '.JPG', '.JPEG'):
                candidate = backup_path.with_suffix(suffix)
                if candidate.is_file():
                    return candidate
    return None


def get_preview_paths(folder: Path,
                      image_path: Path) -> tuple[Path | None, Path | None]:
    """
    The dry-run preview drawn for an image (the reframe verdict first, else
    the watermark mask) and the face mask the pipeline made for it, each
    `None` when there is none.
    """
    try:
        relative_path = image_path.relative_to(folder).as_posix()
    except ValueError:
        return None, None
    previews_directory = get_pipeline_directory(folder) / PREVIEWS_DIRNAME
    preview_path = None
    for kind in ('reframe', 'watermark'):
        candidate = previews_directory / kind / (relative_path + '.jpg')
        if candidate.is_file():
            preview_path = candidate
            break
    mask_path = image_path.parent / MASKS_DIRNAME / (image_path.stem + '.png')
    if not mask_path.is_file():
        mask_path = None
    return preview_path, mask_path


def read_changed_paths(folder: Path, from_report: bool) -> list[Path]:
    """
    The images the last pipeline run changed (from `report.jsonl`) or the last
    dry run planned to change (from `plan.json`), as absolute paths. An image
    the run wrote under another name is returned under that name.
    """
    pipeline_directory = get_pipeline_directory(folder)
    changed = []
    if from_report:
        report_path = pipeline_directory / REPORT_FILE_NAME
        if not report_path.is_file():
            return []
        for line in report_path.read_text(encoding='utf-8').splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get('changed') and 'error' not in entry:
                changed.append(entry.get('written') or entry['path'])
    else:
        plan_path = pipeline_directory / PLAN_FILE_NAME
        if not plan_path.is_file():
            return []
        try:
            plan = json.loads(plan_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return []
        for relative_path, entry in plan.get('images', {}).items():
            if entry.get('changed'):
                changed.append(relative_path)
    return [folder / relative_path for relative_path in changed]


class EnvironmentCheck(QObject):
    """
    Runs `import torch, ultralytics` with the dataset_tools interpreter in
    the background, so a broken environment is reported before a job starts.
    """
    checked = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process: QProcess | None = None

    def start(self, environment: DatasetToolsEnvironment):
        if self.process is not None:
            self.process.kill()
            self.process.deleteLater()
        self.process = QProcess(self)
        self.process.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)
        self.process.finished.connect(self.handle_finished)
        self.process.errorOccurred.connect(
            lambda _: self.checked.emit(
                False, f'Could not start {environment.python}.'))
        self.process.start(str(environment.python),
                           ['-c', 'import torch, ultralytics'])

    def handle_finished(self, exit_code: int, _):
        output = bytes(self.process.readAllStandardOutput()).decode(
            'utf-8', errors='replace').strip()
        if exit_code == 0:
            self.checked.emit(True, 'ready')
        else:
            last_line = output.splitlines()[-1] if output else 'import failed'
            self.checked.emit(False, last_line)


class PipelineProcess(QObject):
    """
    One pipeline run: writes the job file, starts the interpreter, and turns
    the JSON lines on stdout into signals. Anything that is not a JSON line,
    and everything on stderr, is forwarded as text.
    """
    line_received = Signal(dict)
    text_received = Signal(str)
    started = Signal()
    # The exit code and whether the run was cancelled.
    finished = Signal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process: QProcess | None = None
        self.buffer = b''
        self.was_cancelled = False
        self.job: dict | None = None

    def is_running(self) -> bool:
        return (self.process is not None
                and self.process.state() != QProcess.ProcessState.NotRunning)

    def start(self, environment: DatasetToolsEnvironment, job: dict,
              extra_arguments: list[str] | None = None) -> bool:
        if self.is_running():
            return False
        # The job file goes to the temporary folder, not to the run folder
        # of the pipeline: the pipeline makes its backup folder itself (and
        # moves an old `_backup` into it), then keeps a copy of the job there.
        job_directory = Path(tempfile.gettempdir()) / 'taggui'
        job_directory.mkdir(parents=True, exist_ok=True)
        job_path = job_directory / f'pipeline_{os.getpid()}_{JOB_FILE_NAME}'
        job_path.write_text(json.dumps(job, ensure_ascii=False, indent=1),
                            encoding='utf-8')
        self.job = job
        self.buffer = b''
        self.was_cancelled = False
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(environment.script.parent))
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.handle_finished)
        self.process.errorOccurred.connect(self.handle_error)
        arguments = [str(environment.script), '--job', str(job_path)]
        arguments.extend(extra_arguments or [])
        self.process.start(str(environment.python), arguments)
        self.started.emit()
        return True

    def cancel(self):
        if not self.is_running():
            return
        self.was_cancelled = True
        self.text_received.emit('Cancelling...')
        if sys.platform == 'win32':
            # `terminate()` only posts a window-close message, which a console
            # process does not have, and `kill()` leaves the writer processes
            # of the pipeline orphaned, so the whole process tree is ended.
            QProcess.startDetached(
                'taskkill', ['/PID', str(self.process.processId()), '/T',
                             '/F'])
        else:
            self.process.terminate()
        QTimer.singleShot(CANCEL_GRACE_SECONDS * 1000, self.kill_if_running)

    def kill_if_running(self):
        if self.is_running():
            self.process.kill()

    def read_stdout(self):
        self.buffer += bytes(self.process.readAllStandardOutput())
        while b'\n' in self.buffer:
            raw_line, self.buffer = self.buffer.split(b'\n', 1)
            self.handle_line(raw_line.decode('utf-8', errors='replace'))

    def read_stderr(self):
        text = bytes(self.process.readAllStandardError()).decode(
            'utf-8', errors='replace')
        for line in text.splitlines():
            if line.strip():
                self.text_received.emit(line.rstrip())

    def handle_line(self, line: str):
        line = line.rstrip('\r')
        if not line.strip():
            return
        if line.startswith('{'):
            try:
                self.line_received.emit(json.loads(line))
                return
            except json.JSONDecodeError:
                pass
        self.text_received.emit(line)

    def handle_error(self, error: QProcess.ProcessError):
        if error == QProcess.ProcessError.FailedToStart:
            self.text_received.emit(
                f'Could not start {self.process.program()}.')
            self.finished.emit(-1, False)

    def handle_finished(self, exit_code: int, _):
        if self.buffer.strip():
            self.handle_line(self.buffer.decode('utf-8', errors='replace'))
            self.buffer = b''
        self.read_stderr()
        self.finished.emit(exit_code, self.was_cancelled)
