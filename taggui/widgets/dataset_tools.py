"""
The Dataset Tools dock: runs the `pipeline` tool of dataset_tools over the
loaded directory as a subprocess (preview, run, undo), shows its progress and
console output, and hands the changed images to the auto-captioner.
"""
import shlex
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QUrl, Qt, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractScrollArea, QButtonGroup, QDockWidget,
                               QFileDialog, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QRadioButton,
                               QScrollArea, QVBoxLayout, QWidget)

from models.image_list_model import ImageListModel
from utils.big_widgets import TallPushButton
from utils.dataset_tools import (OVERRIDES_FILE_NAME, PASSES, PREVIEWS_DIRNAME,
                                 DatasetToolsEnvironment, EnvironmentCheck,
                                 PipelineProcess, get_backup_path,
                                 get_pipeline_directory, read_changed_paths,
                                 resolve_dataset_tools)
from utils.enums import CaptionPosition
from utils.image import Image
from utils.settings import DEFAULT_SETTINGS, get_settings
from utils.settings_widgets import (FocusedScrollSettingsComboBox,
                                    FocusedScrollSettingsSpinBox,
                                    SettingsBigCheckBox, SettingsLineEdit)
from utils.utils import pluralize
from widgets.auto_captioner import set_text_edit_height
from widgets.image_list import ImageList

WATERMARK_MODES = {'Paint out': 'inpaint', 'Cut off': 'trim'}
REFRAME_MODES = {'Best fit': 'best_fit', 'Square': 'square',
                 'Closest aspect ratio': 'closest'}
PARALLEL_OUTPUT_SUFFIX = '_prepared'
THREADS = 4
CONSOLE_LINE_COUNT = 8
# What the auto-captioner captions after a run: nothing, the images the run
# changed, or every image in the image list.
CAPTION_AFTER_RUN_NONE = 'None'
CAPTION_AFTER_RUN_CHANGED = 'Changed images'
CAPTION_AFTER_RUN_ALL = 'All listed images'
CAPTION_AFTER_RUN_CHOICES = (CAPTION_AFTER_RUN_NONE, CAPTION_AFTER_RUN_CHANGED,
                             CAPTION_AFTER_RUN_ALL)


class DatasetTools(QDockWidget):
    directory_reload_requested = Signal()
    load_directory_requested = Signal(object)
    # The images to select in the image list, as paths.
    select_images_requested = Signal(list)
    # Emitted when a pipeline process ends, with the kind of run: `preview`,
    # `run`, `undo` or `undo_files`.
    run_finished = Signal(str)
    # The images to caption after a run, as paths, and the caption position
    # for that run (`None` keeps the auto-captioner's setting).
    captioning_requested = Signal(list, object)

    def __init__(self, image_list_model: ImageListModel,
                 image_list: ImageList):
        super().__init__()
        self.image_list_model = image_list_model
        self.image_list = image_list
        self.settings = get_settings()
        self.directory_path: Path | None = None
        self.environment: DatasetToolsEnvironment | None = None
        self.is_environment_ready = False
        self.process = PipelineProcess(self)
        self.environment_check = EnvironmentCheck(self)
        self.run_kind: str | None = None
        self.last_finished_kind: str | None = None
        self.selected_paths_before_run: list[Path] = []
        # Set by the main window.
        self.is_captioning: Callable[[], bool] = lambda: False
        self.get_current_image: Callable[[], Image | None] = lambda: None
        # Whether the user changed the output mode by hand. Until then, the
        # mode follows the Reframe stage: a reframed dataset goes to a new
        # folder by default.
        self.is_output_mode_touched = False

        # Each `QDockWidget` needs a unique object name for saving its state.
        self.setObjectName('dataset_tools')
        self.setWindowTitle('Dataset Tools')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addLayout(self.create_buttons())
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.console_text_edit = QPlainTextEdit()
        set_text_edit_height(self.console_text_edit, CONSOLE_LINE_COUNT)
        self.console_text_edit.setReadOnly(True)
        self.console_text_edit.hide()
        layout.addWidget(self.console_text_edit)
        layout.addWidget(self.create_environment_group())
        layout.addWidget(self.create_stages_group())
        layout.addWidget(self.create_scope_and_output_group())
        layout.addWidget(self.create_review_group())
        layout.addStretch()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(container)
        self.setWidget(scroll_area)

        self.process.line_received.connect(self.handle_line)
        self.process.text_received.connect(self.append_console_text)
        self.process.finished.connect(self.handle_finished)
        self.environment_check.checked.connect(self.handle_environment_check)
        # The output mode follows the Reframe stage until the user sets it.
        self.follow_reframe_stage(self.reframe_check_box.checkState().value)
        self.update_buttons()
        self.refresh_environment()

    # Widgets

    def create_buttons(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        row = QHBoxLayout()
        self.preview_button = TallPushButton('Preview')
        self.preview_button.setToolTip(
            'Run the detection passes and draw the previews without '
            'changing any image.')
        self.run_button = TallPushButton('Run')
        self.run_button.setToolTip('Apply the ticked stages to the images.')
        self.undo_button = TallPushButton('Undo Last Run')
        self.undo_button.setToolTip(
            'Return the images of the last run to their originals.')
        row.addWidget(self.preview_button)
        row.addWidget(self.run_button)
        row.addWidget(self.undo_button)
        layout.addLayout(row)
        self.cancel_button = TallPushButton('Cancel')
        self.cancel_button.hide()
        layout.addWidget(self.cancel_button)
        self.preview_button.clicked.connect(lambda: self.start('preview'))
        self.run_button.clicked.connect(lambda: self.start('run'))
        self.undo_button.clicked.connect(lambda: self.start('undo'))
        self.cancel_button.clicked.connect(self.process.cancel)
        return layout

    def create_environment_group(self) -> QGroupBox:
        group = QGroupBox('Dataset tools')
        layout = QVBoxLayout(group)
        self.environment_label = QLabel()
        self.environment_label.setWordWrap(True)
        layout.addWidget(self.environment_label)
        row = QHBoxLayout()
        self.path_line_edit = SettingsLineEdit(
            key='dataset_tools_path',
            default=DEFAULT_SETTINGS['dataset_tools_path'])
        self.path_line_edit.setPlaceholderText(
            'dataset_tools folder or its venv (empty: ../dataset_tools)')
        self.path_line_edit.setClearButtonEnabled(True)
        self.path_line_edit.editingFinished.connect(self.refresh_environment)
        select_button = QPushButton('Select...')
        select_button.clicked.connect(self.select_dataset_tools_path)
        row.addWidget(self.path_line_edit, stretch=1)
        row.addWidget(select_button)
        layout.addLayout(row)
        return group

    def create_stages_group(self) -> QGroupBox:
        group = QGroupBox('Stages, in the order they run')
        layout = QVBoxLayout(group)

        self.borders_check_box = SettingsBigCheckBox(
            key='pipeline_stage_borders', default=True,
            text='Remove borders (frames, lines, text banners)')
        layout.addWidget(self.borders_check_box)

        self.watermark_check_box = SettingsBigCheckBox(
            key='pipeline_stage_watermark', default=True,
            text='Remove watermarks')
        self.watermark_mode_combo_box = FocusedScrollSettingsComboBox(
            key='pipeline_watermark_mode', default='Paint out')
        self.watermark_mode_combo_box.addItems(list(WATERMARK_MODES))
        layout.addLayout(self.create_option_row(
            self.watermark_check_box, self.watermark_mode_combo_box))

        self.reframe_check_box = SettingsBigCheckBox(
            key='pipeline_stage_reframe', default=False,
            text='Reframe photos of people')
        self.reframe_mode_combo_box = FocusedScrollSettingsComboBox(
            key='pipeline_reframe_mode', default='Best fit')
        self.reframe_mode_combo_box.addItems(list(REFRAME_MODES))
        self.reframe_check_box.stateChanged.connect(
            self.follow_reframe_stage)
        layout.addLayout(self.create_option_row(
            self.reframe_check_box, self.reframe_mode_combo_box))

        self.jpeg_cleanup_check_box = SettingsBigCheckBox(
            key='pipeline_stage_jpeg_cleanup', default=True,
            text='Clean JPEG artifacts below quality')
        self.jpeg_cleanup_threshold_spin_box = FocusedScrollSettingsSpinBox(
            key='pipeline_jpeg_cleanup_threshold', default=80, minimum=1,
            maximum=100)
        layout.addLayout(self.create_option_row(
            self.jpeg_cleanup_check_box, self.jpeg_cleanup_threshold_spin_box))

        self.face_masks_check_box = SettingsBigCheckBox(
            key='pipeline_stage_face_masks', default=False,
            text='Face masks (loss masks that hide faces)')
        layout.addWidget(self.face_masks_check_box)
        face_masks_row = QHBoxLayout()
        face_masks_row.setContentsMargins(30, 0, 0, 0)
        self.include_hair_check_box = SettingsBigCheckBox(
            key='pipeline_face_masks_include_hair', default=False,
            text='Include hair')
        self.invert_masks_check_box = SettingsBigCheckBox(
            key='pipeline_face_masks_invert', default=False,
            text='Invert (train only the faces)')
        face_masks_row.addWidget(self.include_hair_check_box)
        face_masks_row.addWidget(self.invert_masks_check_box)
        face_masks_row.addStretch()
        layout.addLayout(face_masks_row)

        layout.addWidget(self.create_horizontal_line())
        self.hide_underscore_folders_check_box = SettingsBigCheckBox(
            key='hide_underscore_folders',
            default=DEFAULT_SETTINGS['hide_underscore_folders'],
            text='Hide _ folders in the image list')
        self.hide_underscore_folders_check_box.setToolTip(
            'The output folders of the tools (_controls, _reframed and the '
            'others) are always hidden. This also hides every other '
            'subfolder whose name starts with an underscore.')
        self.hide_underscore_folders_check_box.stateChanged.connect(
            lambda _: self.directory_reload_requested.emit())
        layout.addWidget(self.hide_underscore_folders_check_box)
        return group

    @staticmethod
    def create_option_row(check_box, option_widget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(check_box)
        row.addWidget(option_widget)
        row.addStretch()
        check_box.stateChanged.connect(
            lambda state: option_widget.setEnabled(
                state == Qt.CheckState.Checked.value))
        option_widget.setEnabled(check_box.isChecked())
        return row

    @staticmethod
    def create_horizontal_line() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def create_scope_and_output_group(self) -> QGroupBox:
        group = QGroupBox('Images and output')
        layout = QVBoxLayout(group)
        scope_row = QHBoxLayout()
        self.whole_folder_radio_button = QRadioButton('All listed images')
        self.whole_folder_radio_button.setToolTip(
            'Every image in the image list, so hidden folders are left out.')
        self.whole_folder_radio_button.setChecked(True)
        self.selected_images_radio_button = QRadioButton('Selected images')
        # Radio buttons with the same parent are exclusive with each other,
        # so the two pairs need their own groups.
        self.scope_button_group = QButtonGroup(self)
        self.scope_button_group.addButton(self.whole_folder_radio_button)
        self.scope_button_group.addButton(self.selected_images_radio_button)
        scope_row.addWidget(self.whole_folder_radio_button)
        scope_row.addWidget(self.selected_images_radio_button)
        scope_row.addStretch()
        layout.addLayout(scope_row)
        layout.addWidget(self.create_horizontal_line())
        self.in_place_radio_button = QRadioButton(
            'In place (originals go to <folder>_backup)')
        self.in_place_radio_button.setChecked(True)
        layout.addWidget(self.in_place_radio_button)
        new_folder_row = QHBoxLayout()
        self.new_folder_radio_button = QRadioButton('New folder:')
        self.output_button_group = QButtonGroup(self)
        self.output_button_group.addButton(self.in_place_radio_button)
        self.output_button_group.addButton(self.new_folder_radio_button)
        self.new_folder_line_edit = QLineEdit()
        self.new_folder_line_edit.setPlaceholderText(
            'the dataset folder with "_prepared" added')
        new_folder_row.addWidget(self.new_folder_radio_button)
        new_folder_row.addWidget(self.new_folder_line_edit, stretch=1)
        layout.addLayout(new_folder_row)
        for radio_button in (self.in_place_radio_button,
                             self.new_folder_radio_button):
            radio_button.clicked.connect(self.mark_output_mode_touched)
        self.save_png_check_box = SettingsBigCheckBox(
            key='pipeline_save_png', default=False,
            text='Save changed images as PNG')
        layout.addWidget(self.save_png_check_box)
        layout.addWidget(self.create_horizontal_line())
        caption_row = QHBoxLayout()
        caption_label = QLabel('After the run, caption:')
        self.caption_after_run_combo_box = FocusedScrollSettingsComboBox(
            key='pipeline_caption_after_run', default=CAPTION_AFTER_RUN_NONE)
        self.caption_after_run_combo_box.addItems(
            list(CAPTION_AFTER_RUN_CHOICES))
        tooltip = ('When a run finishes, the auto-captioner starts on these '
                   'images with its own settings, without asking. A run into '
                   'a new folder loads that folder first. For "Changed '
                   'images" the caption position is "Overwrite all tags" for '
                   'that run only, because the prompt already feeds the old '
                   'caption back through {tags}; the setting of the '
                   'auto-captioner stays as it is.')
        caption_label.setToolTip(tooltip)
        self.caption_after_run_combo_box.setToolTip(tooltip)
        caption_row.addWidget(caption_label)
        caption_row.addWidget(self.caption_after_run_combo_box)
        caption_row.addStretch()
        layout.addLayout(caption_row)
        return group

    def create_review_group(self) -> QGroupBox:
        group = QGroupBox('Review')
        layout = QVBoxLayout(group)
        self.select_changed_button = QPushButton('Select Changed Images')
        self.select_changed_button.setToolTip(
            'Select the images the last run changed, or the last preview '
            'would change, to caption them.')
        self.select_changed_button.clicked.connect(self.select_changed_images)
        layout.addWidget(self.select_changed_button)
        sheets_row = QHBoxLayout()
        open_border_sheets_button = QPushButton('Open Border Sheets')
        open_border_sheets_button.clicked.connect(
            lambda: self.open_previews_folder('borders'))
        open_cleanup_sheets_button = QPushButton('Open Cleanup Sheets')
        open_cleanup_sheets_button.clicked.connect(
            lambda: self.open_previews_folder('cleanup'))
        sheets_row.addWidget(open_border_sheets_button)
        sheets_row.addWidget(open_cleanup_sheets_button)
        layout.addLayout(sheets_row)
        layout.addWidget(self.create_horizontal_line())
        layout.addWidget(QLabel('Reframe correction for the current image '
                                '(then Preview again):'))
        correction_row = QHBoxLayout()
        keep_button = QPushButton('Keep')
        keep_button.setToolTip('Leave this photo unchanged.')
        keep_button.clicked.connect(lambda: self.apply_correction('keep'))
        crop_button = QPushButton('Crop')
        crop_button.setToolTip(
            'Crop this photo even though the rules leave it unchanged.')
        crop_button.clicked.connect(lambda: self.apply_correction('crop'))
        self.people_line_edit = QLineEdit()
        self.people_line_edit.setPlaceholderText('people, e.g. 0,3')
        self.people_line_edit.setToolTip(
            'The subject is exactly these people, numbered as in the '
            'preview.')
        people_button = QPushButton('Set People')
        people_button.clicked.connect(lambda: self.apply_correction('people'))
        correction_row.addWidget(keep_button)
        correction_row.addWidget(crop_button)
        correction_row.addWidget(self.people_line_edit, stretch=1)
        correction_row.addWidget(people_button)
        layout.addLayout(correction_row)
        return group

    # Environment

    @Slot()
    def select_dataset_tools_path(self):
        path = QFileDialog.getExistingDirectory(
            parent=self, caption='Select the dataset_tools folder',
            dir=self.path_line_edit.text() or str(Path.home()))
        if path:
            self.path_line_edit.setText(path)
            self.settings.setValue('dataset_tools_path', path)
            self.refresh_environment()

    @Slot()
    def refresh_environment(self):
        self.environment = resolve_dataset_tools(self.path_line_edit.text())
        self.is_environment_ready = False
        if not self.environment.ok:
            self.set_environment_status(self.environment.error, is_error=True)
            self.update_buttons()
            return
        self.set_environment_status(
            f'Checking {self.environment.root}...', is_error=False)
        self.update_buttons()
        self.environment_check.start(self.environment)

    @Slot(bool, str)
    def handle_environment_check(self, is_ok: bool, message: str):
        self.is_environment_ready = is_ok
        if is_ok:
            self.set_environment_status(
                f'Ready: {self.environment.root}', is_error=False)
        else:
            self.set_environment_status(
                f'Broken: {message} ({self.environment.python})',
                is_error=True)
        self.update_buttons()

    def set_environment_status(self, text: str, is_error: bool):
        self.environment_label.setText(text)
        self.environment_label.setStyleSheet('color: red;' if is_error
                                             else '')

    # State

    def set_directory(self, directory_path: Path | None):
        self.directory_path = directory_path
        if directory_path is not None:
            self.new_folder_line_edit.setPlaceholderText(
                str(directory_path.parent
                    / (directory_path.name + PARALLEL_OUTPUT_SUFFIX)))
        self.update_buttons()

    def is_busy(self) -> bool:
        return self.process.is_running()

    def is_changing_files(self) -> bool:
        """Whether a run that writes images is active."""
        return self.is_busy() and self.run_kind != 'preview'

    def update_buttons(self):
        is_ready = (self.is_environment_ready
                    and self.directory_path is not None)
        is_busy = self.is_busy()
        self.preview_button.setEnabled(is_ready and not is_busy)
        self.run_button.setEnabled(is_ready and not is_busy)
        self.undo_button.setEnabled(is_ready and not is_busy)
        self.cancel_button.setVisible(is_busy)
        self.cancel_button.setEnabled(is_busy
                                      and not self.process.was_cancelled)
        self.select_changed_button.setEnabled(
            self.directory_path is not None and not is_busy)

    @Slot()
    def mark_output_mode_touched(self):
        self.is_output_mode_touched = True

    @Slot(int)
    def follow_reframe_stage(self, state: int):
        if self.is_output_mode_touched:
            return
        if state == Qt.CheckState.Checked.value:
            self.new_folder_radio_button.setChecked(True)
        else:
            self.in_place_radio_button.setChecked(True)

    # Jobs

    def get_output_folder(self) -> Path:
        text = self.new_folder_line_edit.text().strip()
        if text:
            return Path(text).expanduser()
        return (self.directory_path.parent
                / (self.directory_path.name + PARALLEL_OUTPUT_SUFFIX))

    def get_relative_paths(self, images: list[Image]) -> list[str]:
        relative_paths = []
        for image in images:
            try:
                relative_paths.append(
                    image.path.relative_to(self.directory_path).as_posix())
            except ValueError:
                continue
        return relative_paths

    def build_stages(self) -> dict:
        stages = {}
        if self.borders_check_box.isChecked():
            stages['borders'] = {}
        if self.watermark_check_box.isChecked():
            stages['watermark'] = {
                'mode': WATERMARK_MODES[
                    self.watermark_mode_combo_box.currentText()]}
        if self.reframe_check_box.isChecked():
            stages['reframe'] = {
                'mode': REFRAME_MODES[self.reframe_mode_combo_box
                                      .currentText()]}
        if self.jpeg_cleanup_check_box.isChecked():
            stages['jpeg_cleanup'] = {
                'threshold': self.jpeg_cleanup_threshold_spin_box.value()}
        if self.face_masks_check_box.isChecked():
            stages['face_masks'] = {
                'include_hair': self.include_hair_check_box.isChecked(),
                'invert': self.invert_masks_check_box.isChecked()}
        return stages

    def build_job(self, kind: str) -> dict | None:
        """The job for a preview, run or undo, or `None` with a message."""
        job = {'folder': self.directory_path.as_posix(), 'threads': THREADS}
        if kind == 'undo':
            job['undo'] = True
            return job
        stages = self.build_stages()
        if not stages:
            self.show_message('Tick at least one stage.')
            return None
        job['stages'] = stages
        # The job always names its images, so that the pipeline sees exactly
        # what the image list shows: the hidden folders are left out, and so
        # are images filtered out of the list.
        if self.selected_images_radio_button.isChecked():
            images = self.image_list.list_view.get_selected_images()
            if not images:
                self.show_message('No images are selected.')
                return None
        else:
            images = self.image_list_model.images
            if not images:
                self.show_message('There are no images to process.')
                return None
        job['files'] = self.get_relative_paths(images)
        if self.new_folder_radio_button.isChecked():
            output_folder = self.get_output_folder()
            if (output_folder == self.directory_path
                    or self.directory_path in output_folder.parents):
                self.show_message('The new folder must be outside the '
                                  'dataset folder.')
                return None
            job['output'] = {'mode': 'parallel',
                             'folder': output_folder.as_posix()}
        else:
            job['output'] = {'mode': 'in_place'}
        job['save_png'] = self.save_png_check_box.isChecked()
        job['dry_run'] = kind == 'preview'
        return job

    def start(self, kind: str, job: dict | None = None):
        if self.is_busy() or self.directory_path is None:
            return
        if kind != 'preview' and self.is_captioning():
            self.show_message('Wait for the auto-captioning to finish: this '
                              'changes the images it is reading.')
            return
        if job is None:
            job = self.build_job(kind)
            if job is None:
                return
        if kind == 'run' and job['output']['mode'] == 'in_place':
            image_count = len(job.get('files') or [])
            scope = (f'{image_count} selected '
                     f'{pluralize("image", image_count)}'
                     if image_count else 'the whole folder')
            reply = QMessageBox.question(
                self, 'Run Dataset Tools',
                f'Change {scope} in place? The originals of the changed '
                f'images go to the {Path(job["folder"]).name}_backup folder '
                f'next to it, and Undo Last Run puts them back.',
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.run_kind = kind
        self.selected_paths_before_run = [
            image.path
            for image in self.image_list.list_view.get_selected_images()]
        self.console_text_edit.clear()
        self.console_text_edit.show()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat('Starting...')
        self.progress_bar.show()
        extra_arguments = ['--dry-run'] if kind == 'preview' else []
        self.append_console_text(
            f'{kind.replace("_", " ").capitalize()} on {self.directory_path}')
        if not self.process.start(self.environment, job, extra_arguments):
            return
        self.update_buttons()

    # Output of the process

    @Slot(str)
    def append_console_text(self, text: str):
        if self.console_text_edit.isHidden():
            self.console_text_edit.show()
        self.console_text_edit.appendPlainText(text)

    @Slot(dict)
    def handle_line(self, line: dict):
        if line.get('summary'):
            self.append_console_text(self.format_summary(line))
            return
        if 'warning' in line:
            self.append_console_text(f'Warning: {line["warning"]}')
            return
        if line.get('undo') is True and 'path' in line:
            # A per-file undo line.
            if 'skipped' in line:
                self.append_console_text(
                    f'{line["path"]}: skipped ({line["skipped"]})')
            elif 'error' in line:
                self.append_console_text(f'{line["path"]}: {line["error"]}')
            else:
                details = [key for key in ('caption', 'mask')
                           if line.get(key) == 'restored']
                suffix = f' ({", ".join(details)} restored)' if details else ''
                self.append_console_text(f'{line["path"]}: restored{suffix}')
            return
        pass_name = line.get('pass')
        if pass_name and 'index' in line and 'total' in line:
            total = line['total']
            if self.progress_bar.maximum() != total:
                self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(line['index'])
            pass_number = (PASSES.index(pass_name) + 1
                           if pass_name in PASSES else 0)
            self.progress_bar.setFormat(
                f'{pass_name} ({pass_number}/{len(PASSES)}): %v / %m')
        if 'error' in line:
            self.append_console_text(
                f'{line.get("path", "?")}: {line["error"]}')

    @staticmethod
    def format_summary(line: dict) -> str:
        skipped_keys = {'summary', 'pass', 'dry_run', 'undo', 'plan',
                        'previews'}
        parts = [f'{key} {value}' for key, value in line.items()
                 if key not in skipped_keys]
        if line.get('undo'):
            title = 'Undo'
        elif 'pass' in line:
            title = f'Pass {line["pass"]}'
        elif line.get('dry_run'):
            title = 'Preview finished'
        else:
            title = 'Run finished'
        return f'{title}: {", ".join(parts)}'

    @Slot(int, bool)
    def handle_finished(self, exit_code: int, was_cancelled: bool):
        kind = self.run_kind
        self.run_kind = None
        self.last_finished_kind = kind
        self.progress_bar.hide()
        if was_cancelled:
            self.append_console_text(
                'Cancelled. The images written so far are logged; Undo Last '
                'Run puts them back.')
        elif exit_code != 0:
            self.append_console_text(f'Finished with errors (exit code '
                                     f'{exit_code}).')
        job = self.process.job or {}
        is_parallel = (job.get('output') or {}).get('mode') == 'parallel'
        if kind in ('run', 'undo', 'undo_files') and not is_parallel:
            # The reload rebuilds the image list from the files, so the
            # thumbnails and the viewer show the images as they are now.
            self.directory_reload_requested.emit()
            existing_paths = [path for path in self.selected_paths_before_run
                              if path.exists()]
            if existing_paths:
                self.select_images_requested.emit(existing_paths)
        self.update_buttons()
        self.run_finished.emit(kind or '')
        if kind != 'run' or was_cancelled or exit_code != 0:
            return
        caption_choice = self.caption_after_run_combo_box.currentText()
        if not is_parallel:
            self.caption_after_run(caption_choice)
            return
        output_folder = Path(job['output']['folder'])
        if caption_choice == CAPTION_AFTER_RUN_NONE:
            question = f'The prepared dataset is in {output_folder}. Load it?'
        else:
            question = (f'The prepared dataset is in {output_folder}. Load '
                        f'it and caption {caption_choice.lower()}?')
        reply = QMessageBox.question(
            self, 'Run finished', question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        # The report of the run is in the backup folder of the dataset the
        # run read, so the changed images are read before that folder is
        # replaced by the loaded one.
        changed_paths = read_changed_paths(self.directory_path,
                                           from_report=True,
                                           output_folder=output_folder)
        self.load_directory_requested.emit(output_folder)
        self.caption_after_run(caption_choice, changed_paths)

    def caption_after_run(self, caption_choice: str,
                          changed_paths: list[Path] | None = None):
        """
        Select the images the combo box names and hand them to the
        auto-captioner. The changed images come from the report of the run
        unless given; the listed images are the image list after the reload.
        """
        if caption_choice == CAPTION_AFTER_RUN_NONE:
            return
        if caption_choice == CAPTION_AFTER_RUN_CHANGED:
            if changed_paths is None:
                changed_paths = read_changed_paths(self.directory_path,
                                                   from_report=True)
            paths = changed_paths
            # The prompt feeds the old caption back as context, so the new
            # caption replaces it; the setting of the auto-captioner is not
            # touched.
            caption_position = CaptionPosition.OVERWRITE_ALL_TAGS
        else:
            paths = [image.path for image in self.image_list_model.images]
            caption_position = None
        paths = [path for path in paths if path.exists()]
        if not paths:
            self.append_console_text(
                f'Nothing to caption: no {caption_choice.lower()}.')
            return
        self.select_images_requested.emit(paths)
        self.append_console_text(
            f'Captioning {len(paths)} {pluralize("image", len(paths))} '
            f'({caption_choice.lower()})...')
        self.captioning_requested.emit(paths, caption_position)

    # Review

    @Slot()
    def select_changed_images(self):
        if self.directory_path is None:
            return
        from_report = self.last_finished_kind == 'run'
        paths = read_changed_paths(self.directory_path, from_report)
        if not paths:
            self.show_message('No changed images are recorded for this '
                              'folder. Run a Preview or a Run first.')
            return
        self.select_images_requested.emit(paths)
        source = 'the last run' if from_report else 'the last preview'
        self.append_console_text(f'Selected {len(paths)} '
                                 f'{pluralize("image", len(paths))} changed '
                                 f'by {source}.')

    def open_previews_folder(self, kind: str):
        if self.directory_path is None:
            return
        folder = get_pipeline_directory(self.directory_path) / PREVIEWS_DIRNAME
        folder = folder / kind
        if not folder.is_dir():
            self.show_message(f'There are no {kind} sheets yet. Run a Preview '
                              f'with that stage ticked.')
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def apply_correction(self, action: str):
        image = self.get_current_image()
        if image is None or self.directory_path is None:
            return
        try:
            relative_path = image.path.relative_to(self.directory_path)
        except ValueError:
            return
        if action == 'people':
            people = self.people_line_edit.text().strip()
            if not people:
                self.show_message('Enter the numbers of the people, as shown '
                                  'in the preview, e.g. 0,3.')
                return
            action = 'people=' + ''.join(people.split())
        overrides_path = self.directory_path / OVERRIDES_FILE_NAME
        lines = []
        if overrides_path.is_file():
            lines = overrides_path.read_text(
                encoding='utf-8-sig').splitlines()
        relative_path_string = relative_path.as_posix()
        kept_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                try:
                    parts = shlex.split(stripped, posix=True)
                except ValueError:
                    parts = []
                if (parts and parts[0].replace('\\', '/').lower()
                        == relative_path_string.lower()):
                    continue
            kept_lines.append(line)
        quoted_path = (f'"{relative_path_string}"'
                       if ' ' in relative_path_string
                       else relative_path_string)
        kept_lines.append(f'{quoted_path} {action}')
        overrides_path.write_text('\n'.join(kept_lines) + '\n',
                                  encoding='utf-8')
        self.append_console_text(f'{relative_path_string}: {action} written '
                                 f'to {OVERRIDES_FILE_NAME}. Preview again to '
                                 f'apply it.')

    # Per-image undo from the image list

    @Slot(list)
    def undo_images(self, images: list[Image]):
        if self.directory_path is None:
            return
        if self.is_busy():
            self.show_message('Wait for the running dataset tools job to '
                              'finish.')
            return
        if self.is_captioning():
            self.show_message('Wait for the auto-captioning to finish: '
                              'restoring an image changes what it is '
                              'reading.')
            return
        if not self.is_environment_ready:
            self.show_message('The dataset tools are not available. Set '
                              'their folder in the Dataset Tools dock.')
            return
        relative_paths = []
        captions_without_backup_count = 0
        for image in images:
            backup_path = get_backup_path(self.directory_path, image.path)
            if backup_path is None:
                continue
            relative_paths.append(
                image.path.relative_to(self.directory_path).as_posix())
            has_backup_caption = backup_path.with_suffix('.txt').is_file()
            has_caption = image.path.with_suffix('.txt').is_file()
            if has_caption and not has_backup_caption:
                captions_without_backup_count += 1
        if not relative_paths:
            self.show_message('None of the selected images has an original '
                              'in the backup folder.')
            return
        delete_captions = False
        if captions_without_backup_count:
            count = captions_without_backup_count
            reply = QMessageBox.question(
                self, 'Undo',
                f'{count} {pluralize("image", count)} '
                f'{"has" if count == 1 else "have"} empty captions. Delete '
                f'existing captions?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Cancel:
                return
            delete_captions = reply == QMessageBox.StandardButton.Yes
        job = {'folder': self.directory_path.as_posix(), 'undo': True,
               'files': relative_paths,
               'delete_captions_without_backup': delete_captions,
               'threads': THREADS}
        self.start('undo_files', job)

    def show_message(self, text: str):
        QMessageBox.information(self, 'Dataset Tools', text)
