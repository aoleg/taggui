import sys

import requests
from PySide6.QtCore import QModelIndex, Qt, Signal, Slot
from PySide6.QtGui import QFontMetrics, QTextCursor
from PySide6.QtWidgets import (QAbstractScrollArea, QDockWidget, QFormLayout,
                               QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

from auto_captioning.captioning_thread import CaptioningThread
from dialogs.caption_multiple_images_dialog import CaptionMultipleImagesDialog
from dialogs.cloud_captioning_consent_dialog import \
    CloudCaptioningConsentDialog
from models.image_list_model import ImageListModel
from utils.big_widgets import TallPushButton
from utils.enums import CaptionPosition
from utils.settings import get_settings, get_tag_separator
from utils.settings_widgets import (FocusedScrollSettingsComboBox,
                                    FocusedScrollSettingsDoubleSpinBox,
                                    FocusedScrollSettingsSpinBox,
                                    SettingsBigCheckBox, SettingsLineEdit,
                                    SettingsPlainTextEdit)
from utils.utils import pluralize
from widgets.image_list import ImageList

# `is_cloud` backends require an API key field and trigger the cloud
# captioning consent warning. `has_model_list` backends can serve more than
# one model at a time, so a model dropdown (populated via `Connect`) is shown
# instead of the fixed single-model status text.
BACKENDS = {
    'llama.cpp': {'default_url': 'http://localhost:8080', 'is_cloud': False,
                 'has_model_list': False},
    'koboldcpp': {'default_url': 'http://localhost:5001', 'is_cloud': False,
                 'has_model_list': False},
    'LM Studio': {'default_url': 'http://localhost:1234', 'is_cloud': False,
                 'has_model_list': True},
    'Cloud (OpenAI-compatible)': {'default_url': 'https://api.openai.com',
                                  'is_cloud': True, 'has_model_list': True}
}


def set_text_edit_height(text_edit: QPlainTextEdit, line_count: int):
    """
    Set the height of a text edit to the height of a given number of lines.
    """
    # From https://stackoverflow.com/a/46997337.
    document = text_edit.document()
    font_metrics = QFontMetrics(document.defaultFont())
    margins = text_edit.contentsMargins()
    height = int(font_metrics.lineSpacing() * line_count
                 + margins.top() + margins.bottom()
                 + document.documentMargin() * 2
                 + text_edit.frameWidth() * 2)
    text_edit.setFixedHeight(height)


class HorizontalLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Raised)


class CaptionSettingsForm(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        basic_settings_form = QFormLayout()
        basic_settings_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        basic_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.backend_combo_box = FocusedScrollSettingsComboBox(
            key='api_backend')
        self.backend_combo_box.addItems(list(BACKENDS))
        self.base_url_line_edit = SettingsLineEdit(
            key='api_base_url',
            default=BACKENDS[self.backend_combo_box.currentText()][
                'default_url'])
        self.base_url_line_edit.setClearButtonEnabled(True)
        self.api_key_label = QLabel('API key')
        self.api_key_line_edit = SettingsLineEdit(key='api_key')
        self.api_key_line_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_line_edit.setPlaceholderText(
            'Only required by some cloud providers')
        self.api_key_line_edit.setClearButtonEnabled(True)
        self.model_label = QLabel('Model')
        self.model_combo_box = FocusedScrollSettingsComboBox(key='api_model')
        # `setEditable()` must be called before `addItems()` to preserve any
        # custom model that was set.
        self.model_combo_box.setEditable(True)
        self.model_combo_box.addItems([])
        # `QComboBox` defaults to a `Preferred` horizontal size policy, so
        # with no items (or short ones) it stays narrow instead of matching
        # the width of the other fields under `ExpandingFieldsGrow`.
        self.model_combo_box.setSizePolicy(QSizePolicy.Policy.Expanding,
                                           QSizePolicy.Policy.Fixed)
        connect_container = QWidget()
        connect_layout = QHBoxLayout(connect_container)
        connect_layout.setContentsMargins(0, 0, 0, 0)
        self.connect_button = QPushButton('Connect')
        self.connection_status_label = QLabel()
        connect_layout.addWidget(self.connect_button)
        connect_layout.addWidget(self.connection_status_label, 1)
        self.system_prompt_text_edit = SettingsPlainTextEdit(
            key='system_prompt',
            default='You are an expert at writing detailed, accurate '
                    'captions for images used to train AI models.')
        set_text_edit_height(self.system_prompt_text_edit, 4)
        self.prompt_text_edit = SettingsPlainTextEdit(
            key='prompt', default='Describe the image in detail.')
        set_text_edit_height(self.prompt_text_edit, 4)
        self.caption_start_line_edit = SettingsLineEdit(key='caption_start')
        self.caption_start_line_edit.setClearButtonEnabled(True)
        self.caption_position_combo_box = FocusedScrollSettingsComboBox(
            key='caption_position')
        self.caption_position_combo_box.addItems(list(CaptionPosition))
        self.remove_tag_separators_container = QWidget()
        remove_tag_separators_layout = QHBoxLayout(
            self.remove_tag_separators_container)
        remove_tag_separators_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        remove_tag_separators_layout.setContentsMargins(0, 0, 0, 0)
        self.remove_tag_separators_check_box = SettingsBigCheckBox(
            key='remove_tag_separators', default=True)
        remove_tag_separators_label = QLabel(
            'Remove tag separators in caption')
        remove_tag_separators_layout.addWidget(remove_tag_separators_label)
        remove_tag_separators_layout.addWidget(
            self.remove_tag_separators_check_box)
        basic_settings_form.addRow('Backend', self.backend_combo_box)
        basic_settings_form.addRow('API base URL', self.base_url_line_edit)
        basic_settings_form.addRow(self.api_key_label, self.api_key_line_edit)
        basic_settings_form.addRow(self.model_label, self.model_combo_box)
        basic_settings_form.addRow('', connect_container)
        basic_settings_form.addRow('System prompt',
                                   self.system_prompt_text_edit)
        basic_settings_form.addRow('Prompt', self.prompt_text_edit)
        basic_settings_form.addRow('Start caption with',
                                   self.caption_start_line_edit)
        basic_settings_form.addRow('Caption position',
                                   self.caption_position_combo_box)
        basic_settings_form.addRow(self.remove_tag_separators_container)

        self.toggle_advanced_settings_form_button = TallPushButton(
            'Show Advanced Settings')

        self.advanced_settings_form_container = QWidget()
        advanced_settings_form = QFormLayout(
            self.advanced_settings_form_container)
        advanced_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        advanced_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.max_tokens_spin_box = FocusedScrollSettingsSpinBox(
            key='max_tokens', default=300, minimum=1, maximum=8192)
        # The temperature must be positive.
        self.temperature_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='temperature', default=1, minimum=0.01, maximum=2)
        self.temperature_spin_box.setSingleStep(0.01)
        self.top_p_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='top_p', default=1, minimum=0, maximum=1)
        self.top_p_spin_box.setSingleStep(0.01)
        self.top_k_spin_box = FocusedScrollSettingsSpinBox(
            key='top_k', default=50, minimum=0, maximum=200)
        self.repetition_penalty_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='repetition_penalty', default=1, minimum=1, maximum=2)
        self.repetition_penalty_spin_box.setSingleStep(0.01)
        self.allow_reasoning_check_box = SettingsBigCheckBox(
            key='allow_reasoning', default=False)
        advanced_settings_form.addRow('Maximum tokens',
                                      self.max_tokens_spin_box)
        advanced_settings_form.addRow('Temperature',
                                      self.temperature_spin_box)
        advanced_settings_form.addRow('Top-p', self.top_p_spin_box)
        advanced_settings_form.addRow('Top-k', self.top_k_spin_box)
        advanced_settings_form.addRow('Repetition penalty',
                                      self.repetition_penalty_spin_box)
        advanced_settings_form.addRow('Allow reasoning',
                                      self.allow_reasoning_check_box)
        # Reasoning is generated before the caption and counts against the
        # token limit, so a low limit can be used up before any caption is
        # generated. A disabled label is used to style the hint as secondary
        # text.
        reasoning_hint_label = QLabel('Use 2048 or more maximum tokens for '
                                      'reasoning models.')
        reasoning_hint_label.setWordWrap(True)
        reasoning_hint_label.setEnabled(False)
        advanced_settings_form.addRow('', reasoning_hint_label)
        self.advanced_settings_form_container.hide()

        self.addLayout(basic_settings_form)
        self.horizontal_line = HorizontalLine()
        self.addWidget(self.horizontal_line)
        self.addWidget(self.toggle_advanced_settings_form_button)
        self.addWidget(self.advanced_settings_form_container)
        self.addStretch()

        self.backend_combo_box.currentTextChanged.connect(
            self.update_backend_dependent_ui)
        self.connect_button.clicked.connect(self.test_connection)
        self.toggle_advanced_settings_form_button.clicked.connect(
            self.toggle_advanced_settings_form)
        self.update_backend_dependent_ui(self.backend_combo_box.currentText())

    @Slot(str)
    def update_backend_dependent_ui(self, backend: str):
        config = BACKENDS[backend]
        self.api_key_label.setVisible(config['is_cloud'])
        self.api_key_line_edit.setVisible(config['is_cloud'])
        self.model_label.setVisible(config['has_model_list'])
        self.model_combo_box.setVisible(config['has_model_list'])
        if not self.base_url_line_edit.text().strip():
            self.base_url_line_edit.setText(config['default_url'])

    def get_auth_headers(self) -> dict:
        api_key = self.api_key_line_edit.text().strip()
        if api_key:
            return {'Authorization': f'Bearer {api_key}'}
        return {}

    @Slot()
    def test_connection(self):
        backend = self.backend_combo_box.currentText()
        has_model_list = BACKENDS[backend]['has_model_list']
        base_url = self.base_url_line_edit.text().strip().rstrip('/')
        if not base_url:
            self.connection_status_label.setStyleSheet('color: red;')
            self.connection_status_label.setText('API base URL is not set.')
            return
        try:
            response = requests.get(f'{base_url}/v1/models',
                                    headers=self.get_auth_headers(),
                                    timeout=10)
            response.raise_for_status()
            models = response.json().get('data', [])
            model_names = [model['id'] for model in models]
        except requests.RequestException as exception:
            self.connection_status_label.setStyleSheet('color: red;')
            self.connection_status_label.setText(f'Connection failed: '
                                                  f'{exception}')
            return
        self.connection_status_label.setStyleSheet('color: green;')
        if has_model_list:
            current_model = self.model_combo_box.currentText()
            self.model_combo_box.clear()
            self.model_combo_box.addItems(model_names)
            if current_model:
                self.model_combo_box.setCurrentText(current_model)
            elif model_names:
                self.model_combo_box.setCurrentText(model_names[0])
            self.connection_status_label.setText(
                f'Connected. {len(model_names)} '
                f'{pluralize("model", len(model_names))} available.')
        else:
            model_name = model_names[0] if model_names else 'unknown'
            self.connection_status_label.setText(
                f'Connected. Loaded model: {model_name}')

    @Slot()
    def toggle_advanced_settings_form(self):
        if self.advanced_settings_form_container.isHidden():
            self.advanced_settings_form_container.show()
            self.toggle_advanced_settings_form_button.setText(
                'Hide Advanced Settings')
        else:
            self.advanced_settings_form_container.hide()
            self.toggle_advanced_settings_form_button.setText(
                'Show Advanced Settings')

    def get_caption_settings(self) -> dict:
        backend = self.backend_combo_box.currentText()
        allow_reasoning = self.allow_reasoning_check_box.isChecked()
        generation_parameters = {
            'max_tokens': self.max_tokens_spin_box.value(),
            'temperature': self.temperature_spin_box.value(),
            'top_p': self.top_p_spin_box.value(),
            'top_k': self.top_k_spin_box.value(),
            'repetition_penalty': self.repetition_penalty_spin_box.value(),
            # Reasoning is toggled by the chat template, not by the sampler, so
            # the flag has to be passed as a template variable. A top-level
            # `enable_thinking` is silently ignored.
            'chat_template_kwargs': {'enable_thinking': allow_reasoning}
        }
        if not allow_reasoning:
            # LM Studio ignores `chat_template_kwargs` and instead maps
            # `reasoning_effort` onto whatever reasoning fields the model
            # declares, so both have to be sent to cover every backend. Only
            # `none` turns reasoning off; `low` still reasons in full. When
            # reasoning is allowed, the parameter is left out so that the model
            # uses its own default.
            generation_parameters['reasoning_effort'] = 'none'
        return {
            'backend': backend,
            'is_cloud_backend': BACKENDS[backend]['is_cloud'],
            'base_url': self.base_url_line_edit.text(),
            'api_key': self.api_key_line_edit.text(),
            'model': self.model_combo_box.currentText(),
            'system_prompt': self.system_prompt_text_edit.toPlainText(),
            'prompt': self.prompt_text_edit.toPlainText(),
            'caption_start': self.caption_start_line_edit.text(),
            'caption_position': self.caption_position_combo_box.currentText(),
            'remove_tag_separators':
                self.remove_tag_separators_check_box.isChecked(),
            'generation_parameters': generation_parameters
        }


@Slot()
def restore_stdout_and_stderr():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


class AutoCaptioner(QDockWidget):
    caption_generated = Signal(QModelIndex, str, list)

    def __init__(self, image_list_model: ImageListModel,
                 image_list: ImageList):
        super().__init__()
        self.image_list_model = image_list_model
        self.image_list = image_list
        self.settings = get_settings()
        self.is_captioning = False
        self.captioning_thread = None
        # Whether the cloud captioning consent warning has already been
        # accepted this session. Reset only when the application restarts.
        self.cloud_captioning_consent_given = False
        # Whether the last block of text in the console text edit should be
        # replaced with the next block of text that is outputted.
        self.replace_last_console_text_edit_block = False

        # Each `QDockWidget` needs a unique object name for saving its state.
        self.setObjectName('auto_captioner')
        self.setWindowTitle('Auto-Captioner')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)

        self.start_cancel_button = TallPushButton('Start Auto-Captioning')
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat('%v / %m images captioned (%p%)')
        self.progress_bar.hide()
        self.console_text_edit = QPlainTextEdit()
        set_text_edit_height(self.console_text_edit, 4)
        self.console_text_edit.setReadOnly(True)
        self.console_text_edit.hide()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.start_cancel_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.console_text_edit)
        self.caption_settings_form = CaptionSettingsForm()
        layout.addLayout(self.caption_settings_form)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(container)
        self.setWidget(scroll_area)

        self.start_cancel_button.clicked.connect(
            self.start_or_cancel_captioning)

    @Slot()
    def start_or_cancel_captioning(self):
        if self.is_captioning:
            # Cancel captioning.
            self.captioning_thread.is_canceled = True
            self.start_cancel_button.setEnabled(False)
            self.start_cancel_button.setText('Canceling Auto-Captioning...')
        else:
            # Start captioning.
            self.generate_captions()

    def set_is_captioning(self, is_captioning: bool):
        self.is_captioning = is_captioning
        button_text = ('Cancel Auto-Captioning' if is_captioning
                       else 'Start Auto-Captioning')
        self.start_cancel_button.setText(button_text)

    @Slot(str)
    def update_console_text_edit(self, text: str):
        # '\x1b[A' is the ANSI escape sequence for moving the cursor up.
        if text == '\x1b[A':
            self.replace_last_console_text_edit_block = True
            return
        text = text.strip()
        if not text:
            return
        if self.console_text_edit.isHidden():
            self.console_text_edit.show()
        if self.replace_last_console_text_edit_block:
            self.replace_last_console_text_edit_block = False
            # Select and remove the last block of text.
            self.console_text_edit.moveCursor(QTextCursor.MoveOperation.End)
            self.console_text_edit.moveCursor(
                QTextCursor.MoveOperation.StartOfBlock,
                QTextCursor.MoveMode.KeepAnchor)
            self.console_text_edit.textCursor().removeSelectedText()
            # Delete the newline.
            self.console_text_edit.textCursor().deletePreviousChar()
        self.console_text_edit.appendPlainText(text)

    @Slot()
    def show_alert(self):
        if self.captioning_thread.is_canceled:
            return
        if self.captioning_thread.is_error:
            icon = QMessageBox.Icon.Critical
            text = ('An error occurred during captioning. See the '
                    'Auto-Captioner console for more information.')
        else:
            icon = QMessageBox.Icon.Information
            text = 'Captioning has finished.'
        alert = QMessageBox()
        alert.setIcon(icon)
        alert.setText(text)
        alert.exec()

    @Slot()
    def generate_captions(self):
        caption_settings = self.caption_settings_form.get_caption_settings()
        if (caption_settings['is_cloud_backend']
                and not self.cloud_captioning_consent_given):
            consent_dialog = CloudCaptioningConsentDialog()
            reply = consent_dialog.exec()
            if reply != QMessageBox.StandardButton.Ok:
                return
            self.cloud_captioning_consent_given = True
        selected_image_indices = self.image_list.get_selected_image_indices()
        selected_image_count = len(selected_image_indices)
        show_alert_when_finished = False
        if selected_image_count > 1:
            confirmation_dialog = CaptionMultipleImagesDialog(
                selected_image_count)
            reply = confirmation_dialog.exec()
            if reply != QMessageBox.StandardButton.Yes:
                return
            show_alert_when_finished = (confirmation_dialog
                                        .show_alert_check_box.isChecked())
        self.set_is_captioning(True)
        if caption_settings['caption_position'] != CaptionPosition.DO_NOT_ADD:
            self.image_list_model.add_to_undo_stack(
                action_name=f'Generate '
                            f'{pluralize("Caption", selected_image_count)}',
                should_ask_for_confirmation=selected_image_count > 1)
        if selected_image_count > 1:
            self.progress_bar.setRange(0, selected_image_count)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
        tag_separator = get_tag_separator()
        self.captioning_thread = CaptioningThread(
            self, self.image_list_model, selected_image_indices,
            caption_settings, tag_separator)
        self.captioning_thread.text_outputted.connect(
            self.update_console_text_edit)
        self.captioning_thread.clear_console_text_edit_requested.connect(
            self.console_text_edit.clear)
        self.captioning_thread.caption_generated.connect(
            self.caption_generated)
        self.captioning_thread.progress_bar_update_requested.connect(
            self.progress_bar.setValue)
        self.captioning_thread.finished.connect(
            lambda: self.set_is_captioning(False))
        self.captioning_thread.finished.connect(restore_stdout_and_stderr)
        self.captioning_thread.finished.connect(self.progress_bar.hide)
        self.captioning_thread.finished.connect(
            lambda: self.start_cancel_button.setEnabled(True))
        if show_alert_when_finished:
            self.captioning_thread.finished.connect(self.show_alert)
        # Redirect `stdout` and `stderr` so that the outputs are displayed in
        # the console text edit.
        sys.stdout = self.captioning_thread
        sys.stderr = self.captioning_thread
        self.captioning_thread.start()
