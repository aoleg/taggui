from PySide6.QtWidgets import QCheckBox, QMessageBox


class CloudCaptioningConsentDialog(QMessageBox):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cloud Captioning Warning')
        self.setIcon(QMessageBox.Icon.Warning)
        self.setText(
            'The selected images will be uploaded to a third-party cloud '
            'server for processing.\n\n'
            'Do not proceed if your images are private or sensitive and you '
            'do not want them leaving your computer.')
        self.setStandardButtons(QMessageBox.StandardButton.Ok
                                | QMessageBox.StandardButton.Cancel)
        self.setDefaultButton(QMessageBox.StandardButton.Cancel)
        self.consent_check_box = QCheckBox(
            'I consent to sending my images to a cloud server')
        self.setCheckBox(self.consent_check_box)
        self.ok_button = self.button(QMessageBox.StandardButton.Ok)
        self.ok_button.setEnabled(False)
        self.consent_check_box.stateChanged.connect(
            lambda: self.ok_button.setEnabled(
                self.consent_check_box.isChecked()))
