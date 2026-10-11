from pathlib import Path
from typing import Callable

from PySide6.QtCore import QModelIndex, QSize, Qt, Slot
from PySide6.QtGui import QColor, QImage, QImageReader, QPainter, QPixmap, \
    QResizeEvent
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from models.proxy_image_list_model import ProxyImageListModel
from utils.image import Image

# How strongly the face mask is tinted over the image in the preview.
MASK_TINT_OPACITY = 0.55
MASK_TINT_COLOR = QColor(255, 0, 0)


def read_image(image_path: Path) -> QImage:
    image_reader = QImageReader(str(image_path))
    # Rotate the image according to the orientation tag.
    image_reader.setAutoTransform(True)
    return image_reader.read()


def tint_masked_area(image: QImage, mask_path: Path) -> QImage:
    """
    Paint the masked (black) area of a face mask red over the image, the way
    the face mask tool draws its previews.
    """
    mask = QImage(str(mask_path)).convertToFormat(
        QImage.Format.Format_Grayscale8)
    if mask.isNull():
        return image
    if mask.size() != image.size():
        mask = mask.scaled(image.size(), Qt.AspectRatioMode.IgnoreAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
    # Black in the mask is the masked area, so the alpha of the tint is the
    # inverted mask.
    mask.invertPixels()
    tint = QImage(image.size(), QImage.Format.Format_ARGB32)
    tint.fill(MASK_TINT_COLOR)
    tint.setAlphaChannel(mask)
    image = image.convertToFormat(QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    painter.setOpacity(MASK_TINT_OPACITY)
    painter.drawImage(0, 0, tint)
    painter.end()
    return image


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.mask_path = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        # This allows the label to shrink.
        self.setMinimumSize(QSize(1, 1))

    def resizeEvent(self, event: QResizeEvent):
        """Reload the image whenever the label is resized."""
        if self.image_path:
            self.load_image(self.image_path, self.mask_path)

    def load_image(self, image_path: Path, mask_path: Path | None = None):
        self.image_path = image_path
        self.mask_path = mask_path
        image = read_image(image_path)
        if mask_path is not None and not image.isNull():
            image = tint_masked_area(image, mask_path)
        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(self.devicePixelRatio())
        pixmap = pixmap.scaled(
            self.size() * pixmap.devicePixelRatio(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pixmap)


class ImageViewer(QWidget):
    def __init__(self, proxy_image_list_model: ProxyImageListModel):
        super().__init__()
        self.proxy_image_list_model = proxy_image_list_model
        self.image_label = ImageLabel()
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.hide()
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        # When the pipeline preview is shown, the viewer shows the dry-run
        # preview of the current image, or the image with its face mask
        # tinted, instead of the image itself. The resolver is set by the
        # main window and returns the preview path and the mask path, each
        # `None` when there is none.
        self.is_preview_shown = False
        self.preview_resolver: Callable[
            [Path], tuple[Path | None, Path | None]] | None = None
        self.current_image_path: Path | None = None

    @Slot()
    def load_image(self, proxy_image_index: QModelIndex):
        image: Image = self.proxy_image_list_model.data(
            proxy_image_index, Qt.ItemDataRole.UserRole)
        if image is None:
            return
        self.current_image_path = image.path
        self.show_current_image()

    def show_current_image(self):
        image_path = self.current_image_path
        if image_path is None:
            return
        if self.is_preview_shown and self.preview_resolver is not None:
            preview_path, mask_path = self.preview_resolver(image_path)
            if preview_path is not None:
                self.image_label.load_image(preview_path)
                # The kind of preview is the folder under `previews`.
                parts = preview_path.parts
                kind = (parts[parts.index('previews') + 1]
                        if 'previews' in parts else preview_path.parent.name)
                self.set_status(f'Preview: {kind}')
                return
            if mask_path is not None:
                self.image_label.load_image(image_path, mask_path)
                self.set_status('Preview: face mask (red is masked out)')
                return
            self.image_label.load_image(image_path)
            self.set_status('No preview for this image. Run a Preview in '
                            'the Dataset Tools dock.')
            return
        self.image_label.load_image(image_path)
        self.set_status(None)

    def set_status(self, text: str | None):
        if text:
            self.status_label.setText(text)
            self.status_label.show()
        else:
            self.status_label.hide()

    @Slot(bool)
    def set_preview_shown(self, is_shown: bool):
        self.is_preview_shown = is_shown
        self.show_current_image()

    @Slot()
    def reload_current_image(self):
        self.show_current_image()
