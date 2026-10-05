import sys
from pathlib import Path

from PySide6.QtCore import QFile
from PySide6.QtWidgets import QMessageBox


def move_to_trash(path: Path) -> bool:
    """
    Move a file to the trash (the Recycle Bin on Windows) and return whether it
    succeeded.
    """
    if sys.platform != 'win32':
        return QFile(str(path)).moveToTrash()
    # `QFile.moveToTrash()` fails on Windows with "Unspecified error", so use
    # the shell function directly.
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [('hwnd', wintypes.HWND),
                    ('wFunc', wintypes.UINT),
                    ('pFrom', wintypes.LPCWSTR),
                    ('pTo', wintypes.LPCWSTR),
                    ('fFlags', ctypes.c_uint16),
                    ('fAnyOperationsAborted', wintypes.BOOL),
                    ('hNameMappings', ctypes.c_void_p),
                    ('lpszProgressTitle', wintypes.LPCWSTR)]

    FO_DELETE = 0x3
    FOF_SILENT = 0x4
    FOF_NOCONFIRMATION = 0x10
    FOF_ALLOWUNDO = 0x40
    FOF_NOERRORUI = 0x400
    operation = SHFILEOPSTRUCTW()
    operation.wFunc = FO_DELETE
    # The list of paths must end with two null characters. The second one is
    # added by `ctypes`.
    operation.pFrom = str(Path(path).resolve()) + '\0'
    operation.fFlags = (FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT
                        | FOF_NOERRORUI)
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    return (result == 0 and not operation.fAnyOperationsAborted
            and not Path(path).exists())


def get_resource_path(unbundled_resource_path: Path) -> Path:
    """
    Get the path to a resource, ensuring that it is valid even when the program
    is bundled with PyInstaller.
    """
    # PyInstaller stores the path to its temporary directory in `sys._MEIPASS`.
    base_path = getattr(sys, '_MEIPASS', Path(__file__).parent.parent.parent)
    resource_path = (Path(base_path) / unbundled_resource_path).resolve()
    return resource_path


def pluralize(word: str, count: int) -> str:
    if count == 1:
        return word
    return f'{word}s'


def list_with_and(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f'{items[0]} and {items[1]}'
    return ', '.join(items[:-1]) + f', and {items[-1]}'


class ConfirmationDialog(QMessageBox):
    def __init__(self, title: str, question: str):
        super().__init__()
        self.setWindowTitle(title)
        self.setIcon(QMessageBox.Icon.Question)
        self.setText(question)
        self.setStandardButtons(QMessageBox.StandardButton.Yes
                                | QMessageBox.StandardButton.Cancel)
        self.setDefaultButton(QMessageBox.StandardButton.Yes)


def get_confirmation_dialog_reply(title: str, question: str) -> int:
    """Display a confirmation dialog and return the user's reply."""
    confirmation_dialog = ConfirmationDialog(title, question)
    return confirmation_dialog.exec()
