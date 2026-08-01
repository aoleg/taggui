import sys
from pathlib import Path

from PySide6.QtCore import QSettings

SETTINGS_FILE_NAME = 'settings.ini'
# The previous (de)organization/application name used with the OS-native
# `QSettings` backend (Windows registry, macOS plist, or Linux INI file),
# kept only to migrate existing users to the new settings file.
OLD_NATIVE_SETTINGS_ORGANIZATION_AND_APPLICATION = 'taggui'

# Defaults for settings that are accessed from multiple places.
DEFAULT_SETTINGS = {
    'font_size': 16,
    # Common image formats that are supported in PySide6.
    'image_list_file_formats': 'bmp, gif, jpg, jpeg, png, tif, tiff, webp',
    'image_list_image_width': 200,
    'tag_separator': ',',
    'insert_space_after_tag_separator': True,
    'autocomplete_tags': True
}


def get_settings_file_path() -> Path:
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller-bundled executable. `sys._MEIPASS` is a
        # temporary extraction directory that does not persist across runs,
        # so the settings file is stored next to the executable instead.
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).parent.parent.parent
    return base_path / SETTINGS_FILE_NAME


def get_settings() -> QSettings:
    settings_file_path = get_settings_file_path()
    settings_file_exists = settings_file_path.is_file()
    settings = QSettings(str(settings_file_path), QSettings.Format.IniFormat)
    if not settings_file_exists:
        # Migrate settings from the previous OS-native storage the first
        # time the settings file is created.
        old_settings = QSettings(
            OLD_NATIVE_SETTINGS_ORGANIZATION_AND_APPLICATION,
            OLD_NATIVE_SETTINGS_ORGANIZATION_AND_APPLICATION)
        for key in old_settings.allKeys():
            settings.setValue(key, old_settings.value(key))
        settings.sync()
    return settings


def get_tag_separator() -> str:
    settings = get_settings()
    tag_separator = settings.value(
        'tag_separator', defaultValue=DEFAULT_SETTINGS['tag_separator'],
        type=str)
    insert_space_after_tag_separator = settings.value(
        'insert_space_after_tag_separator',
        defaultValue=DEFAULT_SETTINGS['insert_space_after_tag_separator'],
        type=bool)
    if insert_space_after_tag_separator:
        tag_separator += ' '
    return tag_separator
