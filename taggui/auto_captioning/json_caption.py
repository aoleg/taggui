"""
The JSON caption of the command-line captioner: one object per image in the
MOSS-Image form (`caption.long`, `caption.medium`, `caption.short`, plus
`watermark` and `borders`), found in a model's answer, checked, and written
as three sidecars: `<stem>.json` (the whole object), `<stem>.txt`
(caption.long) and `<stem>.short` (caption.short, one line).
"""
import codecs
import json
import re
from pathlib import Path

CAPTION_FIELDS = ('long', 'medium', 'short')
JSON_SUFFIX = '.json'
LONG_SUFFIX = '.txt'
SHORT_SUFFIX = '.short'


def read_text_file(path: Path) -> str:
    """
    Read a prompt or caption file: UTF-16 with a BOM (what Notepad calls
    "Unicode") or UTF-8 with or without a BOM. Line ends become LF. Raises
    `ValueError` for any other encoding, so that a garbled prompt never
    reaches the model.
    """
    data = path.read_bytes()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = data.decode('utf-16')
    else:
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            raise ValueError(f'{path} is neither UTF-8 nor UTF-16 with a '
                             f'BOM. Save it as UTF-8.')
    if '\x00' in text:
        # UTF-16 without a BOM, read as UTF-8 by luck.
        raise ValueError(f'{path} holds null characters: it is UTF-16 '
                         f'without a BOM. Save it as UTF-8.')
    return text.replace('\r\n', '\n').replace('\r', '\n')


def extract_json_object(text: str) -> dict | None:
    """
    The first JSON object in a model's answer. Code fences and any text
    around the object are skipped by trying to decode from every `{`.
    """
    decoder = json.JSONDecoder()
    for match in re.finditer(r'{', text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def get_caption_json_error(caption_json) -> str | None:
    """
    Why an answer is not a usable caption object, or `None` when it is: it
    must have a `caption` object whose `long`, `medium` and `short` are
    non-empty strings. The other fields are kept as they are.
    """
    if not isinstance(caption_json, dict):
        return 'the answer holds no JSON object'
    caption = caption_json.get('caption')
    if not isinstance(caption, dict):
        return '"caption" is missing or is not an object'
    for field in CAPTION_FIELDS:
        value = caption.get(field)
        if not isinstance(value, str):
            return f'"caption.{field}" is missing or is not a string'
        if not value.strip():
            return f'"caption.{field}" is empty'
    return None


def one_line(text: str) -> str:
    return ' '.join(text.split())


def get_sidecar_paths(image_path: Path) -> tuple[Path, Path, Path]:
    """The `.json`, `.txt` and `.short` sidecars of an image."""
    return (image_path.with_suffix(JSON_SUFFIX),
            image_path.with_suffix(LONG_SUFFIX),
            image_path.with_suffix(SHORT_SUFFIX))


def write_caption_sidecars(image_path: Path, caption_json: dict):
    """
    Write the three sidecars of an image, the `.json` last: it marks the
    image as done, so a run that is killed between the writes leaves no
    `.json` and does the image again.
    """
    json_path, long_path, short_path = get_sidecar_paths(image_path)
    caption = caption_json['caption']
    long_text = caption['long'].strip().replace('\r\n', '\n')
    short_text = one_line(caption['short'])
    # `newline=''` keeps the LF line ends on Windows.
    with open(long_path, 'w', encoding='utf-8', newline='') as long_file:
        long_file.write(long_text)
    with open(short_path, 'w', encoding='utf-8', newline='') as short_file:
        short_file.write(short_text)
    with open(json_path, 'w', encoding='utf-8', newline='') as json_file:
        json_file.write(json.dumps(caption_json, ensure_ascii=False,
                                   indent=2))
        json_file.write('\n')
