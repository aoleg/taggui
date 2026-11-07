"""
Command-line captioning with the auto-captioner's request path and settings,
without the window.

    python taggui/run_cli.py <folder> [--sysprompt FILE] [--mainprompt FILE]
                                     [--url URL] [--model NAME] ...

Every image under the folder that has no `<stem>.json` sidecar is sent to the
endpoint of `settings.ini` with the prompts of the `prompts` folder. The
answer is one JSON object in the MOSS-Image form with `caption.long`,
`caption.medium` and `caption.short`; it is written as `<stem>.json`, with
`caption.long` as `<stem>.txt` and `caption.short` as `<stem>.short`. An
existing `.txt` or `.short` is copied to `<folder>_backup` first, once. A
run that stops is continued by starting it again: images with a `.json` are
skipped.
"""
import argparse
import csv
import hashlib
import json
import shutil
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

from auto_captioning.captioner import (MAX_CONSECUTIVE_SKIPS,
                                       STYLE_SHARES, ImageSkipped,
                                       caption_image, style_for)
from auto_captioning.captioning_model import (CaptioningModel,
                                              REQUEST_TIMEOUT_SECONDS,
                                              make_generation_parameters,
                                              replace_template_variables)
from auto_captioning.json_caption import (describe_json_error,
                                          extract_json_object,
                                          get_caption_json_error,
                                          get_sidecar_paths, one_line,
                                          read_text_file,
                                          write_caption_sidecars)
from models.image_list_model import get_file_paths, get_image_suffixes
from utils.dataset_tools import BACKUP_MARK, BACKUP_SUFFIX, get_taggui_root
from utils.image import Image
from utils.settings import DEFAULT_SETTINGS, get_settings, get_tag_separator
from utils.utils import pluralize

PROMPTS_DIRNAME = 'prompts'
SYSTEM_PROMPT_FILE_NAME = 'sysprompt.txt'
MAIN_PROMPT_FILE_NAME = 'main_prompt.txt'
SAMPLE_PREFIX = 'sample_'
# An answer that is not a caption object is asked for again, at the same
# image size, up to this many times in all before the image is skipped.
MAX_ANSWER_ATTEMPTS = 3
LOG_DIRNAME = '_caption'
LOG_FILE_NAME = 'log.csv'
LOG_FIELDS = ['time', 'path', 'status', 'style', 'attempts', 'reason',
              'seconds', 'model', 'prompts']
LOOPBACK_HOSTS = {'localhost', '127.0.0.1', '::1', '[::1]'}


@dataclass
class Run:
    folder: Path
    backup_directory: Path
    model_name: str = ''
    prompts_digest: str = ''
    # The images (relative POSIX paths) an earlier run captioned, from the
    # log: their `.txt` and `.short` are this tool's own output, not
    # originals, so a redo does not back them up.
    captioned_before: set[str] = field(default_factory=set)
    done: int = 0
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    retried: int = 0
    seconds: float = 0.0
    stopped_early: bool = False


def parse_arguments(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='run.bat',
        description='Caption every image under a folder that has no .json '
                    'sidecar, with the auto-captioner\'s endpoint and the '
                    'prompts of the prompts folder. Writes <stem>.json, '
                    '<stem>.txt (caption.long) and <stem>.short '
                    '(caption.short).')
    parser.add_argument('folder', type=Path,
                        help='the dataset folder, searched with its '
                             'subfolders')
    parser.add_argument('--sysprompt', type=Path, metavar='FILE',
                        help=f'the system prompt file (default '
                             f'{PROMPTS_DIRNAME}/{SYSTEM_PROMPT_FILE_NAME})')
    parser.add_argument('--mainprompt', type=Path, metavar='FILE',
                        help=f'the per-image prompt file (default '
                             f'{PROMPTS_DIRNAME}/{MAIN_PROMPT_FILE_NAME})')
    parser.add_argument('--url', help='the API base URL (default: the '
                                      'setting)')
    parser.add_argument('--api-key', help='the API key (default: the setting)')
    parser.add_argument('--model', help='the model name (default: the '
                                        'setting, else the endpoint\'s model)')
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--top-p', type=float)
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--repetition-penalty', type=float)
    reasoning = parser.add_mutually_exclusive_group()
    reasoning.add_argument('--reasoning', dest='allow_reasoning',
                           action='store_true', default=None,
                           help='let the model reason before it answers')
    reasoning.add_argument('--no-reasoning', dest='allow_reasoning',
                           action='store_false',
                           help='turn reasoning off')
    parser.add_argument('--timeout', type=float, metavar='SECONDS',
                        help=f'how long to wait for one answer (default '
                             f'{REQUEST_TIMEOUT_SECONDS})')
    parser.add_argument('--limit', type=int, metavar='N',
                        help='caption at most N images without a .json')
    parser.add_argument('--dry-run', action='store_true',
                        help='list the work and the first request; send '
                             'nothing, write nothing')
    return parser.parse_args(argv)


def setting(settings, key: str, override, type_):
    if override is not None:
        return override
    return settings.value(key, defaultValue=DEFAULT_SETTINGS[key], type=type_)


def resolve_prompt_file(given: Path | None, file_name: str) -> Path:
    """
    The prompt file to use: the given one, else the private file of the
    prompts folder, else the tracked sample of the same role.
    """
    if given is not None:
        return given
    prompts_directory = get_taggui_root() / PROMPTS_DIRNAME
    private = prompts_directory / file_name
    if private.is_file():
        return private
    return prompts_directory / (SAMPLE_PREFIX + file_name)


def load_prompt(path: Path, required: bool) -> str:
    if not path.is_file():
        if required:
            raise SystemExit(f'The prompt file {path} does not exist.')
        return ''
    try:
        text = read_text_file(path).strip()
    except ValueError as exception:
        raise SystemExit(str(exception))
    if required and not text:
        raise SystemExit(f'The prompt file {path} is empty.')
    return text


def is_loopback(url: str) -> bool:
    host = (urlsplit(url).hostname or '').lower()
    return host in LOOPBACK_HOSTS


def read_record(image_path: Path) -> str:
    """The existing `.txt` of an image, whole, for `{tags}`."""
    text_path = image_path.with_suffix('.txt')
    if not text_path.is_file():
        return ''
    # `errors='replace'` as the image list does when it reads a caption.
    return text_path.read_text(encoding='utf-8', errors='replace').strip()


def list_images(folder: Path, settings) -> tuple[list[Path], list[Path]]:
    """The images under the folder: those to caption and those that have a
    `.json` already, each sorted by path."""
    hide_underscore_folders = settings.value(
        'hide_underscore_folders',
        defaultValue=DEFAULT_SETTINGS['hide_underscore_folders'], type=bool)
    image_suffixes = get_image_suffixes(settings)
    image_paths = sorted(
        path for path in get_file_paths(folder, hide_underscore_folders)
        if path.suffix.lower() in image_suffixes)
    to_do = [path for path in image_paths if not has_json(path)]
    captioned = [path for path in image_paths if has_json(path)]
    return to_do, captioned


def has_json(image_path: Path) -> bool:
    return get_sidecar_paths(image_path)[0].is_file()


def files_to_back_up(image_path: Path) -> list[Path]:
    _, long_path, short_path = get_sidecar_paths(image_path)
    return [path for path in (long_path, short_path) if path.is_file()]


def ensure_backup_directory(run: Run):
    if run.backup_directory.is_dir():
        return
    run.backup_directory.mkdir(parents=True)
    (run.backup_directory / BACKUP_MARK).write_text(
        json.dumps({'made_by': 'taggui', 'kind': 'backup',
                    'dataset': run.folder.name}) + '\n', encoding='utf-8')


def get_log_path(run: Run) -> Path:
    return run.backup_directory / LOG_DIRNAME / LOG_FILE_NAME


def read_captioned_before(run: Run) -> set[str]:
    """The images the log says an earlier run captioned."""
    log_path = get_log_path(run)
    if not log_path.is_file():
        return set()
    with open(log_path, encoding='utf-8-sig', newline='') as log_file:
        return {row['path'] for row in csv.DictReader(log_file)
                if row.get('status') == 'ok'}


def back_up(run: Run, image_path: Path) -> list[Path]:
    """
    Copy the `.txt` and `.short` of an image that are about to be
    overwritten to `<folder>_backup/<same relative path>`. A backup that
    exists is kept: it is the untouched original. An image this tool
    captioned before (its `.json` was deleted to redo it) is not backed up:
    its files are this tool's output, and the original, if there was one, is
    in the backup folder already. Returns the paths copied.
    """
    copied = []
    if image_path.relative_to(run.folder).as_posix() in run.captioned_before:
        return copied
    for path in files_to_back_up(image_path):
        destination = run.backup_directory / path.relative_to(run.folder)
        if destination.is_file():
            continue
        ensure_backup_directory(run)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied.append(destination)
    return copied


def write_log_rows(run: Run, rows: list[dict]):
    ensure_backup_directory(run)
    log_path = get_log_path(run)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.is_file() or log_path.stat().st_size == 0
    # A BOM on a new file, so that Excel shows Cyrillic.
    with open(log_path, 'a', encoding='utf-8-sig' if is_new else 'utf-8',
              newline='') as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def log_row(run: Run, image_path: Path, status: str, style: str,
            attempts: int, reason: str, seconds: float) -> dict:
    return {'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'path': image_path.relative_to(run.folder).as_posix(),
            'status': status, 'style': style, 'attempts': attempts,
            'reason': reason, 'seconds': f'{seconds:.1f}',
            'model': run.model_name, 'prompts': run.prompts_digest}


def write_with_deferred_interrupt(write):
    """
    Run the sidecar writes with Ctrl+C held off until they are done, so that
    an image is written whole or not at all; the interrupt is raised after.
    """
    interrupted = []
    previous_handler = signal.signal(
        signal.SIGINT, lambda *_: interrupted.append(True))
    try:
        write()
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    if interrupted:
        raise KeyboardInterrupt


def caption_one(run: Run, model: CaptioningModel, image_path: Path,
                prompt: str, index: int, total: int) -> bool | None:
    """
    Caption one image and write its sidecars. Returns `True` when the image
    was done, `False` when the endpoint or its answers failed it, and `None`
    when the image itself is missing or unreadable (a skip that does not
    count towards stopping the run). A skip is recorded in `run.skipped`
    with its reason.
    """
    start_time = perf_counter()
    style = style_for(image_path.stem)
    record = read_record(image_path)
    image = Image(image_path, dimensions=None,
                  tags=[record] if record else [])
    image_prompt = replace_template_variables(prompt, image,
                                              {'style': style})
    relative_path = image_path.relative_to(run.folder).as_posix()
    reason = ''
    is_endpoint_failure = True
    caption_json = None
    attempts = 0
    for attempts in range(1, MAX_ANSWER_ATTEMPTS + 1):
        try:
            answer = caption_image(model, image, image_prompt)
        except ImageSkipped as skip:
            reason = str(skip)
            is_endpoint_failure = skip.is_endpoint_failure
            break
        caption_json = extract_json_object(answer)
        reason = get_caption_json_error(caption_json) or ''
        if not reason:
            break
        if caption_json is None:
            reason = f'{reason}: {describe_json_error(answer)}'
        caption_json = None
        reason = f'{reason}; answer: {" ".join(answer.split())[:200]}'
        if attempts < MAX_ANSWER_ATTEMPTS:
            run.retried += 1
            print(f'Asking again for {image_path.name}: {reason}')
    seconds = perf_counter() - start_time
    run.seconds += seconds
    if caption_json is None:
        run.skipped.append((image_path, reason))
        print(f'[{index:>{len(str(total))}}/{total}] {relative_path}: skipped, '
              f'{reason}')
        write_log_rows(run, [log_row(run, image_path, 'skipped', style,
                                     attempts, reason, seconds)])
        return False if is_endpoint_failure else None

    def write():
        copied = back_up(run, image_path)
        write_caption_sidecars(image_path, caption_json)
        for path in copied:
            print(f'    backed up {path.name} to {path.parent}')

    write_with_deferred_interrupt(write)
    run.done += 1
    run.captioned_before.add(relative_path)
    print(f'[{index:>{len(str(total))}}/{total}] {relative_path} '
          f'({seconds:.1f} s, {style}): '
          f'{one_line(caption_json["caption"]["short"])}')
    write_log_rows(run, [log_row(run, image_path, 'ok', style, attempts, '',
                                 seconds)])
    return True


def print_summary(run: Run, to_do_count: int):
    remaining = len(list_images(run.folder, get_settings())[0])
    if run.done:
        print(f'Captioned {run.done} {pluralize("image", run.done)} in '
              f'{run.seconds:.0f} s ({run.seconds / run.done:.1f} s/image)'
              f'{f", {run.retried} answers asked again" if run.retried else ""}'
              '.')
    if run.skipped:
        print(f'Skipped {len(run.skipped)} '
              f'{pluralize("image", len(run.skipped))}:')
        for path, reason in run.skipped:
            print(f'  {path.relative_to(run.folder).as_posix()}: {reason}')
    if run.stopped_early:
        print('The run stopped early.')
    if remaining:
        print(f'{remaining} {pluralize("image", remaining)} under {run.folder} '
              f'still {"has" if remaining == 1 else "have"} no .json. Run the '
              f'same command again to continue.')
    elif to_do_count:
        print(f'Every image under {run.folder} has a .json.')


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        # Captions hold any alphabet; the Windows console may not.
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    args = parse_arguments(argv)
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f'{folder} is not a folder.')
    if not folder.name:
        raise SystemExit('Give a folder, not a drive root: the backups go to '
                         'a folder next to it.')
    settings = get_settings()

    system_prompt_path = resolve_prompt_file(args.sysprompt,
                                             SYSTEM_PROMPT_FILE_NAME)
    main_prompt_path = resolve_prompt_file(args.mainprompt,
                                           MAIN_PROMPT_FILE_NAME)
    system_prompt = load_prompt(system_prompt_path, required=False)
    main_prompt = load_prompt(main_prompt_path, required=True)
    prompts_digest = hashlib.sha256(
        (system_prompt + '\n' + main_prompt).encode('utf-8')).hexdigest()[:12]

    base_url = setting(settings, 'api_base_url', args.url, str)
    allow_reasoning = setting(settings, 'allow_reasoning',
                              args.allow_reasoning, bool)
    max_tokens = setting(settings, 'max_tokens', args.max_tokens, int)
    caption_settings = {
        'base_url': base_url,
        'api_key': setting(settings, 'api_key', args.api_key, str),
        'model': setting(settings, 'api_model', args.model, str),
        'system_prompt': system_prompt,
        'prompt': main_prompt,
        'caption_start': '',
        'remove_tag_separators': False,
        'request_timeout': args.timeout or REQUEST_TIMEOUT_SECONDS,
        'generation_parameters': make_generation_parameters(
            max_tokens=max_tokens,
            temperature=setting(settings, 'temperature', args.temperature,
                                float),
            top_p=setting(settings, 'top_p', args.top_p, float),
            top_k=setting(settings, 'top_k', args.top_k, int),
            repetition_penalty=setting(settings, 'repetition_penalty',
                                       args.repetition_penalty, float),
            allow_reasoning=allow_reasoning)
    }
    model = CaptioningModel(caption_settings, get_tag_separator())
    error_message = model.get_error_message()
    if error_message:
        raise SystemExit(error_message)

    to_do, captioned = list_images(folder, settings)
    if args.limit is not None:
        to_do = to_do[:max(args.limit, 0)]
    run = Run(folder=folder,
              backup_directory=folder.with_name(folder.name + BACKUP_SUFFIX),
              prompts_digest=prompts_digest)
    run.captioned_before = read_captioned_before(run)
    print(f'{len(to_do) + len(captioned)} images under {folder}, '
          f'{len(captioned)} with a .json already, {len(to_do)} to do'
          f'{f" (limited to {args.limit})" if args.limit is not None else ""}'
          '.')
    print(f'Endpoint {base_url}, reasoning '
          f'{"on" if allow_reasoning else "off"}, {max_tokens} max tokens, '
          f'{caption_settings["request_timeout"]:.0f} s timeout.')
    print(f'System prompt: {system_prompt_path if system_prompt else "none"}')
    print(f'Main prompt: {main_prompt_path} (prompts {prompts_digest})')
    print(f'Short caption styles: '
          f'{", ".join(f"{letter} {share}%" for letter, share in STYLE_SHARES)}'
          '.')
    if not is_loopback(base_url):
        print(f'The images are sent to {urlsplit(base_url).hostname}, which '
              f'is not this computer.')
    if not to_do:
        print('Nothing to do.')
        return 0

    if args.dry_run:
        # Two images of one stem share their sidecars: list each file once.
        to_back_up = list(dict.fromkeys(
            path for image_path in to_do
            if image_path.relative_to(folder).as_posix()
            not in run.captioned_before
            for path in files_to_back_up(image_path)
            if not (run.backup_directory
                    / path.relative_to(folder)).is_file()))
        print(f'{len(to_back_up)} {pluralize("file", len(to_back_up))} would '
              f'be backed up to {run.backup_directory}.')
        for path in to_back_up[:20]:
            print(f'  {path.relative_to(folder).as_posix()}')
        if len(to_back_up) > 20:
            print(f'  ... and {len(to_back_up) - 20} more')
        first = to_do[0]
        record = read_record(first)
        image = Image(first, dimensions=None, tags=[record] if record else [])
        print(f'\nThe first request, for {first.relative_to(folder).as_posix()}'
              f' (style {style_for(first.stem)}):\n')
        print(replace_template_variables(main_prompt, image,
                                         {'style': style_for(first.stem)}))
        return 0

    try:
        run.model_name = model.load_processor_and_model()
    except RuntimeError as exception:
        raise SystemExit(str(exception))
    consecutive_skips = 0
    try:
        for index, image_path in enumerate(to_do, 1):
            if has_json(image_path):
                # Another image of the same stem was captioned before it.
                print(f'[{index}/{len(to_do)}] '
                      f'{image_path.relative_to(folder).as_posix()}: a .json '
                      f'exists now, skipped')
                continue
            result = caption_one(run, model, image_path, main_prompt, index,
                                 len(to_do))
            if result:
                consecutive_skips = 0
            elif result is False:
                consecutive_skips += 1
                if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                    print(f'Stopped because {MAX_CONSECUTIVE_SKIPS} images in '
                          f'a row were skipped: the endpoint or the prompt is '
                          f'the likely problem.')
                    run.stopped_early = True
                    break
    except KeyboardInterrupt:
        print('\nInterrupted.')
        run.stopped_early = True
    print_summary(run, len(to_do))
    return 0 if not run.skipped and not run.stopped_early else 1


if __name__ == '__main__':
    sys.exit(main())
