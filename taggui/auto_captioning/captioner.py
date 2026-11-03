"""
The part of captioning one image that the auto-captioner thread and the
command-line captioner share: send the image, retry at half size when the
endpoint rejects the request, give up after a few failures.
"""
import hashlib
from typing import Callable

from PIL import UnidentifiedImageError

from auto_captioning.captioning_model import (CaptioningModel,
                                              RequestFailedError)
from utils.image import Image

# When a request fails, the image is resized to half its width and height and
# sent again, up to this many times, before it is skipped.
MAX_RETRY_COUNT = 2
# Captioning stops when this many images in a row are skipped, because the
# endpoint itself is then the likely problem.
MAX_CONSECUTIVE_SKIPS = 3

# The styles of a short caption and their shares over a dataset, for the
# `{style}` template variable of the command-line captioner: a subject phrase
# with trailing tags (A), a tag list (B) and a casual sentence (C). The letter
# of an image comes from a hash of its file name, so a rerun gives the same
# letter and over a dataset the letters follow the shares. The same rule and
# shares as `recaption_json` in dataset_tools, so both tools agree on an
# image's style.
STYLE_SHARES = (('A', 40), ('B', 35), ('C', 25))


class CaptioningCanceled(Exception):
    """Raised by `caption_image()` when `should_cancel()` returns true."""


class ImageSkipped(Exception):
    """
    Raised by `caption_image()` when an image gets no caption. The message
    says why; `is_endpoint_failure` is true when the endpoint rejected the
    request every time, which counts towards stopping a run, and false when
    the image itself is missing or unreadable.
    """

    def __init__(self, reason: str, is_endpoint_failure: bool):
        super().__init__(reason)
        self.is_endpoint_failure = is_endpoint_failure


def style_for(stem: str,
              shares: tuple[tuple[str, int], ...] = STYLE_SHARES) -> str:
    total = sum(share for _, share in shares)
    digest = hashlib.sha1(stem.lower().encode('utf-8')).hexdigest()
    position = int(digest[:8], 16) % total
    cumulative = 0
    for letter, share in shares:
        cumulative += share
        if position < cumulative:
            return letter
    return shares[-1][0]


def caption_image(model: CaptioningModel, image: Image, image_prompt: str,
                  should_cancel: Callable[[], bool] = lambda: False) -> str:
    """
    Caption one image, retrying at half the width and height when the
    endpoint rejects the request. Raises `ImageSkipped` with the reason when
    the image gets no caption, and `CaptioningCanceled` when
    `should_cancel()` returns true between two attempts.
    """
    try:
        pil_image, is_source_jpeg = model.load_image(image)
    except FileNotFoundError:
        raise ImageSkipped('it was deleted or moved', is_endpoint_failure=False)
    except UnidentifiedImageError:
        raise ImageSkipped('its file format is not supported or it is a '
                           'corrupted image', is_endpoint_failure=False)
    for retry_count in range(MAX_RETRY_COUNT + 1):
        if retry_count > 0:
            if should_cancel():
                raise CaptioningCanceled()
            # The endpoint may have rejected the image because it is too
            # large, so try again at half the width and height.
            pil_image = pil_image.reduce(2)
            print(f'Retrying {image.path.name} at '
                  f'{pil_image.width}x{pil_image.height}.')
        model_inputs = model.get_model_inputs(image_prompt, pil_image,
                                              is_source_jpeg)
        try:
            caption, _ = model.generate_caption(model_inputs, image_prompt)
            return caption
        except RequestFailedError as exception:
            print(exception)
    raise ImageSkipped(f'captioning failed {MAX_RETRY_COUNT + 1} times',
                       is_endpoint_failure=True)
