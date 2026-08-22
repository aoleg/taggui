import base64
import re
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING

import requests
from PIL import Image as PilImage
from PIL.ImageOps import exif_transpose

from utils.image import Image

if TYPE_CHECKING:
    import auto_captioning.captioning_thread as captioning_thread

REQUEST_TIMEOUT_SECONDS = 300

# Reasoning models emit their chain of thought inside a dedicated block.
# Endpoints normally return it separately as `reasoning_content`, but it ends
# up inline in `content` when the block is left unterminated, such as when the
# token limit cuts it short, or when the model's tokenizer configuration does
# not mark its end-of-turn token as an end-of-generation token.

# Reasoning blocks that are wrapped in a pair of delimiters, as
# (opening, closing) pairs. The first pair is the Gemma 4 channel format and
# the rest are the formats that most other reasoning models share.
THINKING_DELIMITERS = (('<|channel>', '<channel|>'), ('<think>', '</think>'),
                       ('<thinking>', '</thinking>'),
                       ('<reasoning>', '</reasoning>'))
# The Harmony format used by gpt-oss models splits the response into channels
# instead of wrapping the reasoning in a pair of delimiters. The response is
# the final channel and everything before it is reasoning.
HARMONY_FINAL_CHANNEL = '<|channel|>final<|message|>'
HARMONY_CHANNEL = '<|channel|>'
# Control tokens that a model with an incorrect tokenizer configuration emits
# as literal text instead of as end-of-generation markers.
RESIDUAL_CONTROL_TOKEN = re.compile(r'<\|[\w-]+\|>|<\|[\w-]+>|<[\w-]+\|>'
                                    r'|</?(?:start|end)_of_turn>')
# End-of-turn tokens to stop generation on, for the same reason. They all mark
# the end of a response, so stopping on them cannot cut off a caption. The
# OpenAI API allows at most four stop sequences.
STOP_SEQUENCES = ['<|return|>', '<|im_end|>', '<end_of_turn>', '<|eot_id|>']


def strip_thinking(text: str) -> str:
    """
    Remove any chain-of-thought block or leftover control token that was
    returned inline in the generated text.
    """
    if HARMONY_FINAL_CHANNEL in text:
        # Keep only the final channel, which holds the actual response.
        text = text.split(HARMONY_FINAL_CHANNEL)[-1]
    elif HARMONY_CHANNEL in text:
        # Generation stopped before the final channel was reached, so the text
        # is reasoning only and there is no response to keep.
        text = ''
    for opening, closing in THINKING_DELIMITERS:
        if opening not in text:
            continue
        # This mirrors the `strip_thinking` macro in the Gemma 4 chat
        # template: only the text outside of the delimiter pairs is kept. A
        # block that was truncated before its closing delimiter therefore
        # removes everything that follows it.
        text = ''.join(part.split(opening)[0] if opening in part else part
                       for part in text.split(closing))
    return RESIDUAL_CONTROL_TOKEN.sub('', text)


def replace_template_variable(match: re.Match, image: Image) -> str:
    template_variable = match.group(0)[1:-1].lower()
    if template_variable == 'tags':
        return ', '.join(image.tags)
    if template_variable == 'name':
        return image.path.stem
    if template_variable in ('directory', 'folder'):
        return image.path.parent.name


def replace_template_variables(text: str, image: Image) -> str:
    # Replace template variables inside curly braces that are not escaped.
    text = re.sub(r'(?<!\\){[^{}]+(?<!\\)}',
                  lambda match: replace_template_variable(match, image), text)
    # Unescape escaped curly braces.
    text = re.sub(r'\\([{}])', r'\1', text)
    return text


def encode_image_as_data_url(pil_image: PilImage.Image) -> str:
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{encoded}'


class CaptioningModel:
    """Generates captions by calling an OpenAI-compatible chat completions
    endpoint (llama.cpp server, koboldcpp, LM Studio, or a cloud API)."""

    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        self.thread = captioning_thread_
        self.caption_settings = caption_settings
        self.base_url = caption_settings['base_url'].strip().rstrip('/')
        self.api_key = caption_settings.get('api_key', '').strip()
        self.model = caption_settings.get('model', '').strip() or 'local-model'
        self.system_prompt = caption_settings['system_prompt']
        self.prompt = caption_settings['prompt']
        self.caption_start = caption_settings['caption_start']
        self.remove_tag_separators = caption_settings['remove_tag_separators']
        self.generation_parameters = caption_settings['generation_parameters']

    def get_headers(self) -> dict:
        if self.api_key:
            return {'Authorization': f'Bearer {self.api_key}'}
        return {}

    def get_error_message(self) -> str | None:
        if not self.base_url:
            return 'The API base URL is not set.'
        return None

    def load_processor_and_model(self):
        # Verify that the endpoint is reachable and report the loaded model.
        try:
            response = requests.get(f'{self.base_url}/v1/models',
                                    headers=self.get_headers(), timeout=10)
            response.raise_for_status()
            models = response.json().get('data', [])
            model_name = models[0]['id'] if models else 'unknown'
            print(f'Connected to {self.base_url} '
                  f'(model: {self.model or model_name}).')
        except requests.RequestException as exception:
            raise RuntimeError(
                f'Failed to connect to {self.base_url}: {exception}')

    def monkey_patch_after_loading(self):
        return

    @staticmethod
    def get_captioning_start_datetime_string(
            captioning_start_datetime: datetime) -> str:
        return captioning_start_datetime.strftime('%Y-%m-%d %H:%M:%S')

    def get_captioning_message(self, are_multiple_images_selected: bool,
                               captioning_start_datetime: datetime) -> str:
        if are_multiple_images_selected:
            captioning_start_datetime_string = (
                self.get_captioning_start_datetime_string(
                    captioning_start_datetime))
            return (f'Captioning... (endpoint: {self.base_url}, start time: '
                    f'{captioning_start_datetime_string})')
        return f'Captioning... (endpoint: {self.base_url})'

    def get_image_prompt(self, image: Image) -> str:
        return replace_template_variables(self.prompt, image)

    @staticmethod
    def load_image(image: Image) -> PilImage.Image:
        pil_image = PilImage.open(image.path)
        # Rotate the image according to the orientation tag.
        pil_image = exif_transpose(pil_image)
        pil_image = pil_image.convert('RGB')
        return pil_image

    def get_model_inputs(self, image_prompt: str, image: Image) -> dict:
        pil_image = self.load_image(image)
        image_data_url = encode_image_as_data_url(pil_image)
        messages = []
        if self.system_prompt.strip():
            messages.append({'role': 'system', 'content': self.system_prompt})
        user_content = []
        if image_prompt.strip():
            user_content.append({'type': 'text', 'text': image_prompt})
        user_content.append(
            {'type': 'image_url', 'image_url': {'url': image_data_url}})
        messages.append({'role': 'user', 'content': user_content})
        return {'messages': messages}

    def postprocess_generated_text(self, generated_text: str) -> str:
        generated_text = strip_thinking(generated_text).strip()
        if self.caption_start.strip() and not generated_text.startswith(
                self.caption_start):
            caption = f'{self.caption_start.strip()} {generated_text}'
        else:
            caption = generated_text
        caption = caption.strip()
        if self.remove_tag_separators:
            caption = caption.replace(self.thread.tag_separator, ' ')
        return caption

    def send_chat_completions_request(self,
                                      payload: dict) -> requests.Response:
        try:
            response = requests.post(f'{self.base_url}/v1/chat/completions',
                                     json=payload, headers=self.get_headers(),
                                     timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exception:
            error_response = exception.response
            is_rejected_reasoning_effort = (
                'reasoning_effort' in payload
                and error_response is not None
                and error_response.status_code == 400
                and 'reasoning_effort' in error_response.text)
            if not is_rejected_reasoning_effort:
                raise RuntimeError(f'Request to {self.base_url} failed: '
                                   f'{exception}')
            # Some endpoints reject the parameter instead of ignoring it, so
            # retry without it to still get a caption. It is also removed from
            # the generation parameters so that the retry happens once instead
            # of for every image.
            print('The endpoint rejected the `reasoning_effort` parameter, so '
                  'reasoning cannot be turned off for this model. Retrying '
                  'without it.')
            del payload['reasoning_effort']
            self.generation_parameters.pop('reasoning_effort', None)
            return self.send_chat_completions_request(payload)
        return response

    def generate_caption(self, model_inputs: dict,
                         image_prompt: str) -> tuple[str, str]:
        payload = {
            'model': self.model,
            'messages': model_inputs['messages'],
            'stop': STOP_SEQUENCES,
            **self.generation_parameters
        }
        response = self.send_chat_completions_request(payload)
        response_json = response.json()
        choice = response_json['choices'][0]
        generated_text = choice['message'].get('content') or ''
        caption = self.postprocess_generated_text(generated_text)
        if not caption and choice.get('finish_reason') == 'length':
            print('The model reached the token limit while reasoning, before '
                  'it generated a caption. Turn off "Allow reasoning" or '
                  'increase "Maximum tokens".')
        console_output_caption = caption
        return caption, console_output_caption
