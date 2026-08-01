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
        generated_text = generated_text.strip()
        if self.caption_start.strip() and not generated_text.startswith(
                self.caption_start):
            caption = f'{self.caption_start.strip()} {generated_text}'
        else:
            caption = generated_text
        caption = caption.strip()
        if self.remove_tag_separators:
            caption = caption.replace(self.thread.tag_separator, ' ')
        return caption

    def generate_caption(self, model_inputs: dict,
                         image_prompt: str) -> tuple[str, str]:
        payload = {
            'model': self.model,
            'messages': model_inputs['messages'],
            **self.generation_parameters
        }
        try:
            response = requests.post(f'{self.base_url}/v1/chat/completions',
                                     json=payload, headers=self.get_headers(),
                                     timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exception:
            raise RuntimeError(f'Request to {self.base_url} failed: '
                               f'{exception}')
        response_json = response.json()
        generated_text = response_json['choices'][0]['message']['content']
        caption = self.postprocess_generated_text(generated_text)
        console_output_caption = caption
        return caption, console_output_caption
