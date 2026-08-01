# TagGUI

<img src='images/icon.png' alt='TagGUI icon' width='128'>

Cross-platform desktop application for quickly adding and editing image tags
and captions, aimed towards creators of image datasets for generative AI
models.

<img src='images/screenshot-v1.25.0.png' alt='TagGUI screenshot' width='100%'>

## Features

- Keyboard-friendly interface for fast tagging
- Tag autocomplete based on your own most-used tags
- Automatic caption generation via any OpenAI-compatible vision API
  (llama.cpp server, koboldcpp, etc.)
- Batch tag operations for renaming, deleting, and sorting tags
- Advanced image list filtering

## Installation

The easiest way to use the application is to download the latest release from
the [releases page](https://www.github.com/jhc13/taggui/releases).
Choose the appropriate file for your operating system, extract it wherever you
want, and run the executable file inside.
You may have to install [7-Zip](https://www.7-zip.org/download.html) to
extract the files if you don't have it on your system.

- macOS users: There is no macOS release because it requires a device running
  the OS, and I do not have one. You can still install and run the program
  manually (see below).
- Linux users: You may need to install `libxcb-cursor0`.
  (See [this Stack Overflow answer](https://stackoverflow.com/a/75941575).)

Alternatively, you can install manually by cloning this repository and
installing the dependencies in `requirements.txt`.
Run `taggui/run_gui.py` to start the program.
Python 3.12 is recommended, but Python 3.11 should also work.

## Usage

Load the directory containing your images by clicking the `Load Directory`
button in the center of the window (or `File` -> `Load Directory`).
Tags are loaded from `.txt` files in the directory with the same names as the
images.
Any changes you make to the tags are also automatically saved to these `.txt`
files.

## Automatic Captioning

<img src='images/auto-captioner-v1.18.0.png' alt='Auto-captioner screenshot' width='100%'>

In addition to manual tagging, you can automatically generate captions for
your images inside TagGUI by sending them to any OpenAI-compatible vision
API. Four backends are supported:

- [llama.cpp server](https://github.com/ggml-org/llama.cpp) (local, single
  model)
- [koboldcpp](https://github.com/LostRuins/koboldcpp) (local, single model)
- [LM Studio](https://lmstudio.ai/) (local, can serve multiple downloaded
  models)
- Cloud (OpenAI-compatible) — any hosted API that speaks the OpenAI chat
  completions format (OpenAI itself, OpenRouter, etc.)

To use the feature, select a `Backend` in the Auto-Captioner pane, set the
`API base URL` (and, for a cloud backend, an `API key`), then click
`Connect` to verify the server is reachable. Select the images you want to
caption in the image list and click `Start Auto-Captioning`.
You can select multiple images to batch generate captions for all of them.

### Captioning parameters

`Backend`/`API base URL`: The address of your captioning server's
OpenAI-compatible API (for example `http://localhost:8080` for llama.cpp
server, `http://localhost:5001` for koboldcpp, `http://localhost:1234` for
LM Studio, or your provider's endpoint for a cloud backend).

`API key`: Only shown for the Cloud backend. Sent as an `Authorization:
Bearer` header; leave it blank if your endpoint doesn't require one. It is
saved in plain text in `settings.ini` alongside the other settings.

`Model`: Only shown for backends that can serve more than one model (LM
Studio and Cloud). llama.cpp server and koboldcpp only ever have one model
loaded, so there is nothing to select for those. Click `Connect` to
populate the dropdown with the models currently available on the server.

Selecting the Cloud backend and starting auto-captioning shows a one-time
warning per session that your images will be uploaded to a third-party
server; you must check the consent box to proceed.

`System prompt`: Instructions given to the model that apply to every image,
independent of the per-image prompt below.

`Prompt`: Instructions given to the captioning model for each image.
You can use the following template variables to dynamically insert information
about each image into the prompt:

- `{tags}`: The tags of the image, separated by commas.
- `{name}`: The file name of the image without the extension.
- `{directory}` or `{folder}`: The name of the directory containing the image.

An example prompt using a template variable could be
`Describe the image using the following tags as context: {tags}`.
With this prompt, `{tags}` would be replaced with the existing tags of each
image before the prompt is sent to the model.

`Start caption with`: Generated captions will start with this text. If the
model's response does not already start with it, it is prepended to the
response.

`Remove tag separators in caption`: If checked, tag separators (commas by
default) will be removed from the generated captions.

The advanced settings (`Maximum tokens`, `Temperature`, `Top-p`, `Top-k`,
`Repetition penalty`) are passed directly to the API and control the
model's sampling behavior.

## Advanced Image List Filtering

The basic functionality of filtering for images that contain a certain tag is
available by clicking on the tag in the `All Tags` pane.
In addition to this, you can construct more complex filters in
the `Filter Images` box at the top of the `Images` pane.

<details>
<summary>
Click here to see the full documentation for the filter syntax.
</summary>

### Filter criteria

These are the prefixes you can use to specify the filter criteria you want to
apply:

- `tag:`: Images that have the filter term as a tag
    - `tag:cat` will match images with the tag `cat`.
- `caption`: Images that contain the filter term in the caption
    - The caption is the list of tags as a single string, as it appears in the
      `.txt` file.
    - `caption:cat` will match images that have `cat` anywhere in the
      caption. For example, images with the tag `orange cat` or the
      tag `catastrophe`.
- `name`: Images that contain the filter term in the file name
    - `name:cat` will match images such as `cat-1.jpg` or `large_cat.png`.
- `path`: Images that contain the filter term in the full file path
    - `path:cat` will match images such as `C:\Users\cats\dog.jpg` or
      `/home/dogs/cat.jpg`.
- You can also use a filter term with no prefix to filter for images that
  contain the term in either the caption or the file path.
    - `cat` will match images containing `cat` in the caption or file path.

The following are prefixes for numeric filters. The operators `=` (`==` also
works), `!=`, `<`, `>`, `<=`, and `>=` are used to specify the type of
comparison.

- `tags`: Images that have the specified number of tags
    - `tags:=13` will match images that have exactly 13 tags.
    - `tags:!=7` will match images that do not have exactly 7 tags (images with
      less than 7 tags or more than 7 tags).
- `chars`: Images that have the specified number of characters in the caption
    - `chars:<100` will match images that have less than 100 characters in the
      caption.
    - `chars:>=30` will match images that have 30 or more characters in the
      caption.

### Spaces and quotes

If the filter term contains spaces, you must enclose it in quotes (either
single or double quotes).
For example, to find images with the tag `orange cat`, you must
use `tag:"orange cat"` or `tag:'orange cat'`.
If you have both spaces and quotes in the filter term, you can escape the
quotes with backslashes.
For example, you can use `tag:"orange \"cat\""` for the tag `orange "cat"`.
An alternative is to use different types of quotes for the outer and inner
quotes, like so: `tag:'orange "cat"'`.

### Wildcards

You can use the `*` character as a wildcard to match any number of any
characters, and the `?` character to match any single character.
For example, `tag:*cat` will match images with tags like `orange cat`,
`large cat`, and `cat`.

### Combining filters

Logical operators can be used to combine multiple filters:

- `NOT`: Images that do not match the filter
    - `NOT tag:cat` will match images that do not have the tag `cat`.
- `AND`: Images that match both filters before and after the operator
    - `tag:cat AND tag:orange` will match images that have both the tag `cat`
      and the tag `orange`.
- `OR`: Images that match either filter before or after the operator
    - `tag:cat OR tag:dog` will match images that have either the tag `cat` or
      the tag `dog`, or both.

The lowercase versions of these operators will also work: `not`, `and`,
and `or`.

The operator precedence is `NOT` > `AND` > `OR`, so by default, `NOT` will be
evaluated first, then `AND`, then `OR`.
You can use parentheses to change this order.
For example, in `tag:cat AND (tag:orange OR tag:white)`, the `OR` will be
evaluated first, matching images that have the tag `cat` and either the
tag `orange` or the tag `white`.
You can nest parentheses and operators to create arbitrarily complex filters.
</details>

## Controls

- ⭐ Previous / next image: `Ctrl`+`Up` / `Down` (just `Up` / `Down` also works
  in some cases)
- Jump to the first untagged image: `Ctrl`+`J`
- Focus the `Filter Images` box: `Alt`+`F`
- Focus the `Add Tag` box: `Alt`+`A`
- Focus the `Image Tags` list: `Alt`+`I`
- Focus the `Search Tags` box: `Alt`+`S`
- Focus the `Start Auto-Captioning` button: `Alt`+`C`

### Images pane

- First / last image: `Home` / `End`
- Select multiple images: Hold `Ctrl` or `Shift` and click the images
- Select all images: `Ctrl`+`A`
- Invert selection: `Ctrl`+`I`
- Right-clicking on an image will bring up the context menu, which includes
  actions such as copying and pasting tags and moving or copying selected
  images to another directory.

### Image Tags pane

- Add a tag: Type the tag into the `Add Tag` box and press `Enter`
- ⭐ Add the first tag suggested by autocomplete: `Ctrl`+`Enter`
- Add a tag to multiple images: Select the images in the image list add
  the tag
- Delete a tag: Select the tag and press `Delete`
- Rename a tag: Double-click the tag, or select the tag and press `F2`
- Reorder tags: Drag and drop the tags
- Select multiple tags: Hold `Ctrl` or `Shift` and click the tags

### All Tags pane

- Show all images containing a tag: Select the tag (When `Tag click action` is
  set to `Filter images for tag`)
- Add a tag to selected images: Click the tag (When `Tag click action` is set
  to `Add tag to selected images`)
- Delete all instances of a tag: Select the tag and press `Delete`
- Rename all instances of a tag: Double-click the tag, or select the tag and
  press `F2`

The `Edit` menu contains additional features for batch tag operations, such as
`Find and Replace` (`Ctrl`+`R`) and `Batch Reorder Tags` (`Ctrl`+`B`).
