# Prompts of the command-line captioner

`run.bat <folder>` reads two files from this folder: `sysprompt.txt`, the system prompt, and `main_prompt.txt`, the prompt sent with every image. Both are yours and are ignored by git. When one is missing, the tracked sample of the same role is used instead: `sample_sysprompt.txt` and `sample_main_prompt.txt`, the MOSS-Image caption prompt with the three short-caption styles. `--sysprompt FILE` and `--mainprompt FILE` name other files.

The files may be UTF-8, with or without a BOM, or UTF-16 with a BOM (what Notepad calls "Unicode"). Any other encoding stops the run.

The main prompt may use the template variables of the auto-captioner: `{tags}`, the image's existing `.txt` text, whole; `{name}`, the file name without the extension; `{folder}`, the name of the folder the image is in. The command line adds `{style}`, the letter A, B or C drawn for the image from a hash of its file name with the shares 40, 35 and 25. Braces that are not one of these variables, such as the JSON schema in the prompt, are left as they are.

The model must answer with one JSON object that has `caption.long`, `caption.medium` and `caption.short` as non-empty strings. The run writes the object to `<stem>.json`, `caption.long` to `<stem>.txt` and `caption.short` to `<stem>.short`.
