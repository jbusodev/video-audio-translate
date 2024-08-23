# Audio & Video translation app

This is a script written in Python to translate an audio of video file into desired language.

## Requirements

- Python 3.10.14
- ffmpeg. Instructions on how to install: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- rubberband. from package if linux system or: [https://breakfastquay.com/rubberband/](https://breakfastquay.com/rubberband/) if Windows
- Deepl API Key (free): [https://www.deepl.com/en/your-account/keys](https://www.deepl.com/en/your-account/keys)

`py -m venv .venv`. Make sure to source it before going further.

```pip install -r requirements.txt```
`mv .env.example .env`. Enter your DEEPL_API key in `.env`.

## Usage

```py main.py input_file_path --lang lg [--merge]```

`input_file_path`: Absolute input file path. Accepts either audio of video file.
`--lang` or `-l`: Destination language. Accepts 2-letter language code such as `en`, `fr`, `es`, `de`, etc.
`--merge` or `-m` (Optional): Merges translated audio with original video. If absent, outputs translated audio file instead.

Audio output is in form: `input_filename_lg.wav` and Video output: `input_filename_lg.mp4` and saved in same directory as input file.

## Features

- Extracts audio if input file is video using ffmpeg.
- Transcribes audio using OpenAI Whisper.
- Translate transcript using Deepl API.
- Generate translation using coquiTTS.
- Matches translated with original audio and merges with video if input file is video.

## Possible improvements

### Performance & Quality

- [ ] Split and merge generations for use with long duration audio.
- [ ] Improve audio matching.
- [ ] Pass generated audio to RVC for better quality.
- [x] Support for bulk translation.

### Quality of Life & UX

- [ ] Add output file path parameter.
- [ ] Turn into Web App.
