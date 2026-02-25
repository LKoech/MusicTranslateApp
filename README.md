# MusicTranslateApp

A Gradio app that listens to spoken Zulu or Xhosa, transcribes it with Whisper, and is structured to translate the result into English.

## What it does
- Records audio from your microphone in a web UI.
- Runs speech-to-text with `openai/whisper-small` using Hugging Face Transformers.
- Lets you choose source language: `Zulu` or `Xhosa`.
- Returns recognized text live in the interface.

## Current status
- Transcription path is active and working.
- GMI-based translation integration exists in code (`translate_text_gmi`) but is currently commented out in the UI flow.

## Tech stack
- Python
- Gradio
- PyTorch
- Transformers (Whisper)
- Librosa + NumPy
- Requests

## Requirements
- Python 3.10+ recommended
- Optional NVIDIA GPU (falls back to CPU automatically)
- Internet access to download model weights on first run

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Optional environment variable
If you want to enable translation through the GMI API, set:

```bash
GMI_API_KEY=your_api_key_here
```

PowerShell example:

```powershell
$env:GMI_API_KEY="your_api_key_here"
```

## Run

```bash
python translator.py
```

Then open the Gradio URL shown in the terminal.

## How to use
1. Select `Zulu` or `Xhosa` in the dropdown.
2. Speak into your microphone.
3. Read the output text in the result box.

## Notes
- First launch can take a while because Whisper model files are downloaded.
- CPU inference can be slow; GPU significantly improves responsiveness.
- The output label currently says "English Translation", but in the current code path it returns transcribed text.

## Project files
- `translator.py`: App logic, Whisper pipeline, and Gradio UI.
- `requirements.txt`: Python dependencies.

## Roadmap ideas
- Re-enable translation in `voice_to_text`.
- Add error handling and user-facing status messages.
- Add language auto-detection and confidence display.
- Add tests for preprocessing and inference flow.
