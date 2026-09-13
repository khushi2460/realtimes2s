# Real-Time Speech-to-Speech Translation

A modular, fully open-source pipeline that listens to speech through a microphone, transcribes it, translates it into a language of your choice, and (optionally) speaks the translation back out loud — all running locally, with no paid API required.

```
Mic → Voice Activity Detection (chunking) → ASR (Whisper) → MT (NLLB-200) → TTS (Piper) → Speaker
```

## Why this project

Most publicly available "real-time speech-to-speech translator" repos fall into one of two camps:

- **API wrappers** — they call OpenAI or Google Cloud for every stage. This works, but it costs money per request, needs an internet connection and API keys, and gives you no actual NLP model to inspect, tune, or evaluate.
- **Heavy end-to-end research models** (e.g. Meta's SeamlessM4T) — state-of-the-art, but hard to run, and not modular — you can't easily swap out just the translation stage to compare models.

This project sits in between: every stage (ASR, MT, TTS) uses a free, open-source model that runs entirely on your own machine, and each stage is isolated in its own class so it can be swapped or benchmarked independently — e.g. comparing NLLB against MarianMT, or Whisper-small against Whisper-medium, on the same audio. That modularity and transparency is the main thing this project offers over a one-off script tied to a single API or model.

## How it works

1. **Voice Activity Detection (VAD)** continuously watches the raw mic stream and detects natural pauses in speech, automatically segmenting audio into utterances — this is what makes the system feel real-time instead of requiring a manual "stop recording" step.
2. **ASR (Automatic Speech Recognition)** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcribes each utterance to text and detects the spoken language.
3. **MT (Machine Translation)** — [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) (via Hugging Face `transformers`) translates the transcribed text into the target language you choose.
4. **TTS (Text-to-Speech)** — [Piper](https://github.com/rhasspy/piper) synthesizes the translated text into spoken audio (optional stage — commented out by default until you download a voice model).

## Requirements

- Python 3.9–3.11
- A working microphone
- ~2–4 GB free disk space (for model downloads on first run)
- Windows, macOS, or Linux

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install faster-whisper transformers sentencepiece torch sounddevice numpy webrtcvad piper-tts
```

**Windows note:** if `webrtcvad` fails to build with a "Microsoft Visual C++ 14.0 required" error, install the prebuilt-wheel fork instead — it's a drop-in replacement, no code changes needed:
```bash
pip uninstall webrtcvad -y
pip install webrtcvad-wheels
```

**macOS/Linux note:** `sounddevice` requires PortAudio:
```bash
# macOS
brew install portaudio
# Ubuntu/Debian
sudo apt install portaudio19-dev
```

## Usage

```bash
python realtime_s2s_translate.py
```

The script will ask which language to translate into:

```
Choose a target language: hindi, spanish, french, german, japanese, chinese, arabic, bengali, russian, english
Translate into: spanish
Listening... (Ctrl+C to stop)
```

Speak into your mic. After each pause, it prints the detected spoken language, the transcription, and the translation:

```
[6.82s] Detected language: English
        SRC: Thank you.
        TGT: Gracias.
```

Press `Ctrl+C` to stop.

## Enabling spoken output (TTS)

By default the script only prints text. To have it speak the translation out loud:

1. Download a Piper voice model (`.onnx` + matching `.onnx.json`) for your target language from the [Piper voices list](https://github.com/rhasspy/piper/blob/master/VOICES.md).
2. In `realtime_s2s_translate.py`, uncomment the `tts = TTS(...)` line in `main()` and point it at your downloaded `.onnx` file path.
3. Uncomment the `tts.speak(translated_text)` line further down in the same function.

## Configuration

Adjustable constants near the top of the script:

| Constant | Purpose |
|---|---|
| `WHISPER_MODEL_SIZE` | `tiny`/`base`/`small`/`medium`/`large-v3` — bigger is slower but more accurate |
| `VAD_AGGRESSIVENESS` | 0–3, how aggressively silence is filtered from speech |
| `SILENCE_TIMEOUT_MS` | how long a pause must be before an utterance is considered complete |
| `LANGUAGE_MENU` | add more target languages here using NLLB (FLORES-200) codes |

## Adding a new language

Add an entry to `LANGUAGE_MENU` in the script using the language's [NLLB/FLORES-200 code](https://huggingface.co/facebook/nllb-200-distilled-600M), e.g.:

```python
LANGUAGE_MENU = {
    ...
    "italian": "ita_Latn",
}
```

## Known limitations

- Small Whisper models can mishear words, especially with background noise or unclear audio — try a larger `WHISPER_MODEL_SIZE` for better accuracy.
- Currently supports a fixed language menu (10 languages) — NLLB itself supports ~200, so the menu can be extended as needed.
- TTS output is disabled by default until a Piper voice model is downloaded and configured.

## Possible extensions

- Benchmark NLLB vs. MarianMT on translation quality (BLEU score) and latency.
- Compare Whisper model sizes on accuracy vs. speed trade-off.
- Add a simple GUI instead of a terminal-only interface.
- Support live streaming translation (partial results before an utterance finishes) instead of per-utterance batching.

## License

This is a personal/academic project skeleton. Check the individual licenses of the underlying models and libraries used (Whisper, NLLB-200, Piper) before any commercial use.
