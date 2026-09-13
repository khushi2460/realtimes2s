"""
Real-Time Speech-to-Speech Translation Pipeline
=================================================
Architecture: Mic -> VAD chunking -> ASR (Whisper) -> MT (NLLB) -> TTS (Piper) -> Speaker

Install (CPU-friendly versions shown; swap for GPU builds if you have CUDA):
    pip install faster-whisper transformers sentencepiece torch sounddevice numpy webrtcvad
    pip install piper-tts   # or: pip install TTS   (Coqui, heavier but higher quality)

This is a SKELETON meant to be extended, not a finished product. Each stage is
isolated in its own function/class so you can swap models independently
(e.g. replace NLLB with MarianMT, or Piper with Coqui) without touching the rest.
"""

import collections
import queue
import sys
import time

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # required by Whisper + webrtcvad
FRAME_MS = 30                # webrtcvad only accepts 10/20/30ms frames
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
VAD_AGGRESSIVENESS = 2       # 0-3, higher = more aggressive at filtering non-speech
SILENCE_TIMEOUT_MS = 700     # how long silence before we consider an utterance "done"
WHISPER_MODEL_SIZE = "small"  # tiny/base/small/medium/large-v3 (bigger = slower, more accurate)

# NLLB (FLORES-200) codes for the languages you can pick as a translation target.
# Add more entries here any time — key is what the user types, value is the NLLB code.
LANGUAGE_MENU = {
    "hindi": "hin_Deva",
    "spanish": "spa_Latn",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "japanese": "jpn_Jpan",
    "chinese": "zho_Hans",
    "arabic": "arb_Arab",
    "bengali": "ben_Beng",
    "russian": "rus_Cyrl",
    "english": "eng_Latn",
}

# Whisper reports detected source language as a short ISO code (e.g. "en", "hi").
# This just makes the printed output readable instead of showing raw codes.
WHISPER_LANG_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
    "de": "German", "ja": "Japanese", "zh": "Chinese", "ar": "Arabic",
    "bn": "Bengali", "ru": "Russian",
}


def choose_target_language() -> str:
    """Ask the user which language to translate INTO, every time the script runs."""
    print("Choose a target language:", ", ".join(LANGUAGE_MENU.keys()))
    while True:
        choice = input("Translate into: ").strip().lower()
        if choice in LANGUAGE_MENU:
            return LANGUAGE_MENU[choice]
        print(f"'{choice}' not recognized — pick one of: {', '.join(LANGUAGE_MENU.keys())}")

# ---------------------------------------------------------------------------
# STAGE 0: Voice Activity Detection + audio chunking
# ---------------------------------------------------------------------------
class VADChunker:
    """
    Reads raw mic frames continuously and yields complete "utterances"
    (as numpy float32 arrays) once it detects a pause after speech.
    This is what makes the system feel "real-time" instead of waiting
    for you to press stop — it auto-segments on natural pauses.
    """
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.audio_q = queue.Queue()
        self.ring_buffer = collections.deque(maxlen=int(SILENCE_TIMEOUT_MS / FRAME_MS))

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        self.audio_q.put(indata.copy())

    def stream(self):
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SIZE,
            callback=self._callback,
        ):
            voiced_frames = []
            triggered = False
            print("Listening... (Ctrl+C to stop)")
            while True:
                frame = self.audio_q.get()
                frame_bytes = frame.tobytes()
                is_speech = self.vad.is_speech(frame_bytes, SAMPLE_RATE)

                if not triggered:
                    self.ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, s in self.ring_buffer if s])
                    if num_voiced > 0.5 * self.ring_buffer.maxlen:
                        triggered = True
                        voiced_frames.extend(f for f, s in self.ring_buffer)
                        self.ring_buffer.clear()
                else:
                    voiced_frames.append(frame)
                    self.ring_buffer.append((frame, is_speech))
                    num_unvoiced = len([f for f, s in self.ring_buffer if not s])
                    if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                        # pause detected -> emit the utterance
                        triggered = False
                        audio = np.concatenate(voiced_frames, axis=0).flatten()
                        yield audio.astype(np.float32) / 32768.0  # normalize int16 -> float32
                        voiced_frames = []
                        self.ring_buffer.clear()


# ---------------------------------------------------------------------------
# STAGE 1: ASR — speech to text (Whisper)
# ---------------------------------------------------------------------------
class ASR:
    def __init__(self, model_size=WHISPER_MODEL_SIZE):
        # compute_type="int8" is a big speed win on CPU; use "float16" on GPU
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray):
        """Returns (text, detected_language_code) — language=None lets Whisper auto-detect."""
        segments, info = self.model.transcribe(audio, language=None, vad_filter=False)
        text = " ".join(seg.text.strip() for seg in segments)
        detected_lang = info.language  # e.g. "en", "hi" — Whisper's own ISO code
        return text, detected_lang


# ---------------------------------------------------------------------------
# STAGE 2: MT — text to text translation (NLLB-200)
# ---------------------------------------------------------------------------
class Translator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, text: str, src_lang="eng_Latn", tgt_lang="hin_Deva") -> str:
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt")
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        generated = self.model.generate(
            **inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=200
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# STAGE 3: TTS — text to speech (Piper, offline + fast)
# ---------------------------------------------------------------------------
class TTS:
    """
    Piper needs a downloaded .onnx voice model per language, e.g.:
        https://github.com/rhasspy/piper/blob/master/VOICES.md
    Swap this class for Coqui TTS if you want voice cloning instead.
    """
    def __init__(self, voice_model_path: str):
        from piper import PiperVoice
        self.voice = PiperVoice.load(voice_model_path)

    def speak(self, text: str):
        import sounddevice as sd
        audio_gen = self.voice.synthesize_stream_raw(text)
        for chunk in audio_gen:
            audio_np = np.frombuffer(chunk, dtype=np.int16)
            sd.play(audio_np, samplerate=self.voice.config.sample_rate)
            sd.wait()


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------
def main():
    target_lang_code = choose_target_language()

    chunker = VADChunker()
    asr = ASR()
    translator = Translator()
    # tts = TTS("path/to/hi_IN-voice.onnx")  # uncomment once you have a voice model

    for utterance_audio in chunker.stream():
        t0 = time.time()
        source_text, detected_lang = asr.transcribe(utterance_audio)
        if not source_text:
            continue

        # NLLB needs its own source-language code format (e.g. "eng_Latn"), so map
        # Whisper's short code ("en") to the closest NLLB code. Falls back to
        # English if Whisper detects something not in our small lookup table.
        src_lang_code = {
            "en": "eng_Latn", "hi": "hin_Deva", "es": "spa_Latn", "fr": "fra_Latn",
            "de": "deu_Latn", "ja": "jpn_Jpan", "zh": "zho_Hans", "ar": "arb_Arab",
            "bn": "ben_Beng", "ru": "rus_Cyrl",
        }.get(detected_lang, "eng_Latn")

        translated_text = translator.translate(source_text, src_lang=src_lang_code, tgt_lang=target_lang_code)
        latency = time.time() - t0

        lang_label = WHISPER_LANG_NAMES.get(detected_lang, detected_lang)
        print(f"[{latency:.2f}s] Detected language: {lang_label}")
        print(f"        SRC: {source_text}")
        print(f"        TGT: {translated_text}")

        # tts.speak(translated_text)


if __name__ == "__main__":
    main()
