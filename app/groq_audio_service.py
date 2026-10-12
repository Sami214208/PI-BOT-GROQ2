import array
import io
import logging
import math
import os
import sys
import threading
import time
import wave
from collections import deque

import alsaaudio
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class GroqAudioService:
    SAMPLE_RATE = 16000
    FRAME_SAMPLES = 480
    SILENCE_SECONDS = 0.75
    MAX_UTTERANCE_SECONDS = 20
    PRE_ROLL_FRAMES = 10
    TTS_MAX_CHARACTERS = 200

    def __init__(
        self,
        input_device,
        output_device,
        voice,
        on_transcription,
        is_listening,
        logger=None,
    ):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.stt_model = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
        self.tts_model = (
            os.getenv("GROQ_TTS_MODEL") or "canopylabs/orpheus-v1-english"
        )
        self.voice = os.getenv("GROQ_TTS_VOICE") or voice
        self.language = os.getenv("GROQ_STT_LANGUAGE") or None
        self.input_device = input_device
        self.output_device = output_device
        self.on_transcription = on_transcription
        self.is_listening = is_listening
        self.log = logger or logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._thread = None
        self._capture = None
        self.error = None
        self.total_speech_seconds = 0
        self._validate_voice()

    def _validate_voice(self):
        voices_by_model = {
            "canopylabs/orpheus-v1-english": {
                "autumn", "diana", "hannah", "austin", "daniel", "troy"
            },
            "canopylabs/orpheus-arabic-saudi": {
                "abdullah", "fahad", "sultan", "lulwa", "noura", "aisha"
            },
        }
        allowed_voices = voices_by_model.get(self.tts_model)
        if allowed_voices is None:
            raise ValueError(f"Unsupported Groq TTS model: {self.tts_model}")
        if self.voice not in allowed_voices:
            raise ValueError(
                f"Voice '{self.voice}' is not supported by {self.tts_model}. "
                f"Choose one of: {', '.join(sorted(allowed_voices))}"
            )

    def start(self):
        self._capture = self._open_capture()
        self._thread = threading.Thread(
            target=self._listen_loop, name="groq-audio-listener", daemon=True
        )
        self._thread.start()

    def _open_capture(self):
        return alsaaudio.PCM(
            type=alsaaudio.PCM_CAPTURE,
            mode=alsaaudio.PCM_NONBLOCK,
            device=self.input_device,
            channels=1,
            rate=self.SAMPLE_RATE,
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=self.FRAME_SAMPLES,
        )

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._capture is not None and (
            self._thread is None or not self._thread.is_alive()
        ):
            self._capture.close()
            self._capture = None

    def _listen_loop(self):
        frames = []
        pre_roll = deque(maxlen=self.PRE_ROLL_FRAMES)
        last_voice_time = None
        frame_duration = self.FRAME_SAMPLES / self.SAMPLE_RATE
        threshold = int(os.getenv("GROQ_VAD_RMS_THRESHOLD", "450"))

        try:
            while not self._stop_event.is_set():
                if not self.is_listening():
                    frames.clear()
                    pre_roll.clear()
                    last_voice_time = None
                    self._capture.read()
                    time.sleep(0.03)
                    continue

                frame_count, data = self._capture.read()
                if frame_count <= 0 or not data:
                    time.sleep(0.01)
                    continue

                energy = self._rms(data)
                now = time.monotonic()

                if energy >= threshold:
                    if not frames:
                        frames.extend(pre_roll)
                    frames.append(data)
                    last_voice_time = now
                elif frames:
                    frames.append(data)
                    if (
                        last_voice_time is not None
                        and now - last_voice_time >= self.SILENCE_SECONDS
                    ):
                        self._transcribe(frames)
                        self._reset_capture()
                        frames.clear()
                        last_voice_time = None
                else:
                    pre_roll.append(data)

                if (
                    frames
                    and len(frames) * frame_duration >= self.MAX_UTTERANCE_SECONDS
                ):
                    self._transcribe(frames)
                    self._reset_capture()
                    frames.clear()
                    last_voice_time = None
        except Exception as exc:
            self.error = exc
            self.log.exception("Groq audio listener stopped")
            print(f"Groq audio listener stopped: {exc}", flush=True)
        finally:
            if self._capture is not None:
                self._capture.close()
                self._capture = None

    def _reset_capture(self):
        if self._capture is not None:
            self._capture.close()
        self._capture = self._open_capture()

    @staticmethod
    def _rms(data):
        samples = array.array("h")
        samples.frombytes(data)
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            return 0
        return math.sqrt(sum(sample * sample for sample in samples) / len(samples))

    def _transcribe(self, frames):
        audio_file = io.BytesIO()
        with wave.open(audio_file, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(b"".join(frames))

        request = {
            "file": ("utterance.wav", io.BytesIO(audio_file.getvalue())),
            "model": self.stt_model,
            "response_format": "json",
        }
        if self.language:
            request["language"] = self.language

        try:
            result = self.client.audio.transcriptions.create(**request)
        except Exception:
            self.log.exception("Groq speech-to-text request failed")
            print("Groq speech-to-text request failed; see the bot log.", flush=True)
            return

        text = result.text.strip()
        if text:
            self.on_transcription(text)

    @staticmethod
    def split_text(text, max_characters=TTS_MAX_CHARACTERS):
        remaining = text.strip()
        chunks = []
        while remaining:
            if len(remaining) <= max_characters:
                chunks.append(remaining)
                break
            split_at = remaining.rfind(" ", 0, max_characters + 1)
            if split_at <= 0:
                split_at = max_characters
            chunks.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        return chunks

    def speak(self, text, should_stop):
        total_duration = 0
        for chunk in self.split_text(text):
            if should_stop():
                break
            response = self.client.audio.speech.create(
                model=self.tts_model,
                voice=self.voice,
                input=chunk,
                response_format="wav",
            )
            total_duration += self._play_wav(response.read(), should_stop)
        self.total_speech_seconds += total_duration
        return total_duration

    def _play_wav(self, wav_data, should_stop):
        with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            format_name = {
                1: "PCM_FORMAT_U8",
                2: "PCM_FORMAT_S16_LE",
                3: "PCM_FORMAT_S24_3LE",
                4: "PCM_FORMAT_S32_LE",
            }.get(sample_width)
            pcm_format = getattr(alsaaudio, format_name, None) if format_name else None
            if pcm_format is None:
                raise ValueError(
                    f"Unsupported Groq TTS WAV sample width: {sample_width} bytes"
                )

            player = alsaaudio.PCM(
                type=alsaaudio.PCM_PLAYBACK,
                mode=alsaaudio.PCM_NORMAL,
                device=self.output_device,
                channels=channels,
                rate=sample_rate,
                format=pcm_format,
                periodsize=1024,
            )
            frames_played = 0
            try:
                while True:
                    audio_chunk = wav_file.readframes(1024)
                    if not audio_chunk or should_stop():
                        break
                    player.write(audio_chunk)
                    frames_played += len(audio_chunk) // (
                        channels * sample_width
                    )
            finally:
                if should_stop():
                    player.drop()
                player.close()

            return frames_played / sample_rate
