# todo: whisperx here for multi-speaker when friends talk to it
import os
import tempfile

import numpy as np
from faster_whisper import WhisperModel

from echo_common import logger

DEFAULT_MODEL = "base"
DEFAULT_DEVICE = "auto"
DEFAULT_COMPUTE = "auto"

COMPUTE_FLOAT16 = "float16"
COMPUTE_INT8 = "int8"
COMPUTE_DEFAULT = "default"


class FasterWhisperSTT:
    """STT engine using faster-whisper."""

    def __init__(
        self, model=DEFAULT_MODEL, device=DEFAULT_DEVICE, compute=DEFAULT_COMPUTE
    ):
        self.model_name = model
        self.device = device
        self.compute = compute
        self.model = None

    def load_model(self):
        """Load model into memory."""
        if self.model is not None:
            return

        logger.info(f"loading faster-whisper model: {self.model_name}")

        try:
            compute = self.compute

            if compute == DEFAULT_COMPUTE:
                if self.device == "cuda":
                    compute = COMPUTE_FLOAT16
                elif self.device == "cpu":
                    compute = COMPUTE_INT8
                else:
                    compute = COMPUTE_DEFAULT

            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=compute,
            )
            logger.info(
                f"faster-whisper model loaded: {self.model_name} on {self.device}"
            )

        except Exception as e:
            logger.exception(f"failed to load faster-whisper model: {e}")
            raise

    def transcribe(
        self,
        path,
        lang=None,
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
    ):
        """Transcribe audio file."""
        self.load_model()

        logger.info(f"transcribing: {path}")

        try:
            segs, info = self.model.transcribe(
                path,
                language=lang,
                task=task,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                word_timestamps=word_timestamps,
            )

            seg_list = []
            parts = []

            for s in segs:
                seg = {"start": s.start, "end": s.end, "text": s.text}

                # todo: tag words with speaker when friends join the stream
                if word_timestamps and s.words:
                    seg["words"] = [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in s.words
                    ]

                seg_list.append(seg)
                parts.append(s.text)

            text = "".join(parts).strip()
            logger.info(
                f"transcription complete: {len(text)} chars, lang={info.language}"
            )

            return {
                "text": text,
                "segments": seg_list,
                "language": info.language,
                "language_probability": info.language_probability,
            }

        except Exception as e:
            logger.exception(f"transcription failed: {e}")
            raise

    def transcribe_bytes(self, audio, suffix=".wav", lang=None, task="transcribe"):
        """Transcribe audio from bytes."""
        tmp = None

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio)
                tmp = f.name

            return self.transcribe(tmp, lang=lang, task=task)

        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    def transcribe_numpy(self, audio, sample_rate=16000, lang=None, task="transcribe"):
        """Transcribe audio from numpy array."""
        self.load_model()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        try:
            segs, info = self.model.transcribe(
                audio, language=lang, task=task, vad_filter=True
            )

            seg_list = []
            parts = []

            for s in segs:
                seg_list.append({"start": s.start, "end": s.end, "text": s.text})
                parts.append(s.text)

            return {
                "text": "".join(parts).strip(),
                "segments": seg_list,
                "language": info.language,
            }

        except Exception as e:
            logger.exception(f"transcription failed: {e}")
            raise


engine = None


def init(model=DEFAULT_MODEL, device=DEFAULT_DEVICE, compute=DEFAULT_COMPUTE):
    """Initialize global STT engine."""
    global engine

    engine = FasterWhisperSTT(model=model, device=device, compute=compute)
    engine.load_model()


def transcribe(path, lang=None):
    """Transcribe audio file."""
    if engine is None:
        raise RuntimeError("STT engine not initialized")

    result = engine.transcribe(path, lang=lang)
    return result["text"]


def transcribe_bytes(audio, suffix=".wav", lang=None):
    """Transcribe audio bytes."""
    if engine is None:
        raise RuntimeError("STT engine not initialized")

    result = engine.transcribe_bytes(audio, suffix=suffix, lang=lang)
    return result["text"]


def transcribe_full(path, lang=None):
    """Transcribe audio with full segment details."""
    if engine is None:
        raise RuntimeError("STT engine not initialized")

    return engine.transcribe(path, lang=lang)
