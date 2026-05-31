import os
import sys
import json
import time
import wave
import argparse
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
import yaml
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import stt
import vad
from log import STTLogger

# HTTP status codes
HTTP_ERR_INTERNAL = 500
HTTP_ERR_UNAVAILABLE = 503

# Server defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 47102

# Audio constants
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SUFFIX = ".wav"
MIN_AUDIO_BYTES = 1000
CHUNK_ACK_INTERVAL = 32000

# VAD defaults
DEFAULT_VAD_THRESHOLD = 0.02

app = FastAPI(title="STT Service")
logger = STTLogger()


class TranscribeResp(BaseModel):
    text: str
    language: str | None = None
    segments: list | None = None
    words: list | None = None


class HealthResp(BaseModel):
    status: str
    model: str | None = None
    device: str | None = None


class VADAnalyzeResp(BaseModel):
    has_speech: bool
    segments: list = []
    energy: float = 0.0


class VADStatusResp(BaseModel):
    available: bool
    webrtcvad_installed: bool
    pyaudio_installed: bool


def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_suffix(filename):
    return os.path.splitext(filename or DEFAULT_SUFFIX)[1] or DEFAULT_SUFFIX


@app.get("/health", response_model=HealthResp)
async def health():
    """Check service health."""
    if stt.engine is None:
        return HealthResp(status="not_initialized", model=None, device=None)

    loaded = stt.engine.model is not None
    return HealthResp(
        status="ok" if loaded else "model_not_loaded",
        model=stt.engine.model_name,
        device=stt.engine.device if loaded else None,
    )


@app.post(
    "/transcribe",
    response_model=TranscribeResp,
    response_model_exclude_none=True,
)
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    segments: bool = Form(False),
    words: bool = Form(False),
    translate: bool = Form(False),
):
    """Transcribe audio, optionally with segments, word timestamps, or translation."""
    if stt.engine is None:
        raise HTTPException(
            status_code=HTTP_ERR_UNAVAILABLE, detail="STT engine not initialized"
        )

    suffix = get_suffix(file.filename)
    tmp = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            content = await file.read()
            f.write(content)
            tmp = f.name

        task = "translate" if translate else "transcribe"
        result = stt.engine.transcribe(
            tmp, lang=language, task=task, word_timestamps=words
        )

        resp = TranscribeResp(text=result["text"], language=result.get("language"))

        if segments:
            resp.segments = [
                {
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "text": s.get("text", "").strip(),
                }
                for s in result.get("segments", [])
            ]

        if words:
            # flatten segment words into one searchable list
            flat = []
            for s in result.get("segments", []):
                for w in s.get("words", []):
                    flat.append(
                        {
                            "word": w.get("word", "").strip(),
                            "start": w.get("start"),
                            "end": w.get("end"),
                            "probability": w.get("probability"),
                        }
                    )
            resp.words = flat

        return resp

    except Exception as e:
        logger.exception(f"transcription failed: {e}")
        raise HTTPException(status_code=HTTP_ERR_INTERNAL, detail=str(e)) from e

    finally:
        if tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass


@app.post("/vad/analyze", response_model=VADAnalyzeResp)
async def vad_analyze(
    file: UploadFile = File(...), threshold: float = Form(DEFAULT_VAD_THRESHOLD)
):
    """Analyze audio for voice activity."""
    suffix = get_suffix(file.filename)
    tmp = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            content = await file.read()
            f.write(content)
            tmp = f.name

        try:
            with wave.open(tmp, "rb") as wf:
                audio = wf.readframes(wf.getnframes())
                rate = wf.getframerate()
        except Exception:
            audio = content
            rate = DEFAULT_SAMPLE_RATE

        segs = vad.detect_speech_energy(audio, threshold=threshold, sample_rate=rate)
        energy = vad.audio_energy(audio)

        return VADAnalyzeResp(has_speech=len(segs) > 0, segments=segs, energy=energy)

    except Exception as e:
        logger.exception(f"VAD analysis failed: {e}")
        raise HTTPException(status_code=HTTP_ERR_INTERNAL, detail=str(e)) from e

    finally:
        if tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass


@app.get("/vad/status", response_model=VADStatusResp)
async def vad_status():
    """Check VAD availability."""
    return VADStatusResp(
        available=vad.HAS_WEBRTCVAD,
        webrtcvad_installed=vad.HAS_WEBRTCVAD,
        pyaudio_installed=vad.HAS_PYAUDIO,
    )


active_vad_conns = []


@app.websocket("/ws/vad")
async def websocket_vad(ws: WebSocket):
    """WebSocket for real-time VAD."""
    await ws.accept()
    active_vad_conns.append(ws)

    try:
        detector = vad.VoiceActivityDetector()
    except ImportError:
        await ws.send_json({"error": "webrtcvad not installed"})
        await ws.close()
        return

    try:
        while True:
            data = await ws.receive_bytes()

            prev = detector.state
            new = detector.process_frame(data)

            if new != prev or new == vad.VADState.SPEECH:
                await ws.send_json(
                    {
                        "state": new,
                        "is_speaking": detector.is_speaking(),
                        "time_in_state": detector.time_since_state_change(),
                    }
                )

    except WebSocketDisconnect:
        pass

    finally:
        if ws in active_vad_conns:
            active_vad_conns.remove(ws)


class StreamingSession:
    """Manages streaming STT session."""

    def __init__(self):
        self.chunks = []
        self.is_speaking = False
        self.speech_start = 0.0
        self.last_activity = 0.0
        self.total_bytes = 0

    def add_chunk(self, chunk):
        self.chunks.append(chunk)
        self.total_bytes += len(chunk)
        self.last_activity = time.time()
        return self.total_bytes

    def get_audio(self):
        return b"".join(self.chunks)

    def clear(self):
        self.chunks.clear()
        self.total_bytes = 0
        self.is_speaking = False
        self.speech_start = 0.0


streaming_sessions = {}


@app.websocket("/ws/stt")
async def websocket_stt(ws: WebSocket):
    """WebSocket for streaming STT."""
    await ws.accept()
    sid = str(id(ws))
    sess = StreamingSession()
    streaming_sessions[sid] = sess

    logger.info(f"streaming STT session started: {sid}")

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if msg["type"] == "websocket.receive":
                if "bytes" in msg:
                    total = sess.add_chunk(msg["bytes"])

                    if total % CHUNK_ACK_INTERVAL == 0:
                        await ws.send_json(
                            {"type": "chunk_ack", "bytes_received": total}
                        )

                elif "text" in msg:
                    try:
                        data = json.loads(msg["text"])
                    except Exception:
                        continue

                    typ = data.get("type", "")

                    if typ == "start":
                        sess.clear()
                        sess.is_speaking = True
                        sess.speech_start = time.time()
                        await ws.send_json({"type": "started"})

                    elif typ == "end":
                        sess.is_speaking = False
                        audio = sess.get_audio()

                        if len(audio) < MIN_AUDIO_BYTES:
                            await ws.send_json(
                                {
                                    "type": "transcription",
                                    "text": "",
                                    "final": True,
                                    "error": "audio_too_short",
                                }
                            )
                            sess.clear()
                            continue

                        await ws.send_json({"type": "transcribing"})

                        try:
                            if stt.engine is None:
                                raise RuntimeError("STT engine not initialized")

                            suffix = data.get("format", ".webm")
                            if not suffix.startswith("."):
                                suffix = f".{suffix}"

                            result = stt.engine.transcribe_bytes(audio, suffix=suffix)

                            await ws.send_json(
                                {
                                    "type": "transcription",
                                    "text": result["text"],
                                    "language": result.get("language"),
                                    "final": True,
                                }
                            )

                        except (WebSocketDisconnect, RuntimeError):
                            break

                        except Exception as e:
                            logger.exception(f"streaming transcription failed: {e}")
                            try:
                                await ws.send_json(
                                    {
                                        "type": "transcription",
                                        "text": "",
                                        "final": True,
                                        "error": str(e),
                                    }
                                )
                            except Exception:
                                break

                        sess.clear()

                    elif typ == "cancel":
                        sess.clear()
                        await ws.send_json({"type": "cancelled"})

                    elif typ == "ping":
                        await ws.send_json({"type": "pong"})

    except (WebSocketDisconnect, RuntimeError):
        logger.info(f"streaming STT session disconnected: {sid}")

    except Exception as e:
        logger.exception(f"streaming STT error: {e}")

    finally:
        if sid in streaming_sessions:
            del streaming_sessions[sid]


async def broadcast_vad_event(event, data=None):
    """Broadcast VAD event to all clients."""
    msg = {"event": event, **(data or {})}

    for ws in active_vad_conns:
        try:
            await ws.send_json(msg)
        except Exception:
            pass


public_dir = os.path.join(os.path.dirname(__file__), "public")
if os.path.isdir(public_dir):
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="ui")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg",
        default=os.getenv(
            "CFG", os.path.join(os.path.dirname(__file__), "private", "config.yaml")
        ),
    )
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    logger = STTLogger(debug=a.debug)
    logger.info(f"debug={a.debug}")

    cfg = {}
    if os.path.exists(a.cfg):
        logger.info(f"config: {a.cfg}")
        try:
            cfg = load_cfg(a.cfg)
        except Exception as e:
            logger.warning(f"failed to load config: {e}")

    whisper_cfg = cfg.get("whisper", {})
    model = a.model or whisper_cfg.get("model", "base")
    device = a.device or whisper_cfg.get("device", "auto")
    compute = whisper_cfg.get("compute_type", "auto")

    server_cfg = cfg.get("server", {})
    host = a.host or server_cfg.get("host", DEFAULT_HOST)
    port = a.port or server_cfg.get("port", DEFAULT_PORT)

    logger.info(
        f"initializing faster-whisper model: {model} (device={device}, compute={compute})"
    )
    stt.init(model=model, device=device, compute=compute)

    uvicorn.run(app, host=host, port=port, log_level="debug" if a.debug else "info")
