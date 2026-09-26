import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contextlib

import uvicorn
import yaml
from echo_common import (
    HTTP_ERR_INTERNAL,
    HTTP_ERR_UNAVAILABLE,
    configure_logging,
    logger,
    resolve_path,
    service_root,
    service_version,
)
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel

import stt

# HTTP status codes

SERVICE_ROOT = service_root(__file__)

# Audio constants
DEFAULT_SUFFIX = ".wav"

app = FastAPI(title="STT Service")


class TranscribeResp(BaseModel):
    text: str
    language: str | None = None
    segments: list | None = None
    words: list | None = None


class HealthResp(BaseModel):
    status: str
    model: str | None = None
    device: str | None = None


def load_cfg(p):
    with open(p, encoding="utf-8") as f:
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
            resp.words = [
                {
                    "word": w.get("word", "").strip(),
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "probability": w.get("probability"),
                }
                for s in result.get("segments", [])
                for w in s.get("words", [])
            ]

        return resp

    except Exception as e:
        logger.exception(f"transcription failed: {e}")
        raise HTTPException(status_code=HTTP_ERR_INTERNAL, detail=str(e)) from e

    finally:
        if tmp:
            with contextlib.suppress(Exception):
                os.remove(tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg",
        default=os.getenv("CFG", resolve_path("src/private/config.yaml", SERVICE_ROOT)),
    )
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    configure_logging(debug=args.debug)
    logger.info(f"debug={args.debug}")

    cfg = {}
    if os.path.exists(args.cfg):
        logger.info(f"config: {args.cfg}")
        try:
            cfg = load_cfg(args.cfg)
        except Exception as e:
            logger.warning(f"failed to load config: {e}")

    whisper_cfg = cfg.get("whisper", {})
    model = args.model or whisper_cfg.get("model", "base")
    device = args.device or whisper_cfg.get("device", "auto")
    compute = whisper_cfg.get("compute_type", "auto")

    host = args.host or cfg.get("server", {}).get("host", "0.0.0.0")
    port = args.port or cfg.get("server", {}).get("port", 47102)

    logger.info(
        f"initializing faster-whisper model: {model}"
        f" (device={device}, compute={compute})"
    )
    logger.info(f"service version: {service_version(__file__)}")
    stt.init(model=model, device=device, compute=compute)

    uvicorn.run(app, host=host, port=port, log_level="debug" if args.debug else "info")
