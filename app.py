"""
ManiaMapper Web API
-------------------
FastAPI backend that accepts an audio file upload and returns a generated .osz file.

Endpoints:
    POST /api/generate   — upload audio, returns .osz download
    GET  /api/health     — health check

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import tempfile
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ManiaMapper API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "mania_model.pt"
STATIC_DIR  = BASE_DIR / "static"
WORK_DIR    = Path(tempfile.gettempdir()) / "maniamapper_jobs"
WORK_DIR.mkdir(parents=True, exist_ok=True)

VALID_DIFFICULTIES = {"Easy", "Normal", "Hard", "Insane"}
VALID_EXTENSIONS   = {".mp3", ".ogg", ".wav", ".flac", ".m4a"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
    }


@app.post("/api/generate")
async def generate(
    audio: UploadFile = File(...),
    difficulty: str   = Form("Hard"),
    title: str        = Form(""),
    artist: str       = Form(""),
):
    # ── Validate inputs ───────────────────────────────────────────────────────
    if difficulty not in VALID_DIFFICULTIES:
        raise HTTPException(400, f"difficulty must be one of {VALID_DIFFICULTIES}")

    ext = Path(audio.filename).suffix.lower()
    if ext not in VALID_EXTENSIONS:
        raise HTTPException(400, f"Unsupported audio format: {ext}")

    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model not loaded on server yet.")

    # ── Save uploaded audio to temp dir ──────────────────────────────────────
    job_id   = uuid.uuid4().hex
    job_dir  = WORK_DIR / job_id
    job_dir.mkdir()

    audio_path = job_dir / f"audio{ext}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    # ── Run ManiaMapper ───────────────────────────────────────────────────────
    safe = lambda s: "".join(c for c in s if c.isalnum() or c in " -_") or "map"
    osz_name = f"{safe(artist)} - {safe(title)} [{difficulty}].osz" if title else f"map [{difficulty}].osz"
    osz_out  = job_dir / osz_name
    cmd = [
        sys.executable, str(BASE_DIR / "ManiaMapper.py"),
        str(audio_path),
        "--nn",         str(MODEL_PATH),
        "--output",     str(osz_out),
        "--difficulty", difficulty,
        "--no-sv",
    ]
    if title:
        cmd += ["--title", title]
    if artist:
        cmd += ["--artist", artist]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(BASE_DIR),
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Generation timed out (>120s).")

    if result.returncode != 0:
        raise HTTPException(500, f"Generation failed:\n{result.stderr[-2000:]}")

    # ── Find the .osz output ──────────────────────────────────────────────────
    osz_files = list(job_dir.glob("*.osz"))
    if not osz_files:
        raise HTTPException(500, "No .osz file was produced.")

    osz_path = osz_files[0]
    return FileResponse(
        path=str(osz_path),
        media_type="application/octet-stream",
        filename=osz_path.name,
        headers={"X-Job-Id": job_id},
    )


# ── Serve frontend ────────────────────────────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
