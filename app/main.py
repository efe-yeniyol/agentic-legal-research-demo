"""FastAPI demo UI for Northbridge legal research agent."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from northbridge.runner import (  # noqa: E402
    default_method,
    has_openai_key,
    load_dotenv_if_present,
    run_query,
)

load_dotenv_if_present()

app = FastAPI(
    title="Northbridge Legal Research Demo",
    description="MIS311 midterm Type B demo — fictional corpus, not legal advice.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RunRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    method: Literal["deterministic", "openai"] = "deterministic"
    variant: Literal["baseline", "agent_v1", "agent_v2"] = "agent_v2"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "openai_configured": has_openai_key(),
        "default_method": default_method(),
    }


@app.post("/api/run")
def api_run(body: RunRequest) -> dict:
    if body.method == "openai" and not has_openai_key():
        raise HTTPException(
            status_code=400,
            detail=(
                "OPENAI_API_KEY is not set. Copy .env.example to .env, add your key, "
                "restart the server, or choose Deterministic."
            ),
        )
    try:
        # Force tool error path when the famous missing id is requested
        force = "NB-XX-99" in body.prompt.upper()
        result = run_query(
            body.prompt,
            method=body.method,
            variant=body.variant,
            force_tool_error=force,
        )
        return result
    except Exception as exc:  # noqa: BLE001 — surface to UI
        raise HTTPException(status_code=500, detail=str(exc)) from exc
