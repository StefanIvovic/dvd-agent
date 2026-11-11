"""FastAPI service.  uvicorn dvd_agent.api.main:app --port 8001"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ..agent import schema
from ..agent.graph import ask, get_agent
from ..config import get_settings
from ..db import postgres as pg
from ..llm import ensure_embed_model_loaded
from ..tracing import init_mlflow

log = logging.getLogger("dvd_agent.api")
DEPLOYMENT = "sync-embeddings/sync-embeddings"


def _sync_in_background() -> None:
    def run():
        try:
            from ..sync.flows import sync
            log.info("embedding sync: %s", sync())
        except Exception as e:
            log.warning("embedding sync failed: %s", e)
    threading.Thread(target=run, daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_mlflow()
    log.info("embedding model: %s", ensure_embed_model_loaded())
    get_agent()            # compile graph + load (optionally optimized) DSPy program once
    _sync_in_background()  # embed films added/changed while the API was down
    yield


app = FastAPI(title="DVD Rental Agent", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, examples=["Which store made more revenue in 2007?"])
    # send back the conversation_id of the previous answer so "page 2" / "20 per page" page its list
    conversation_id: str | None = Field(default=None, max_length=100)


@app.post("/ask")
def ask_endpoint(req: AskRequest) -> dict:
    # sync endpoint -> FastAPI runs it in a worker thread; LangGraph + DSPy calls are blocking
    result = ask(req.question, req.conversation_id or uuid.uuid4().hex)
    result["trace_id"] = mlflow.get_last_active_trace_id()
    return result


@app.get("/schema", response_class=PlainTextResponse)
def schema_text() -> str:
    return schema.describe()


@app.get("/stats")
def stats() -> dict:
    return pg.embedding_stats()


@app.post("/sync")
async def sync_endpoint() -> dict:
    try:  # preferred: the Prefect deployment served by `python -m dvd_agent.sync.flows --serve`
        from prefect.deployments import run_deployment

        run = await run_deployment(DEPLOYMENT, timeout=0)
        return {"mode": "prefect-deployment", "flow_run_id": str(run.id),
                "ui": f"http://127.0.0.1:4201/runs/flow-run/{run.id}"}
    except Exception as e:
        log.info("Prefect deployment unavailable (%s); syncing in a background thread", e)
    _sync_in_background()
    return {"mode": "background-thread", **pg.embedding_stats()}


@app.get("/health")
def health() -> dict:
    s = get_settings()
    checks = {}
    try:
        with pg.reader() as c:
            checks["postgres"] = c.execute("SELECT count(*) AS n FROM film").fetchone()["n"] > 0
        st = pg.embedding_stats()
        checks["embeddings_current"] = (st["with_embedding"] == st["films"] and st["stale"] == 0) or st
    except Exception as e:
        checks["postgres"] = str(e)
    try:
        if s.llm_provider == "lmstudio":
            models = httpx.get(f"{s.llm_base_url}/models", timeout=5).json()["data"]
            checks["llm"] = s.llm_model in {m["id"] for m in models}
        else:  # cloud provider: only check that a key is configured
            key = s.llm_api_key if s.llm_api_key != "lm-studio" else os.environ.get(
                "ANTHROPIC_API_KEY" if s.llm_provider == "anthropic" else "OPENAI_API_KEY")
            checks["llm"] = bool(key) or f"no API key for {s.llm_provider}"
        state = httpx.get(f"{s.embed_base_url.rsplit('/v1', 1)[0]}/api/v0/models", timeout=5).json()["data"]
        # must be loaded already: a JIT load would land on the GPU and evict the chat model
        checks["embed_model_loaded"] = any(m["id"] == s.embed_model and m.get("state") == "loaded" for m in state)
    except Exception as e:
        checks["llm"] = str(e)
    if not all(v is True for v in checks.values()):
        raise HTTPException(503, checks)
    return checks


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
