"""Keeps film_embedding in sync with film (title + description), embedding only what changed.

    python -m dvd_agent.sync.flows            # run once (Prefect flow run, visible in the Prefect UI)
    python -m dvd_agent.sync.flows --serve    # serve the `sync-embeddings` deployment every N minutes

New films and films whose title/description changed get (re-)embedded; deleted films lose their
embedding through ON DELETE CASCADE. `sync()` is the same work without Prefect (API start, POST /sync).
"""
from __future__ import annotations

import argparse
import logging

from prefect import flow, get_run_logger, task

from ..config import get_settings
from ..db import postgres as pg
from ..llm import embed_texts, ensure_embed_model_loaded

log = logging.getLogger(__name__)
BATCH = 64


def pending() -> list[dict]:
    return [r for r in pg.film_texts() if r["hash"] != r["stored_hash"]]


def embed_batch(rows: list[dict]) -> int:
    vectors = embed_texts([r["text"] for r in rows])
    return pg.upsert_embeddings([{**r, "embedding": v} for r, v in zip(rows, vectors)])


def sync(max_rows: int | None = None) -> dict:
    """Embeds new/changed films. Returns counts. max_rows bounds the work (lazy top-up during a search)."""
    todo = pending()[:max_rows] if max_rows else pending()
    if todo:
        ensure_embed_model_loaded()
    done = sum(embed_batch(todo[i:i + BATCH]) for i in range(0, len(todo), BATCH))
    return {"embedded": done, **pg.embedding_stats()}


@task(retries=2, retry_delay_seconds=10)
def embed_batch_task(rows: list[dict]) -> int:
    return embed_batch(rows)


@flow(name="sync-embeddings", log_prints=True)
def sync_embeddings() -> dict:
    logger = get_run_logger()
    logger.info("embedding model: %s", ensure_embed_model_loaded())
    todo = pending()
    logger.info("%d films to embed", len(todo))
    done = sum(embed_batch_task(todo[i:i + BATCH]) for i in range(0, len(todo), BATCH))
    stats = {"embedded": done, **pg.embedding_stats()}
    logger.info("done: %s", stats)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="serve the scheduled deployment (Ctrl+C to stop)")
    ap.add_argument("--interval", type=int, default=get_settings().sync_interval_minutes, help="minutes")
    a = ap.parse_args()
    if a.serve:
        sync_embeddings.serve(name="sync-embeddings", interval=a.interval * 60)
    else:
        print(sync_embeddings())


if __name__ == "__main__":
    main()
