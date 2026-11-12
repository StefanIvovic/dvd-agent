"""Unit tests: no LLM. Tests marked `db` need the Postgres container (docker compose up -d postgres)
and are skipped when it is not reachable."""
import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _db_up() -> bool:
    try:
        from dvd_agent.db import postgres as pg

        with pg.reader() as c:
            c.execute("SELECT 1")
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if not any("db" in i.keywords for i in items) or _db_up():
        return
    skip = pytest.mark.skip(reason="Postgres not reachable (docker compose up -d postgres)")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)
