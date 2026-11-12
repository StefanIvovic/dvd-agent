"""The MCP server: its tools as a client sees them, run_sql's guard, and the HTTP token check. No LLM."""
import asyncio

import pytest

from dvd_agent import mcp_server
from dvd_agent.config import get_settings


@pytest.fixture
def server(monkeypatch):
    # skip the startup work (MLflow, LM Studio, compiling the agent): these tests call no LLM
    for name in ("init_mlflow", "ensure_embed_model_loaded", "get_agent"):
        monkeypatch.setattr(mcp_server, name, lambda: None)
    return mcp_server.build_server()


def call(server, tool=None, args=None):
    from mcp import Client

    async def run():
        async with Client(server) as c:
            return await (c.call_tool(tool, args or {}) if tool else c.list_tools())
    return asyncio.run(run())


def test_tools_are_listed_read_only(server):
    tools = {t.name: t for t in call(server).tools}
    assert set(tools) == {"ask", "page", "run_sql", "search_films", "schema"}
    assert all(t.annotations.read_only_hint for t in tools.values())


def test_page_rejects_non_paging_text(server):
    r = call(server, "page", {"conversation_id": "x", "request": "hello"})
    assert r.is_error and "not a paging request" in r.content[0].text


@pytest.mark.db
@pytest.mark.parametrize("sql, message", [
    ("DELETE FROM rental", "only SELECT"),
    ("SELECT username, password FROM staff", "not available"),
    ("SELECT nope FROM film", "UndefinedColumn"),
])
def test_run_sql_errors(server, sql, message):
    r = call(server, "run_sql", {"sql": sql})
    assert r.is_error and message in r.content[0].text


@pytest.mark.db
def test_run_sql_pages(server):
    r = call(server, "run_sql", {"sql": "SELECT title FROM film ORDER BY title", "page": 2, "page_size": 10})
    d = r.structured_content
    assert not r.is_error
    assert (d["row_count"], d["pages"], d["page"], len(d["rows"])) == (1000, 100, 2, 10)


def test_http_token(server, monkeypatch):
    from starlette.testclient import TestClient

    monkeypatch.setenv("MCP_TOKEN", "test-token")
    get_settings.cache_clear()
    try:
        client = TestClient(mcp_server.http_app(server), raise_server_exceptions=False)
        assert client.post("/mcp", json={}).status_code == 401
        assert client.post("/mcp", json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.post("/mcp", json={}, headers={"Authorization": "Bearer test-token"}).status_code != 401
    finally:
        get_settings.cache_clear()
