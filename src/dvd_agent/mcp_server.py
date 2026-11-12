"""MCP server: the DVD rental agent as tools for Claude Code, Claude Desktop or any MCP client.

python -m dvd_agent.mcp_server                      stdio (the client starts it)
python -m dvd_agent.mcp_server --http --port 8002   Streamable HTTP at http://127.0.0.1:8002/mcp
docker compose up -d mcp                            the same, in Docker

Tools: ask (the agent, with its self-check), page (more pages of the last list), run_sql (your own
read-only SQL), search_films (plot/theme search), schema. Over HTTP, MCP_TOKEN (if set) must be sent
as "Authorization: Bearer <token>".
"""
from __future__ import annotations

import argparse
import contextlib
import hmac
import logging
import uuid
from typing import Any

import mlflow

from .agent import paging, schema, sql_guard, tools
from .agent.graph import ask as agent_ask
from .agent.graph import get_agent
from .config import get_settings
from .llm import ensure_embed_model_loaded
from .tracing import init_mlflow, trace_url

log = logging.getLogger("dvd_agent.mcp")
MAX_PAGE_SIZE = 500

INSTRUCTIONS = """\
Answers questions about a DVD rental business (sample PostgreSQL database "dvdrental": films, actors,
categories, customers, stores, staff, inventory, rentals, payments). Rentals run May 2005 - Feb 2006;
payments only Feb - May 2007.

- Use `ask` for questions in plain English. The agent writes and runs the SQL, checks its answer
  against the results, cites them ([Q1] = first SQL result, [S1] = first film search hit) and says
  "not in the data" instead of guessing. Pass the returned conversation_id back for follow-ups
  ("who acts in it?") and for paging.
- Use `page` for more rows of a long list from `ask` (50 rows per page).
- Use `run_sql` to run your own read-only SELECT (see `schema` first). Nothing checks that your SQL
  answers the question: prefer `ask` unless you need a specific query.
- Use `search_films` to find films by plot, theme or setting.
Staff usernames/passwords, street addresses, postal codes and phone numbers are not available.
"""


@contextlib.asynccontextmanager
async def _lifespan(server):
    init_mlflow()
    log.info("embedding model: %s", ensure_embed_model_loaded())
    get_agent()   # compile the graph and load the DSPy program once
    yield {}


def build_server():
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    mcp = MCPServer("dvd-agent", title="DVD Rental Agent", instructions=INSTRUCTIONS, lifespan=_lifespan)
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @mcp.tool(annotations=read_only)
    def ask(question: str, conversation_id: str | None = None) -> dict[str, Any]:
        """Answer a question about the DVD rental data in plain English (takes 10-60 s).

        Returns the answer with [Q#]/[S#] citations, a status (answered | partial | not_found), the
        SQL results it is based on (page 1 of each), and a conversation_id. Pass the conversation_id
        back to ask a follow-up ("which of them are rated R?") or to page a long list. Paging
        requests such as "page 2" or "20 per page" also work here.
        """
        cid = conversation_id or uuid.uuid4().hex
        r = agent_ask(question, cid)
        sources = []
        for c in r.get("citations", []):
            if c["type"] == "sql":
                sources.append({k: c.get(k) for k in ("id", "sql", "columns", "rows", "row_count", "page", "pages")})
            else:
                sources.append({k: c.get(k) for k in ("id", "film_id", "title", "description")})
        return {"conversation_id": cid, "status": r["status"], "answer": r["answer"],
                "understood_as": r.get("standalone_question"), "sources": sources,
                "trace_url": trace_url(mlflow.get_last_active_trace_id())}

    @mcp.tool(annotations=read_only)
    def page(conversation_id: str, request: str) -> dict[str, Any]:
        """Show other pages of the last long list that `ask` returned in this conversation.

        request: "page 2", "pages 2, 5", "pages 3 to 5", "next page", "previous page", "last page",
        or a page size such as "20 per page". The query runs again; rows come straight from the database.
        """
        if paging.parse_command(request) is None:
            raise ToolError('not a paging request; use e.g. "page 2", "pages 2, 5", "next page", '
                            '"last page" or "20 per page"')
        r = agent_ask(request, conversation_id)
        return {"conversation_id": conversation_id, "status": r["status"], "answer": r["answer"],
                "pages": [{k: c.get(k) for k in ("page", "pages", "page_size", "row_count", "columns", "rows")}
                          for c in r.get("citations", []) if c.get("type") == "sql"]}

    @mcp.tool(annotations=read_only)
    def run_sql(sql: str, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        """Run one read-only PostgreSQL SELECT (WITH ... SELECT is fine) and return one page of rows.

        Runs as a read-only role with a 10 s timeout; tables and columns as listed by `schema`.
        Returns columns, rows, row_count (all rows), page and pages. On an error, the tool fails with
        the database's message so you can fix the query. page_size: 1-500.
        """
        size = max(1, min(int(page_size), MAX_PAGE_SIZE))
        try:
            safe = sql_guard.check(sql, schema.table_names(), schema.blocked_columns())
        except sql_guard.UnsafeSQL as e:
            raise ToolError(str(e)) from None
        try:
            r = paging.run_paged(safe, size, max(1, int(page)))
        except Exception as e:
            msg = " ".join(str(e).strip().splitlines()[:3])
            raise ToolError(f"{type(e).__name__}: {msg}"[:500]) from None
        return {k: r[k] for k in ("sql", "columns", "rows", "row_count", "page", "pages", "page_size")}

    @mcp.tool(annotations=read_only)
    def search_films(query: str, k: int = 10) -> list[dict[str, Any]]:
        """Find films by what they are about (plot, theme, setting), e.g. "a robot", "set on a submarine".

        Hybrid keyword (BM25) + semantic search over titles and descriptions. Returns the k best
        matches (k: 1-50); these are top matches, not every matching film.
        """
        hits = tools.search_films([query], k=max(1, min(int(k), 50)))
        return [{key: h.get(key) for key in ("film_id", "title", "description", "release_year", "rating", "score")}
                for h in hits]

    @mcp.tool(name="schema", annotations=read_only)
    def schema_text() -> str:
        """Tables and views you can query with `run_sql`: columns, keys (->), known values, date ranges."""
        return schema.describe()

    return mcp


def http_app(mcp):
    """The Streamable HTTP app at /mcp, with the Host check and the optional bearer token."""
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.responses import JSONResponse

    s = get_settings()
    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"] + [h.strip() for h in s.mcp_allowed_hosts.split(",") if h.strip()]
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts,
        allowed_origins=[f"http://{h}" for h in hosts] + [f"https://{h}" for h in hosts])
    app = mcp.streamable_http_app(transport_security=security, host="0.0.0.0")
    if not s.mcp_token:
        return app
    expected = f"Bearer {s.mcp_token}".encode()

    async def with_token(scope, receive, send):
        if scope["type"] == "http":
            got = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(got, expected):
                await JSONResponse({"error": "missing or wrong bearer token"}, status_code=401)(scope, receive, send)
                return
        await app(scope, receive, send)
    return with_token


def main() -> None:
    ap = argparse.ArgumentParser(description="DVD rental agent as an MCP server")
    ap.add_argument("--http", action="store_true", help="Streamable HTTP at /mcp instead of stdio")
    ap.add_argument("--host", default="127.0.0.1", help="HTTP: address to listen on (0.0.0.0 in Docker)")
    ap.add_argument("--port", type=int, default=8002)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mcp = build_server()
    if not a.http:
        mcp.run("stdio")   # stdout is the protocol; the SDK sends stray prints to stderr
        return
    import uvicorn

    if a.host not in ("127.0.0.1", "localhost") and not get_settings().mcp_token:
        log.warning("listening on %s without MCP_TOKEN: anyone who can reach this port can query the data "
                    "(in Docker, the compose file publishes it on 127.0.0.1 only)", a.host)
    uvicorn.run(http_app(mcp), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
