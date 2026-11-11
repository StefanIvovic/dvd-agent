"""Paging of SQL results: every row is reachable, PAGE_SIZE rows at a time.

The answer step sees page 1 of a result. The user can then ask for other pages ("page 2", "pages 3, 5",
"next page", "last page") or another page size ("20 per page"). Pages are not cached: each request runs
the query again with LIMIT/OFFSET, so it always shows the current data. To keep pages consistent across
runs, a query with more than one page gets every output column appended to its ORDER BY as tie-breakers.

Per conversation only the query, its page size and the current page are kept (no rows), for 30 minutes.
"""
from __future__ import annotations

import math
import re
import threading
import time

import sqlglot
from sqlglot import exp

from ..db import postgres as pg

STATE_TTL_SECONDS = 30 * 60


# ------------------------------------------------------------------ SQL rewriting
def _parse(sql: str) -> exp.Query:
    return sqlglot.parse_one(sql, read="postgres")


def _int(node) -> int | None:
    try:
        return int(node.name)
    except (AttributeError, ValueError, TypeError):
        return None


def _bounds(tree: exp.Query) -> tuple[int | None, int, bool]:
    """(limit, offset, rewritable). Not rewritable: FETCH ... WITH TIES or a non-literal limit/offset."""
    lim, off = tree.args.get("limit"), tree.args.get("offset")
    offset = 0
    if off is not None:
        offset = _int(off.expression)
        if offset is None:
            return None, 0, False
    if lim is None:
        return None, offset, True
    if isinstance(lim, exp.Fetch):
        if "TIES" in lim.sql(dialect="postgres").upper():
            return None, offset, False
        n = _int(lim.args.get("count"))
    else:
        n = _int(lim.expression)
    return (n, offset, True) if n is not None else (None, offset, False)


def page_sql(sql: str, page: int, size: int) -> str:
    """The query restricted to one page (keeps a LIMIT the query already has, e.g. "top 100")."""
    tree = _parse(sql)
    limit, offset, ok = _bounds(tree)
    start = (page - 1) * size
    if not ok:   # wrap: paging a WITH TIES / non-literal limit from the outside
        return f"SELECT * FROM ({sql}) AS _paged LIMIT {size} OFFSET {start}"
    n = size if limit is None else max(0, min(size, limit - start))
    tree.set("limit", exp.Limit(expression=exp.Literal.number(n)))
    tree.set("offset", exp.Offset(expression=exp.Literal.number(offset + start)) if offset + start else None)
    return tree.sql(dialect="postgres")


def stable_sql(sql: str, n_columns: int) -> str:
    """Appends every output column (by position) to ORDER BY, so rows come back in the same order each run."""
    tree = _parse(sql)
    _, _, ok = _bounds(tree)
    keys = [exp.Ordered(this=exp.Literal.number(i), nulls_first=False) for i in range(1, n_columns + 1)]
    if not ok:
        inner = f"SELECT * FROM ({sql}) AS _q"
        return _parse(inner).order_by(*keys, copy=False).sql(dialect="postgres")
    return tree.order_by(*keys, append=True, copy=False).sql(dialect="postgres")


def count_rows(sql: str) -> int:
    return pg.run_readonly(f"SELECT count(*) AS n FROM ({sql}) AS _count").rows[0]["n"]


def pages_for(total: int, size: int) -> int:
    return max(1, math.ceil(total / size))


def run_paged(sql: str, size: int, page: int = 1, *, first_run: bool = True) -> dict:
    """Runs the query and returns one page with the total row count.

    first_run: the query comes from the SQL step; if it has more than one page, it is made stable
    (ORDER BY tie-breakers) and the stable SQL is returned, to be reused for later pages.
    """
    if first_run:
        probe = pg.run_readonly(page_sql(sql, 1, size + 1))
        if len(probe.rows) <= size:   # one page: every row is already here
            return _page(sql, probe.columns, probe.rows, len(probe.rows), 1, size)
        try:
            stable = stable_sql(sql, len(probe.columns))
            pg.run_readonly(page_sql(stable, 1, 1))      # e.g. json columns cannot be ordered
            sql = stable
        except Exception:
            pass
    total = count_rows(sql)
    page = min(max(1, page), pages_for(total, size))
    r = pg.run_readonly(page_sql(sql, page, size))
    return _page(sql, r.columns, r.rows, total, page, size)


def _page(sql, columns, rows, total, page, size) -> dict:
    return {"sql": sql, "columns": columns, "rows": rows, "row_count": total, "page": page,
            "pages": pages_for(total, size), "page_size": size,
            "first_row": (page - 1) * size + 1 if rows else 0, "last_row": (page - 1) * size + len(rows)}


# ------------------------------------------------------------------ "page 2", "20 per page", ...
_SIZE = [re.compile(r"\b(\d+)\s*(?:rows?|results?|items?|records?|lines?)?\s*(?:per|a|each|on a|on each)\s+page\b"),
         re.compile(r"\bpage\s*size\s*(?:of|to|=|:)?\s*(\d+)\b"),
         re.compile(r"\b(\d+)\s+(?:rows?|results?|items?|records?|lines?)(?:\s+at\s+a\s+time)?\b")]
_PAGES = re.compile(r"\bpages?\s*(?:number|no\.?|nr\.?|#)?\s*((?:\d+\s*(?:,|and|&|\+|-|–|to|through|\s)\s*)*\d+)\b")
_FILLER = set("""page pages p show me give go to the and please next previous prev back last first per rows row
results result items item records record lines line size of with a at time now then ok okay can you i want see
display get number no nr also only again more on each us let lets let's would like could jump move open view
list
""".split())
_KEYWORDS = {"page", "pages", "next", "previous", "prev", "more", "per", "size", "rows", "results", "items"}


def _expand(spec: str) -> list[int]:
    out = []
    for a, b in re.findall(r"(\d+)\s*(?:(?:-|–|to|through)\s*(\d+))?", spec):
        lo, hi = int(a), int(b or a)
        out += list(range(lo, hi + 1)) if hi >= lo and hi - lo < 1000 else [lo]
    return list(dict.fromkeys(out))


def size_in(text: str, in_question: bool = False) -> int | None:
    """A page size the text asks for ("20 per page"), or None. In a question only "per page" / "page size"
    count ("the top 10 items" is not a page size)."""
    t = text.lower()
    for rx in _SIZE[:2] if in_question else _SIZE:
        m = rx.search(t)
        if m and int(m.group(1)) > 0:
            return int(m.group(1))
    return None


def split_question(question: str) -> tuple[str, int | None]:
    """("list all customers of store 2", 100) for "list all customers of store 2, 100 per page": the page
    size is for the pager, not for the SQL (the model would turn it into LIMIT 100)."""
    size = size_in(question, in_question=True)
    if not size:
        return question, None
    t = question
    for rx in _SIZE[:2]:
        t = re.sub(r"[,;(]?\s*(?:with|show|showing|in pages of)?\s*" + rx.pattern.replace(r"\b(\d+)", r"\b\d+", 1)
                   + r"\s*\)?", " ", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip(" ,;")
    return (t or question), size


_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}


def parse_command(text: str) -> dict | None:
    """A paging request, or None if the text is a normal question.

    Returns {"pages": [2, 5] or None, "rel": "next" | "prev" | "last" | "first" | None, "size": 20 or None}.
    Only short messages made of paging words and numbers count, so a question never becomes a page request.
    """
    t = text.lower().strip().rstrip("?.!")
    t = re.sub(r"[a-z]+", lambda m: _NUMBER_WORDS.get(m.group(0), m.group(0)), t)
    words = re.findall(r"[a-z']+", t)
    if not words or len(words) > 14 or any(w not in _FILLER for w in words) or not (_KEYWORDS & set(words)):
        return None
    size = size_in(t)
    rest = t
    for rx in _SIZE:
        rest = rx.sub(" ", rest)
    m = _PAGES.search(rest)
    pages = _expand(m.group(1)) if m else None
    rel = None
    if not pages:
        if re.search(r"\b(next|more)\b", rest):
            rel = "next"
        elif re.search(r"\b(previous|prev|back)\b", rest):
            rel = "prev"
        elif re.search(r"\blast\b", rest):
            rel = "last"
        elif re.search(r"\bfirst\b", rest):
            rel = "first"
    if not pages and not rel and not size:
        return None
    return {"pages": pages, "rel": rel, "size": size}


# ------------------------------------------------------------------ per-conversation state (no rows)
_lock = threading.Lock()
_state: dict[str, dict] = {}


def remember(conversation_id: str | None, result: dict, qid: str, question: str) -> None:
    if not conversation_id:
        return
    with _lock:
        now = time.time()
        for k in [k for k, v in _state.items() if now - v["at"] > STATE_TTL_SECONDS]:
            del _state[k]
        _state[conversation_id] = {"sql": result["sql"], "page_size": result["page_size"], "page": result["page"],
                                   "qid": qid, "question": question, "at": now}


def recall(conversation_id: str | None) -> dict | None:
    if not conversation_id:
        return None
    with _lock:
        s = _state.get(conversation_id)
        if s and time.time() - s["at"] <= STATE_TTL_SECONDS:
            return dict(s)
    return None


def forget(conversation_id: str | None) -> None:
    with _lock:
        _state.pop(conversation_id or "", None)


# ------------------------------------------------------------------ answering a paging request
def footer(r: dict, qid: str) -> str:
    """The note under an answer whose result has more than one page."""
    return (f"_Page {r['page']} of {r['pages']} (rows {r['first_row']}–{r['last_row']} of {r['row_count']}) "
            f"[{qid}]. Ask for another page (e.g. “page 2”, “pages 2, 3”, “last page”) or a page size "
            f"(“20 per page”)._")


def _cell(v) -> str:
    if isinstance(v, float):
        v = round(v, 4)
    return str(v if v is not None else "").replace("|", "\\|").replace("\n", " ")


def render(r: dict, qid: str) -> str:
    """One page as markdown, straight from the rows (no LLM): a numbered list for one column, else a table."""
    head = (f"**Page {r['page']} of {r['pages']}** (rows {r['first_row']}–{r['last_row']} of {r['row_count']}) "
            f"[{qid}]")
    if not r["rows"]:
        return head + "\n\n(no rows)"
    n0, cols = r["first_row"], r["columns"]
    if len(cols) == 1:
        body = "\n".join(f"{n0 + i}. {_cell(row[cols[0]])}" for i, row in enumerate(r["rows"]))
    else:
        body = "\n".join(["| # | " + " | ".join(_cell(c) for c in cols) + " |",
                          "|---|" + "---|" * len(cols)]
                         + [f"| {n0 + i} | " + " | ".join(_cell(row.get(c)) for c in cols) + " |"
                            for i, row in enumerate(r["rows"])])
    return head + "\n\n" + body


def answer_command(cmd: dict, conversation_id: str) -> dict:
    """Runs a paging request against the conversation's last paged query. Returns an ask()-shaped dict."""
    t0 = time.time()
    st = recall(conversation_id)
    if not st:
        return _reply("The last answer has no more pages, so there is nothing to page through. Ask for a list "
                      "first, e.g. “List all Sci-Fi films”.", "not_found", [], t0, "no_list")
    size = cmd["size"] or st["page_size"]
    resized = size != st["page_size"]
    total = count_rows(st["sql"])
    n_pages = pages_for(total, size)
    current = 1 if resized else st["page"]
    if cmd["pages"]:
        wanted = cmd["pages"]
    elif cmd["rel"] == "next":
        wanted = [current + 1]
    elif cmd["rel"] == "prev":
        wanted = [current - 1]
    elif cmd["rel"] == "last":
        wanted = [n_pages]
    else:   # "first", or only a new page size
        wanted = [1]
    valid = [p for p in wanted if 1 <= p <= n_pages]
    invalid = [p for p in wanted if p not in valid]
    blocks, results = [], []
    for p in valid:
        r = run_paged(st["sql"], size, p, first_run=False)
        results.append(r)
        blocks.append(render(r, st["qid"]))
    if invalid:
        which = ", ".join(map(str, invalid))
        blocks.append(f"Page {which} does not exist: there {'is' if n_pages == 1 else 'are'} {n_pages} "
                      f"page{'s' if n_pages != 1 else ''} of {size} rows ({total} rows).")
    if results:
        last = results[-1]
        remember(conversation_id, last, st["qid"], st["question"])
        if last["pages"] > 1:
            blocks.append(f"_Ask for another page (1–{last['pages']}), “next page”, or a page size "
                          f"(“20 per page”)._")
    else:   # keep the new page size even if no page was valid
        with _lock:
            if conversation_id in _state:
                _state[conversation_id].update(page_size=size, at=time.time())
    status = "answered" if results else "not_found"
    return _reply("\n\n".join(blocks), status, [dict(r, id=st["qid"], type="sql") for r in results], t0,
                  "" if results else "page_out_of_range", question_for=st["question"])


def _reply(answer: str, status: str, citations: list[dict], t0: float, reason: str, question_for: str = "") -> dict:
    try:
        import mlflow
        mlflow.update_current_trace(tags={"answer_status": status, "status_reason": reason or "page_request",
                                          "paged_question": question_for[:250]})
    except Exception:
        pass
    return {"status": status, "answer": answer, "citations": citations,
            "sql": [{k: c[k] for k in ("sql", "columns", "rows", "row_count", "page", "pages", "page_size")}
                    for c in citations],
            "films": [], "plan": {"page_request": True}, "assessment": {},
            "steps": [{"node": "page", "seconds": round(time.time() - t0, 2)}]}
