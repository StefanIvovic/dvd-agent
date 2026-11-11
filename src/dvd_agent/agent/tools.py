"""Agent tools: guarded SQL, hybrid film search (LlamaIndex retriever), source formatting."""
from __future__ import annotations

import json
import re
from typing import Sequence

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from ..config import get_settings
from ..db import postgres as pg
from ..llm import embed_query
from . import paging, schema, sql_guard


# ------------------------------------------------------------------ film search
class FilmRetriever(BaseRetriever):
    """LlamaIndex retriever: pg_search BM25 + pgvector over film title/description, fused with RRF in SQL."""

    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        q = query_bundle.query_str
        return [NodeWithScore(node=TextNode(id_=str(h.film_id), text=h.description, metadata={
                    "film_id": h.film_id, "title": h.title, "release_year": h.release_year, "rating": h.rating,
                    "bm25_rank": h.bm25_rank, "vec_rank": h.vec_rank}), score=h.score)
                for h in pg.hybrid_film_search(q, embed_query(q), k=self.k)]


def search_films(queries: Sequence[str], k: int | None = None) -> list[dict]:
    """Runs each query, merges by film_id (best score wins). Films added since the last sync are embedded first."""
    k = k or get_settings().film_search_k
    try:
        from ..sync.flows import sync
        sync(max_rows=50)   # lazy top-up; the scheduled flow does the bulk
    except Exception:
        pass
    found: dict[int, dict] = {}
    for q in queries:
        for n in FilmRetriever(k).retrieve(q):
            m = n.node.metadata
            if m["film_id"] not in found or n.score > found[m["film_id"]]["score"]:
                found[m["film_id"]] = {**m, "description": n.node.get_content(), "score": n.score, "query": q}
    return sorted(found.values(), key=lambda h: h["score"], reverse=True)[:k]


# ------------------------------------------------------------------ SQL
def run_sql(sql: str, question: str, page_size: int | None = None) -> dict:
    """Guard + run, page 1. Returns paging.run_paged's dict (rows of page 1, row_count = all rows,
    page, pages, page_size) or {"sql", "error"}. page_size: PAGE_SIZE unless the user asked for one."""
    s = get_settings()
    try:
        safe = sql_guard.check(sql, schema.table_names(), schema.blocked_columns(question))
    except sql_guard.UnsafeSQL as e:
        return {"sql": sql_guard.clean(sql), "error": str(e)}
    try:
        return paging.run_paged(safe, page_size or s.page_size)
    except Exception as e:  # syntax/semantic errors, timeout: shown to the LLM for a retry
        msg = str(e).strip().splitlines()
        return {"sql": safe, "error": f"{type(e).__name__}: {' '.join(msg[:3])}"[:500]}


# ------------------------------------------------------------------ formatting sources for the LLM
def _table(columns: list[str], rows: list[dict]) -> str:
    def cell(v):
        if isinstance(v, float):
            v = round(v, 4)
        return str(v).replace("|", "/").replace("\n", " ")[:200]
    lines = [" | ".join(columns)]
    lines += [" | ".join(cell(r.get(c)) for c in columns) for r in rows]
    return "\n".join(lines)


def format_sources(results: list[dict], films: list[dict]) -> tuple[str, dict]:
    """Returns (prompt text, citation map id -> metadata). [Q#] = a SQL result, [S#] = a film search hit."""
    parts, cmap = [], {}
    for i, r in enumerate([r for r in results if "error" not in r], 1):
        qid = f"Q{i}"
        cmap[qid] = {"type": "sql", **{k: r[k] for k in ("sql", "columns", "rows", "row_count", "page", "pages",
                                                          "page_size", "first_row", "last_row")}}
        if r["pages"] > 1:
            info = (f"{r['row_count']} rows in {r['pages']} pages of {r['page_size']}; page 1 shown "
                    f"(rows 1-{r['last_row']}). The user sees a note with the page count and can ask for "
                    f"any other page, so list the rows of this page; the other pages are not missing.")
        else:
            info = f"{r['row_count']} rows"
        parts.append(f"[{qid}] SQL: {r['sql']}\nResult: {info}\n"
                     f"{_table(r['columns'], r['rows']) if r['rows'] else '(no rows)'}")
    for i, f in enumerate(films, 1):
        sid = f"S{i}"
        cmap[sid] = {"type": "film", "film_id": f["film_id"], "title": f["title"], "description": f["description"]}
        parts.append(f"[{sid}] film_id {f['film_id']}: {f['title']} ({f['release_year']}, {f['rating']}) - "
                     f"{f['description']}")
    return "\n\n".join(parts) if parts else "(no sources found)", cmap


def format_film_hits(films: list[dict]) -> str:
    """Film search hits passed to the SQL step, so it can filter by film_id."""
    return json.dumps([{"film_id": f["film_id"], "title": f["title"], "description": f["description"]}
                       for f in films]) if films else "[]"


# ------------------------------------------------------------------ "the data does not contain it" detection
# (copied from sec-10k-agent: only the opening paragraph counts, concessive clauses are skipped)
_SRC = r"(provided |available |given )?(sources?|data(base)?|results?|tables?|records?|schema|query results?)"
_NOT_IN_SOURCES = re.compile(
    rf"{_SRC} (do(es)? not|don't|did not) (contain|include|provide|specify|record|track|store|have|show|list)"
    rf"|not (recorded|tracked|stored|available|provided|specified|included|found) in (the |any )?{_SRC}"
    r"|\bno (specific )?(mention|information|data|records?|column)s? (of|about|on|for|regarding)\b"
    r"|\b(couldn't|could not|cannot|can't|unable to) (find|determine|answer)\b|not giving an answer"
    r"|\b(does not|doesn't|did not|do not) (record|track|store|include)\b",
    re.I)
_SENTENCE_END = re.compile(r"[.!?](?=\s)|\n")
_CONCESSIVE = re.compile(r"(while|although|though|even though|even if)\b", re.I)


def _sentence_around(clean: str, m: re.Match) -> str:
    bounds = [b.end() for b in _SENTENCE_END.finditer(clean)]  # "." followed by space, so "$4.99" stays whole
    start = max([b for b in bounds if b <= m.start()], default=0)
    end = min([b for b in bounds if b > m.end()], default=len(clean))
    return clean[start:end].strip()


def not_in_sources_match(text: str) -> str:
    clean = re.sub(r"[*_`]", "", text)
    m = _NOT_IN_SOURCES.search(clean)
    return _sentence_around(clean, m) if m else ""


def says_not_in_sources(text: str) -> bool:
    return bool(not_in_sources_match(text))


def lead_not_in_sources_match(text: str) -> str:
    """Only where the answer's opening says the requested information is missing (what makes it 'partial')."""
    clean = re.sub(r"[*_`]", "", text).strip()
    lead = re.split(r"\n\s*\n", clean, maxsplit=1)[0]
    for m in _NOT_IN_SOURCES.finditer(lead):
        sentence = _sentence_around(lead, m)
        if _CONCESSIVE.match(sentence) and "," in sentence[sentence.find(m.group(0)):]:
            continue
        return sentence
    return ""
