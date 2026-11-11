"""PostgreSQL access.

Two pools: the owner role (embedding sync, hybrid film search) and dvd_reader, which runs every SQL
statement the agent writes. dvd_reader has SELECT only, read-only transactions and a 10 s statement
timeout (sql/init.sql); the database enforces that, whatever the SQL guard lets through.
"""
from __future__ import annotations

import decimal
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Sequence

import numpy as np
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ..config import get_settings


def _register_vector(c) -> None:
    try:
        register_vector(c)
    except Exception:  # pgvector extension not visible yet
        pass


@lru_cache
def get_pool() -> ConnectionPool:
    return ConnectionPool(get_settings().postgres_dsn, min_size=1, max_size=4,
                          kwargs={"row_factory": dict_row, "autocommit": True},
                          configure=_register_vector, open=True)


@lru_cache
def get_reader_pool() -> ConnectionPool:
    return ConnectionPool(get_settings().postgres_reader_dsn, min_size=1, max_size=8,
                          kwargs={"row_factory": dict_row, "autocommit": True}, open=True)


@contextmanager
def conn():
    with get_pool().connection() as c:
        yield c


@contextmanager
def reader():
    with get_reader_pool().connection() as c:
        yield c


# ---------------------------------------------------------------- agent SQL
@dataclass
class SqlResult:
    sql: str
    columns: list[str]
    rows: list[dict]
    row_count: int          # rows returned


def _jsonable(v):
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat(sep=" ") if isinstance(v, datetime) else v.isoformat()
    if isinstance(v, (bytes, memoryview)):
        return "<binary>"
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    return v


def run_readonly(sql: str) -> SqlResult:
    """Runs one guarded SELECT as dvd_reader. Raises psycopg errors (the agent shows them to the LLM).
    Fetches every row the SQL returns: callers page it with LIMIT/OFFSET (agent/paging.py)."""
    with reader() as c:
        cur = c.execute(sql)
        cols = [d.name for d in cur.description or []]
        rows = [{k: _jsonable(v) for k, v in r.items()} for r in cur.fetchall()]
    return SqlResult(sql, cols, rows, len(rows))


# ---------------------------------------------------------------- schema (read live)
def schema_rows() -> dict:
    """Tables/views of the public schema that do not belong to an extension (PostGIS, pg_stat_statements)
    and that dvd_reader may read, with columns (readable = dvd_reader has SELECT on it), PKs and FKs."""
    with conn() as c:
        rels = c.execute("""
            SELECT cl.relname AS name, CASE cl.relkind WHEN 'v' THEN 'view' ELSE 'table' END AS kind,
                   cl.reltuples::bigint AS est_rows
            FROM pg_class cl JOIN pg_namespace n ON n.oid = cl.relnamespace
            WHERE n.nspname = 'public' AND cl.relkind IN ('r', 'v', 'p')
              AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.objid = cl.oid AND d.deptype = 'e')
              AND has_any_column_privilege('dvd_reader', cl.oid, 'SELECT')   -- also tables with column-only grants
            ORDER BY cl.relname""").fetchall()
        cols = c.execute("""
            SELECT table_name, column_name, data_type, udt_name,
                   has_column_privilege('dvd_reader', format('%I.%I', table_schema, table_name),
                                        column_name, 'SELECT') AS readable
            FROM information_schema.columns WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position""").fetchall()
        fks = c.execute("""
            SELECT tc.table_name, kcu.column_name, ccu.table_name AS ref_table, ccu.column_name AS ref_column,
                   tc.constraint_type
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
            WHERE tc.table_schema = 'public' AND tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')""").fetchall()
        enums = c.execute("""
            SELECT t.typname, array_agg(e.enumlabel ORDER BY e.enumsortorder) AS labels
            FROM pg_type t JOIN pg_enum e ON e.enumtypid = t.oid GROUP BY t.typname""").fetchall()
    return {"relations": rels, "columns": cols, "keys": fks, "enums": {e["typname"]: e["labels"] for e in enums}}


def distinct_values(table: str, column: str, max_values: int = 30) -> list:
    """Small value lists (category names, languages) so the LLM writes exact WHERE values."""
    from psycopg import sql as psql
    q = psql.SQL("SELECT DISTINCT {c} AS v FROM {t} ORDER BY 1 LIMIT %s").format(
        c=psql.Identifier(column), t=psql.Identifier(table))
    with conn() as c:
        return [r["v"].rstrip() if isinstance(r["v"], str) else r["v"]   # language.name is char(20)
                for r in c.execute(q, (max_values + 1,)).fetchall()]


# ---------------------------------------------------------------- film search (hybrid)
@dataclass
class FilmHit:
    film_id: int
    title: str
    description: str
    release_year: int | None
    rating: str | None
    score: float
    bm25_rank: int | None
    vec_rank: int | None


def hybrid_film_search(query: str, embedding: list[float], k: int = 10, candidates: int = 40,
                       rrf_k: int = 60) -> list[FilmHit]:
    """Reciprocal-rank fusion of pg_search BM25 (title + description) and pgvector cosine search."""
    params = {"q": query, "emb": np.asarray(embedding, dtype=np.float32), "n": candidates, "k": k, "rrf": rrf_k}
    sql = """
    WITH bm25 AS (
        SELECT film_id, ROW_NUMBER() OVER (ORDER BY pdb.score(film_id) DESC) AS r
        FROM film WHERE title ||| %(q)s OR description ||| %(q)s
        ORDER BY pdb.score(film_id) DESC LIMIT %(n)s
    ),
    vec AS (
        SELECT film_id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(emb)s) AS r
        FROM film_embedding ORDER BY embedding <=> %(emb)s LIMIT %(n)s
    ),
    fused AS (
        SELECT COALESCE(b.film_id, v.film_id) AS film_id, b.r AS bm25_rank, v.r AS vec_rank,
               COALESCE(1.0 / (%(rrf)s + b.r), 0) + COALESCE(1.0 / (%(rrf)s + v.r), 0) AS score
        FROM bm25 b FULL OUTER JOIN vec v ON b.film_id = v.film_id
    )
    SELECT f.film_id, f.title, f.description, f.release_year::int AS release_year, f.rating::text AS rating,
           x.score::float AS score, x.bm25_rank, x.vec_rank
    FROM fused x JOIN film f ON f.film_id = x.film_id
    ORDER BY x.score DESC LIMIT %(k)s
    """
    with conn() as c:
        return [FilmHit(**r) for r in c.execute(sql, params).fetchall()]


# ---------------------------------------------------------------- embedding sync
def film_texts() -> list[dict]:
    """Every film with the md5 of the text that gets embedded, and the stored hash (None if missing)."""
    with conn() as c:
        return c.execute("""
            SELECT f.film_id, f.title || '. ' || COALESCE(f.description, '') AS text,
                   md5(f.title || '. ' || COALESCE(f.description, '')) AS hash, e.text_hash AS stored_hash
            FROM film f LEFT JOIN film_embedding e ON e.film_id = f.film_id
            ORDER BY f.film_id""").fetchall()


def upsert_embeddings(rows: Sequence[dict]) -> int:
    """rows: film_id, hash, embedding"""
    with conn() as c, c.transaction(), c.cursor() as cur:
        cur.executemany(
            """INSERT INTO film_embedding (film_id, text_hash, embedding) VALUES (%s, %s, %s)
               ON CONFLICT (film_id) DO UPDATE SET text_hash = EXCLUDED.text_hash,
                   embedding = EXCLUDED.embedding, embedded_at = now()""",
            [(r["film_id"], r["hash"], np.asarray(r["embedding"], dtype=np.float32)) for r in rows])
    return len(rows)


def embedding_stats() -> dict:
    with conn() as c:
        return c.execute("""
            SELECT (SELECT count(*) FROM film) AS films,
                   (SELECT count(*) FROM film_embedding) AS with_embedding,
                   (SELECT count(*) FROM film f JOIN film_embedding e USING (film_id)
                     WHERE e.text_hash <> md5(f.title || '. ' || COALESCE(f.description, ''))) AS stale""").fetchone()
