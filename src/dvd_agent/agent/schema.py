"""Schema description for the SQL step, read live from the database (so new tables/columns appear).

Cached for SCHEMA_CACHE_SECONDS. Columns that are never needed (password hash, picture bytes, the
tsvector) are hidden and blocked; personal contact columns are only shown when the question asks for
them (see `describe`).
"""
from __future__ import annotations

import re
import threading
import time

from ..config import get_settings
from ..db import postgres as pg

INTERNAL_TABLES = {"film_embedding"}                       # the agent's own data, not the business schema
HIDDEN_COLUMNS = {("staff", "password"), ("staff", "picture"), ("film", "fulltext")}
CONTACT_COLUMNS = {("customer", "email"), ("staff", "email"), ("address", "address"), ("address", "address2"),
                   ("address", "phone"), ("address", "postal_code"), ("customer_list", "address"),
                   ("customer_list", "phone"), ("customer_list", "zip code"), ("staff_list", "address"),
                   ("staff_list", "phone"), ("staff_list", "zip code")}
_CONTACT_WORDS = re.compile(r"\b(e-?mails?|phones?|addresse?s?|street|postal|zip|contact)\b", re.I)
_VALUE_LISTS = [("category", "name"), ("language", "name"), ("country", "country")]

_lock = threading.Lock()
_cache: dict = {"at": 0.0, "data": None}


def _load() -> dict:
    with _lock:
        if _cache["data"] is None or time.time() - _cache["at"] > get_settings().schema_cache_seconds:
            raw = pg.schema_rows()
            values = {}
            for t, c in _VALUE_LISTS:
                try:
                    v = pg.distinct_values(t, c, 30)
                    values[(t, c)] = v if len(v) <= 30 else None   # long lists (109 countries) are not inlined
                except Exception:
                    pass
            ranges = {}
            for t, c in (("rental", "rental_date"), ("payment", "payment_date")):
                try:
                    with pg.conn() as cn:
                        r = cn.execute(f"SELECT min({c})::date AS lo, max({c})::date AS hi FROM {t}").fetchone()
                    ranges[(t, c)] = (r["lo"], r["hi"])
                except Exception:
                    pass
            _cache.update(at=time.time(), data={**raw, "values": values, "ranges": ranges})
        return _cache["data"]


def blocked_columns(question: str = "") -> set[tuple[str, str]]:
    """Columns the guard rejects for this question."""
    return HIDDEN_COLUMNS | (set() if _CONTACT_WORDS.search(question) else CONTACT_COLUMNS)


def table_names() -> set[str]:
    return {r["name"] for r in _load()["relations"] if r["name"] not in INTERNAL_TABLES}


def describe(question: str = "") -> str:
    """Compact schema text: one block per table/view with columns, keys and known values."""
    d = _load()
    blocked = blocked_columns(question)
    cols: dict[str, list[dict]] = {}
    for c in d["columns"]:
        cols.setdefault(c["table_name"], []).append(c)
    pks, fks = {}, {}
    for k in d["keys"]:
        if k["constraint_type"] == "PRIMARY KEY":
            pks.setdefault(k["table_name"], set()).add(k["column_name"])
        else:
            fks.setdefault(k["table_name"], {})[k["column_name"]] = f"{k['ref_table']}.{k['ref_column']}"
    out = []
    for rel in d["relations"]:
        t = rel["name"]
        if t in INTERNAL_TABLES:
            continue
        parts = []
        for c in cols.get(t, []):
            name = c["column_name"]
            if (t, name) in blocked:
                continue
            typ = c["udt_name"] if c["data_type"] in ("USER-DEFINED", "ARRAY") else c["data_type"]
            typ = {"character varying": "varchar", "timestamp without time zone": "timestamp",
                   "_text": "text[]"}.get(typ, typ)
            s = f"{name} {typ}"
            if name in pks.get(t, ()):
                s += " PK"
            if name in fks.get(t, {}):
                s += f" -> {fks[t][name]}"
            if typ in d["enums"]:
                s += f" ({', '.join(d['enums'][typ])})"
            if d["values"].get((t, name)):
                s += f" (values: {', '.join(map(str, d['values'][(t, name)]))})"
            if (t, name) in d["ranges"]:
                lo, hi = d["ranges"][(t, name)]
                s += f" (from {lo} to {hi})"
            parts.append(s)
        rows = f", ~{rel['est_rows']} rows" if rel["kind"] == "table" and rel["est_rows"] > 0 else ""
        out.append(f"{rel['kind']} {t}{rows}: {', '.join(parts)}")
    return "\n".join(out)
