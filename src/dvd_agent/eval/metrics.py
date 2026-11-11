"""Pass/fail rules for one eval answer. Expected values come from running the line's gold_sql."""
from __future__ import annotations

import re

from ..agent.tools import says_not_in_sources  # same detector the agent's finalize step uses
from ..db import postgres as pg
from .cases import MAX_GOLD_ROWS

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


class GoldError(Exception):
    pass


def gold(row: dict) -> dict:
    """Runs gold_sql as dvd_reader. Returns {"texts": [...], "numbers": [...], "rows": [...]}."""
    try:
        r = pg.run_readonly(row["gold_sql"], MAX_GOLD_ROWS + 1)
    except Exception as e:
        raise GoldError(f"gold_sql failed: {e}") from e
    if not r.rows:
        raise GoldError("gold_sql returned no rows")
    if len(r.rows) > MAX_GOLD_ROWS:
        raise GoldError(f"gold_sql returned more than {MAX_GOLD_ROWS} rows; narrow the question")
    missing = [c for c in row.get("text_col", []) + row.get("number_col", []) if c not in r.columns]
    if missing:
        raise GoldError(f"gold_sql has no column(s) {missing}; columns: {r.columns}")
    return {"texts": [str(x[c]).strip() for x in r.rows for c in row.get("text_col", [])],
            "numbers": [float(x[c]) for x in r.rows for c in row.get("number_col", []) if x[c] is not None],
            "rows": r.rows}


def numbers_in(text: str) -> list[float]:
    out = []
    for num in _NUM.findall(text.replace("−", "-")):
        try:
            out.append(float(num.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def numeric_match(answer: str, expected: float, rel_tol: float = 0.005) -> bool:
    """Any number in the answer within rel_tol of expected (or within 0.01 for small values)."""
    return any(abs(v - expected) <= max(abs(expected) * rel_tol, 0.01) for v in numbers_in(answer))


def has_citation(answer: str) -> bool:
    return bool(re.search(r"\[[QS]\d+", answer))


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def check(row: dict, answer: str, status: str) -> tuple[bool, str]:
    """(passed, reason). status is the agent's: answered | partial | not_found."""
    expect, refused = row["expect"], status == "not_found"
    if expect == "not_found":
        if refused or says_not_in_sources(answer):
            return True, ""
        return False, "made up an answer (should say not found)"
    if expect == "correction":
        if refused or any(_norm(k) in _norm(answer) for k in row["contains"]):
            return True, ""
        return False, "accepted the false premise (no correction word: " + ", ".join(row["contains"]) + ")"
    # expect == "answer"
    if refused:
        return False, "said not found (should answer)"
    g = gold(row)
    reasons = []
    missing_text = [t for t in dict.fromkeys(g["texts"]) if _norm(t) not in _norm(answer)]
    if missing_text:
        reasons.append("missing: " + ", ".join(missing_text[:8]) + (" ..." if len(missing_text) > 8 else ""))
    tol = float(row.get("tolerance", 0.005))
    missing_num = [n for n in dict.fromkeys(g["numbers"]) if not numeric_match(answer, n, tol)]
    if missing_num:
        reasons.append("numbers not in answer: " + ", ".join(f"{n:,.2f}" for n in missing_num[:8]))
    return not reasons, "; ".join(reasons)
