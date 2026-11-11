"""Pass/fail rules for one eval answer. Expected values come from running the line's gold_sql."""
from __future__ import annotations

import math
import re

from ..agent.tools import says_not_in_sources  # same detector the agent's finalize step uses
from ..db import postgres as pg
from .cases import MAX_GOLD_ROWS, PAGE_SIZE

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


class GoldError(Exception):
    pass


def gold(row: dict) -> dict:
    """Runs gold_sql as dvd_reader. Returns {"texts": [...], "numbers": [...], "rows": [...]}."""
    try:
        r = pg.run_readonly(row["gold_sql"])
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


# ------------------------------------------------------------------ expect=pages
def gold_list(row: dict) -> list[str]:
    """Every value of text_col in the full gold list (no row maximum: the list is what is being paged)."""
    try:
        r = pg.run_readonly(row["gold_sql"])
    except Exception as e:
        raise GoldError(f"gold_sql failed: {e}") from e
    if not r.rows:
        raise GoldError("gold_sql returned no rows")
    if row["text_col"] not in r.columns:
        raise GoldError(f"gold_sql has no column {row['text_col']!r}; columns: {r.columns}")
    return [str(x[row["text_col"]]).strip() for x in r.rows]


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", s.lower()))


def _row_value(row: dict, gold_tokens: dict[str, set]) -> str | None:
    """The gold value a result row stands for (its words appear among the row's cells, e.g. "Mary Smith"
    for first_name | last_name), or None."""
    cells = _tokens(" ".join(str(v) for v in row.values()))
    hits = [g for g, t in gold_tokens.items() if t and t <= cells]
    return max(hits, key=lambda g: len(gold_tokens[g])) if hits else None


def _in_text(value: str, text: str) -> bool:
    """value appears on one line of the text (works for numbered lists and markdown tables)."""
    t = _tokens(value)
    return any(t <= _tokens(line) for line in text.splitlines())


def check_pages(row: dict, turns: list[dict]) -> tuple[bool, str]:
    """turns: ask() results for the question and each follow_up, in order.

    Page 1 (the agent's answer) and every requested page must: say "Page p of P" and "of N" (when there is
    more than one page), hold the right number of rows, only rows of the gold list, and pages of the same
    size must not overlap. If the pages seen cover a whole size, together they must be the gold list.
    """
    gold_vals = gold_list(row)
    n = len(gold_vals)
    gold_tokens = {g: _tokens(g) for g in gold_vals}
    reasons, seen = [], {}   # seen[size][page] = values

    def check_page(text: str, cite: dict | None, page: int, size: int, label: str) -> None:
        n_pages = max(1, math.ceil(n / size))
        want = max(0, min(size, n - (page - 1) * size))
        if n_pages > 1 and not re.search(rf"Page\W*{page}\W*of\W*{n_pages}(?!\d)", text):
            reasons.append(f"{label}: no 'Page {page} of {n_pages}'")
        if n_pages > 1 and not re.search(rf"of {n}(?!\d)", text):
            reasons.append(f"{label}: total {n} not stated")
        if n_pages == 1 and re.search(r"Page\W*1\W*of", text):
            reasons.append(f"{label}: paging shown for a one-page list")
        if cite is None:
            reasons.append(f"{label}: no result for page {page}")
            return
        vals = [_row_value(r, gold_tokens) for r in cite["rows"]]
        if len(vals) != want:
            reasons.append(f"{label}: {len(vals)} rows on page {page}, expected {want}")
        wrong = sum(v is None for v in vals)
        if wrong:
            reasons.append(f"{label}: {wrong} rows not in the gold list")
        missing = [v for v in vals if v and not _in_text(v, text)]
        if missing:
            reasons.append(f"{label}: rows not in the answer text: {', '.join(missing[:5])}")
        seen.setdefault(size, {})[page] = [v for v in vals if v]

    first = turns[0]
    if first["status"] == "not_found":
        return False, "said not found (should list)"
    size = int(row.get("size", PAGE_SIZE))
    paged = [c for c in first["citations"] if c.get("type") == "sql" and c.get("pages", 1) > 1]
    if math.ceil(n / size) > 1:
        check_page(first["answer"], paged[0] if paged else None, 1, size, "answer")
    else:   # one page: the whole list must be in the answer, without paging
        missing = [g for g in gold_vals if not _in_text(g, first["answer"])]
        if missing:
            reasons.append(f"answer: missing {', '.join(missing[:5])}" + (" ..." if len(missing) > 5 else ""))
        if paged or re.search(r"Page\W*1\W*of", first["answer"]):
            reasons.append("answer: paging shown for a one-page list")
    for f, res in zip(row.get("follow_ups", []), turns[1:]):
        size = int(f.get("size", size))
        label = repr(f["say"])
        if res["status"] == "not_found":
            reasons.append(f"{label}: no page returned ({res['answer'][:80]})")
            continue
        cites = [c for c in res["citations"] if c.get("type") == "sql"]
        for i, p in enumerate(f["pages"]):
            block = res["answer"].split("**Page")[i + 1] if res["answer"].count("**Page") > i else res["answer"]
            check_page("**Page" + block, cites[i] if i < len(cites) else None, p, size, label)
    for size, pages in seen.items():
        flat = [v for vs in pages.values() for v in vs]
        if len(flat) != len(set(flat)):
            reasons.append(f"pages of {size} overlap")
        if set(pages) == set(range(1, max(1, math.ceil(n / size)) + 1)) and sorted(flat) != sorted(gold_vals):
            reasons.append(f"all pages of {size} together are not the gold list")
    return not reasons, "; ".join(dict.fromkeys(reasons))
