"""Eval questions: data/eval/dev.jsonl and data/eval/test.jsonl, one line per question.

dev   tune against it freely (its scores are optimistic)
test  held out: run it only to report a score. If a test failure makes you change code,
      move that question to dev and write a new one.

Line format (see data/eval/README.md):
  question    required
  expect      required: "answer" | "not_found" | "correction"
  gold_sql    expect=answer: SQL whose result the answer must contain (run at eval time, so it follows the data)
  text_col    column(s) of gold_sql whose values (every row) must appear in the answer
  number_col  column(s) of gold_sql whose values (every row) must appear as numbers in the answer
  tolerance   relative tolerance for numbers (default 0.005)
  contains    expect=correction: any one of these phrases shows the premise was corrected
  id, why     notes for you; not used for scoring
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("data/eval")
EXPECTS = {"answer", "not_found", "correction"}
MAX_GOLD_ROWS = 20


def path(split: str) -> Path:
    return ROOT / f"{split}.jsonl"


def norm(q: str) -> str:
    return " ".join(q.lower().split()).rstrip("?")


def _as_list(v):
    return [v] if isinstance(v, str) else list(v or [])


def load(split: str) -> list[dict]:
    """Rows of a split, validated. Raises SystemExit listing every problem."""
    p = path(split)
    if not p.exists():
        raise SystemExit(f"{p} not found")
    rows, problems, seen = [], [], {}
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{p}:{i}"
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"{where} invalid JSON ({e})")
            continue
        problems += [f"{where} {msg}" for msg in _check_row(r)]
        if isinstance(r.get("question"), str):
            k = norm(r["question"])
            if k in seen:
                problems.append(f"{where} same question as line {seen[k]}")
            seen.setdefault(k, i)
        for f in ("text_col", "number_col", "contains"):
            if f in r:
                r[f] = _as_list(r[f])
        r["line"] = i
        rows.append(r)
    if split == "test":
        dev = {norm(r["question"]) for r in load("dev")} if path("dev").exists() else set()
        problems += [f"{p}:{r['line']} also in dev.jsonl: {r['question']!r}" for r in rows
                     if isinstance(r.get("question"), str) and norm(r["question"]) in dev]
    if problems:
        raise SystemExit(f"{len(problems)} problem(s) in {p}:\n  " + "\n  ".join(problems))
    if not rows:
        raise SystemExit(f"{p} has no questions (format: data/eval/README.md)")
    return rows


def _check_row(r: dict) -> list[str]:
    out = []
    if not isinstance(r.get("question"), str) or not r["question"].strip():
        out.append("missing 'question'")
    e = r.get("expect")
    if e not in EXPECTS:
        out.append(f"'expect' must be one of {sorted(EXPECTS)}, got {e!r}")
    if e == "answer":
        if not r.get("gold_sql"):
            out.append("expect=answer needs 'gold_sql'")
        if not (r.get("text_col") or r.get("number_col")):
            out.append("expect=answer needs 'text_col' and/or 'number_col'")
    elif any(k in r for k in ("gold_sql", "text_col", "number_col")):
        out.append("'gold_sql', 'text_col' and 'number_col' only go with expect=answer")
    if e == "correction" and not r.get("contains"):
        out.append("expect=correction needs 'contains' (words that show the premise was corrected)")
    if e == "not_found" and "contains" in r:
        out.append("expect=not_found takes no 'contains'")
    return out
