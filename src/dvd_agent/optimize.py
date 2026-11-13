"""Optimize and measure the SQL step on its own (DSPy), against the eval questions' gold_sql.

The full eval grades final answers, and those pass almost always (dev 87/88, test 80/80), partly
because the self-check sends a wrong first query back for another round. This measures the first
query alone: execution accuracy, the usual text-to-SQL metric. Does the result of the SQL the model
writes on its first try contain what gold_sql returns?

    python -m dvd_agent.optimize inputs   --split dev      # analyze + film search once, saved
    python -m dvd_agent.optimize measure  --split dev      # first-try accuracy of the current program
    python -m dvd_agent.optimize compile                   # BootstrapFewShot on dev -> candidate
    python -m dvd_agent.optimize measure  --split test --program artifacts/candidate_program.json

Only expect=answer questions without "context" (follow-ups) are used. Inputs (the schema text and
the film-search hits the agent would pass) are built once per split and saved, so every program is
measured on the same inputs. compile uses dev only; test is for the gate.

The candidate is only promoted by hand, after the full eval on test agrees: copy it to
src/dvd_agent/agent/compiled_program.json, which the agent loads by default. To run the full eval
with a candidate first: COMPILED_PROGRAM_PATH=artifacts/candidate_program.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from decimal import Decimal
from pathlib import Path

import dspy

from .agent import schema, tools
from .agent.signatures import DvdQAProgram, GenerateSQL
from .eval import cases
from .eval.metrics import GoldError, gold
from .llm import configure_dspy, ensure_embed_model_loaded

ARTIFACTS = Path("artifacts")
CANDIDATE = ARTIFACTS / "candidate_program.json"
RESULT_PAGE = 500        # rows read per query when grading (gold has at most 20)
EXTRA_ROWS = 5           # a result may hold this many rows more than gold (e.g. both stores, not just the winner)


# ------------------------------------------------------------------ the metric
def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"]


def _month_name(v) -> str | None:
    """A month column's value as its name: 4, Decimal(4), '2007-04', '2007-04-01', a date or a timestamp."""
    if hasattr(v, "month"):
        return _MONTHS[v.month - 1]
    n = _num(v)
    if n is not None:
        return _MONTHS[int(n) - 1] if n == int(n) and 1 <= n <= 12 else None
    m = re.fullmatch(r"\s*\d{4}-(\d{1,2})(?:-\d{1,2}.*)?\s*", str(v))
    return _MONTHS[int(m.group(1)) - 1] if m and 1 <= int(m.group(1)) <= 12 else None


def _row_words(row: dict) -> set[str]:
    """Words of a row's values. A column named like "month" also gives its month name, so a query
    returning month 4 or '2007-04' answers "which month" as well as one returning 'April' does."""
    words = _tokens(" ".join(str(v) for v in row.values() if v is not None))
    for k, v in row.items():
        if "month" in k.lower() and v is not None and (name := _month_name(v)):
            words.add(name)
    return words


def _num(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float, Decimal)):
        return float(v)
    return None


def hits_gold(results: list[dict], gold_texts: list[str], gold_numbers: list[float],
              gold_rows: int, tol: float = 0.005) -> tuple[bool, str]:
    """(passed, reason). Every gold number must be a cell value (within tol), and every gold text's
    words must all be in one row, in results with at most gold_rows + EXTRA_ROWS rows: a query that
    returns every customer contains the top 5, but doesn't answer "who are the top 5"."""
    ok = [r for r in results if "error" not in r and r.get("row_count", 0) <= gold_rows + EXTRA_ROWS]
    if not ok:
        errors = [r["error"][:120] for r in results if "error" in r]
        too_big = [r["row_count"] for r in results if "error" not in r]
        return False, ("error: " + "; ".join(errors)) if errors else f"no result small enough (rows {too_big})"
    numbers = [n for r in ok for row in r["rows"] for v in row.values() if (n := _num(v)) is not None]
    row_words = [_row_words(row) for r in ok for row in r["rows"]]
    missing_num = [g for g in dict.fromkeys(gold_numbers)
                   if not any(abs(n - g) <= max(abs(g) * tol, 0.01) for n in numbers)]
    missing_text = [t for t in dict.fromkeys(gold_texts)
                    if not any(_tokens(t) <= w for w in row_words)]
    if missing_num or missing_text:
        parts = []
        if missing_text:
            parts.append("missing: " + ", ".join(missing_text[:5]))
        if missing_num:
            parts.append("numbers: " + ", ".join(f"{n:g}" for n in missing_num[:5]))
        return False, "; ".join(parts)
    return True, ""


def run_queries(queries: list[str], question: str) -> list[dict]:
    return [tools.run_sql(q, question, RESULT_PAGE) for q in queries if q and q.strip()][:3]


def metric(example: dspy.Example, pred, trace=None) -> bool:
    results = run_queries(list(pred.queries or []), example.question)
    return hits_gold(results, example.gold_texts, example.gold_numbers, example.gold_rows, example.tol)[0]


# ------------------------------------------------------------------ inputs, built once per split
def _inputs_path(split: str) -> Path:
    return ARTIFACTS / f"sql-inputs-{split}.json"


def build_inputs(split: str) -> list[dict]:
    """The SQL step's inputs as the agent builds them (analyze, then film search if planned), plus
    the gold values. Saved so every program sees the same inputs."""
    ensure_embed_model_loaded()
    program = DvdQAProgram()
    out = []
    rows = [r for r in cases.load(split) if r["expect"] == "answer" and not r.get("context")]
    for i, r in enumerate(rows, 1):
        try:
            g = gold(r)
        except GoldError as e:
            raise SystemExit(f"{r.get('id')}: {e}") from e
        p = program.analyze(question=r["question"])
        films = tools.search_films((p.film_search_queries or [r["question"]])[:2]) if p.needs_film_search else []
        out.append({"id": r.get("id"), "question": r["question"], "db_schema": schema.describe(r["question"]),
                    "film_hits": tools.format_film_hits(films), "gold_texts": g["texts"],
                    "gold_numbers": g["numbers"], "gold_rows": len(g["rows"]),
                    "tol": float(r.get("tolerance", 0.005))})
        print(f"{i}/{len(rows)} {r.get('id')} film_search={bool(films)}", flush=True)
    ARTIFACTS.mkdir(exist_ok=True)
    _inputs_path(split).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def load_inputs(split: str) -> list[dict]:
    p = _inputs_path(split)
    if not p.exists():
        raise SystemExit(f"{p} not found: run `python -m dvd_agent.optimize inputs --split {split}` first")
    return json.loads(p.read_text(encoding="utf-8"))


def examples(items: list[dict]) -> list[dspy.Example]:
    return [dspy.Example(**{k: v for k, v in it.items() if k != "id"}, previous_attempts="")
            .with_inputs("question", "db_schema", "film_hits", "previous_attempts") for it in items]


# ------------------------------------------------------------------ the SQL step alone
class SQLStep(dspy.Module):
    def __init__(self):
        super().__init__()
        self.sql = dspy.ChainOfThought(GenerateSQL)

    def forward(self, question, db_schema, film_hits, previous_attempts=""):
        return self.sql(question=question, db_schema=db_schema, film_hits=film_hits,
                        previous_attempts=previous_attempts)


def load_step(program_path: str | None) -> SQLStep:
    step = SQLStep()
    if program_path:
        program = DvdQAProgram()
        program.load(program_path)
        step.sql = program.sql
    return step


def measure(split: str, program_path: str | None, label: str) -> dict:
    step = load_step(program_path)
    items = load_inputs(split)
    rows, t_all = [], time.time()
    for ex, it in zip(examples(items), items, strict=True):
        t0 = time.time()
        try:
            pred = step(**ex.inputs())
            queries = list(pred.queries or [])
            ok, why = hits_gold(run_queries(queries, ex.question), ex.gold_texts, ex.gold_numbers,
                                ex.gold_rows, ex.tol)
        except Exception as e:   # a broken model reply counts as a miss, the run goes on
            queries, ok, why = [], False, f"{type(e).__name__}: {str(e)[:200]}"
        secs = time.time() - t0
        rows.append({"id": it["id"], "passed": ok, "reason": why, "queries": queries, "seconds": round(secs, 1)})
        print(f"{it['id']} {'ok ' if ok else 'MISS'} {secs:5.1f}s {why}", flush=True)
    n = len(rows)
    summary = {"split": split, "program": program_path or "(uncompiled)", "questions": n,
               "first_try_accuracy": round(sum(r["passed"] for r in rows) / n, 3),
               "median_seconds": sorted(r["seconds"] for r in rows)[n // 2],
               "minutes": round((time.time() - t_all) / 60, 1)}
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"sql-{label}-{split}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def compile_candidate(demos: int, seed: int) -> Path:
    """BootstrapFewShot on dev: run the SQL step on dev questions (shuffled), keep up to `demos`
    attempts that hit gold, as worked examples in the prompt. Saves the whole program as a candidate."""
    trainset = examples(load_inputs("dev"))
    random.Random(seed).shuffle(trainset)
    optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=demos, max_labeled_demos=0,
                                      max_rounds=1)
    compiled = optimizer.compile(SQLStep(), trainset=trainset)
    program = DvdQAProgram()
    program.sql = compiled.sql
    ARTIFACTS.mkdir(exist_ok=True)
    program.save(str(CANDIDATE))
    demos = sum(len(pred.demos) for _, pred in compiled.named_predictors())
    print(f"saved {CANDIDATE} with {demos} demos", flush=True)
    return CANDIDATE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["inputs", "measure", "compile"])
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--program", help="saved program to measure (default: the uncompiled one)")
    ap.add_argument("--label", help="name for the results file (default: baseline or candidate)")
    ap.add_argument("--demos", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_dspy()
    if a.command == "inputs":
        build_inputs(a.split)
    elif a.command == "measure":
        measure(a.split, a.program, a.label or ("candidate" if a.program else "baseline"))
    else:
        compile_candidate(a.demos, a.seed)


if __name__ == "__main__":
    main()
