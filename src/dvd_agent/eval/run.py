"""Run the agent over an eval split and log the results to MLflow.

python -m dvd_agent.eval.run --split dev
python -m dvd_agent.eval.run --split test --run-name test-v1
python -m dvd_agent.eval.run --split dev --expect not_found --limit 5

Each question is answered once (not_found / correction questions --repeats times, default 3, to
check the agent refuses reliably; --repeat-all repeats every question). Expected values come from each line's gold_sql, run at eval time.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid

import mlflow
import pandas as pd

from ..agent.graph import ask, get_agent
from ..config import get_settings
from ..llm import ensure_embed_model_loaded
from ..tracing import init_mlflow
from . import cases
from .metrics import GoldError, check, check_pages, gold, gold_list, has_citation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--repeats", type=int, default=3, help="runs per not_found / correction question")
    ap.add_argument("--repeat-all", action="store_true", help="repeat answer questions too")
    ap.add_argument("--expect", choices=sorted(cases.EXPECTS), help="only questions with this expect")
    ap.add_argument("--id", action="append", help="only these ids (repeatable)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--run-name")
    a = ap.parse_args()
    # MLflow prints an emoji at run end; Windows' cp1252 console encoding can't when output is piped
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = cases.load(a.split)              # validates the file before any model is loaded
    if a.expect:
        rows = [r for r in rows if r["expect"] == a.expect]
    if a.id:
        rows = [r for r in rows if r.get("id") in a.id]
    rows = rows[: a.limit] if a.limit else rows
    bad = []
    for r in rows:                          # every gold_sql must work before the agent runs
        if r["expect"] in ("answer", "pages"):
            try:
                gold(r) if r["expect"] == "answer" else gold_list(r)
            except GoldError as e:
                bad.append(f"line {r['line']}: {e}")
    if bad:
        raise SystemExit("gold_sql problems:\n  " + "\n  ".join(bad))
    runs = [(r, i) for r in rows for i in range(a.repeats if a.repeat_all or r["expect"] not in ("answer", "pages") else 1)]
    print(f"{a.split}: {len(rows)} questions, {len(runs)} agent runs (~{len(runs) * 25 / 60:.0f} min)", flush=True)

    ensure_embed_model_loaded()
    init_mlflow()
    get_agent()
    s = get_settings()
    out = []
    with mlflow.start_run(run_name=a.run_name or a.split, tags={"split": a.split}):
        mlflow.log_params({"split": a.split, "n_questions": len(rows), "n_runs": len(runs), "repeats": a.repeats, "repeat_all": a.repeat_all,
                           "expect_filter": a.expect or "all", "llm_provider": s.llm_provider,
                           "llm_model": s.llm_model, "embed_model": s.embed_model,
                           "sql_retries": s.sql_retries, "film_search_k": s.film_search_k,
                           "max_refinements": s.max_refinements, "temperature": s.llm_temperature,
                           "top_p": s.llm_top_p, "top_k": s.llm_top_k})
        for n, (r, rep) in enumerate(runs, 1):
            t0 = time.time()
            base = {"line": r["line"], "id": r.get("id", ""), "question": r["question"], "expect": r["expect"],
                    "repeat": rep}
            try:
                if r["expect"] == "pages":   # the question, then each follow-up in the same conversation
                    conv = uuid.uuid4().hex
                    turns = [ask(r["question"], conv)] + [ask(f["say"], conv) for f in r.get("follow_ups", [])]
                    ok, reason = check_pages(r, turns)
                    res = {**turns[0], "answer": "\n\n---\n\n".join(t["answer"] for t in turns),
                           "sql": [q for t in turns for q in t["sql"]]}
                else:
                    res = ask(r["question"])
                    ok, reason = check(r, res["answer"], res["status"])
                sql = " ;; ".join(q["sql"] for q in res["sql"] if "error" not in q)
                out.append({**base, "status": res["status"], "passed": ok, "reason": reason,
                            "cited": has_citation(res["answer"]), "seconds": round(time.time() - t0, 1),
                            "answer": res["answer"], "sql": sql,
                            "sql_errors": sum("error" in q for q in res["sql"]), "error": ""})
            except Exception as e:
                out.append({**base, "status": "error", "passed": False, "reason": f"error: {e!r}"[:300],
                            "cited": False, "seconds": round(time.time() - t0, 1), "answer": "", "sql": "",
                            "sql_errors": 0, "error": repr(e)})
            o = out[-1]
            print(f"[{n}/{len(runs)}] {'PASS' if o['passed'] else 'FAIL'} {r['expect']:<10} {o['status']:<9} "
                  f"{o['seconds']}s  {r['question'][:60]}" + (f"\n      -> {o['reason']}" if o["reason"] else ""),
                  flush=True)

        df = pd.DataFrame(out)
        m = summarize(df)
        mlflow.log_table(df, "eval_results.json")
        mlflow.log_metrics({k: float(v) for k, v in m.items()})

    print("\n" + pd.Series(m).round(3).to_string())
    fails = df[~df["passed"]].drop_duplicates("question")
    if len(fails):
        print(f"\nfailed ({len(fails)} questions):")
        for _, f in fails.iterrows():
            print(f"  line {f['line']} ({f['id']}): {f['question'][:70]}\n      -> {f['reason']}")


def summarize(df: pd.DataFrame) -> dict:
    m = {"pass_rate": df["passed"].mean(), "error_rate": (df["status"] == "error").mean(),
         "mean_seconds": df["seconds"].mean(), "sql_error_rate": (df["sql_errors"] > 0).mean()}
    ans = df[df["expect"] == "answer"]
    if len(ans):
        m["answer_pass_rate"] = ans["passed"].mean()
        m["false_refusal_rate"] = (ans["status"] == "not_found").mean()   # said "not found" to an answerable question
        m["partial_rate"] = (ans["status"] == "partial").mean()
        m["cited_rate"] = ans["cited"].mean()
        if ans["repeat"].nunique() > 1:
            m["answer_consistency"] = ans.groupby("question")["passed"].nunique().eq(1).mean()
    pg_ = df[df["expect"] == "pages"]
    if len(pg_):
        m["pages_pass_rate"] = pg_["passed"].mean()
    for e in ("not_found", "correction"):
        sub = df[df["expect"] == e]
        if len(sub):
            m[f"{e}_pass_rate"] = sub["passed"].mean()
            if sub["repeat"].nunique() > 1:     # same result on every repeat
                m[f"{e}_consistency"] = sub.groupby("question")["passed"].nunique().eq(1).mean()
    if len(df[df["expect"] == "not_found"]):
        m["made_up_rate"] = 1 - m["not_found_pass_rate"]
    return m


if __name__ == "__main__":
    main()
