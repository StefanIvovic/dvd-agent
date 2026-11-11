"""Ask from the terminal:  python -m dvd_agent.cli "Which store made more revenue in 2007?" [--json]"""
import argparse
import json
import sys

from .agent.graph import ask
from .llm import ensure_embed_model_loaded
from .tracing import init_mlflow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-trace", action="store_true")
    a = ap.parse_args()
    if not a.no_trace:
        init_mlflow()
    ensure_embed_model_loaded()
    res = ask(a.question)
    if a.json:
        json.dump(res, sys.stdout, indent=2, default=str)
        return
    print(f"[{res['status']}]")
    print(res["answer"], "\n")
    print("Sources:")
    for c in res["citations"]:
        if c["type"] == "sql":
            print(f"  [{c['id']}] {c['row_count']} rows: {c['sql']}")
        else:
            print(f"  [{c['id']}] film {c['film_id']} {c['title']}")
    failed = [q for q in res["sql"] if "error" in q]
    if failed:
        print(f"\n{len(failed)} failed SQL attempt(s), fixed on retry:" if res["citations"] else "\nFailed SQL:")
        for q in failed:
            print(f"  {q['error'][:150]}")
    print("\nSteps:", " -> ".join(f"{s['node']}({s['seconds']}s)" for s in res["steps"]))


if __name__ == "__main__":
    main()
