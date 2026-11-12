"""Ask from the terminal.

python -m dvd_agent.cli "Which store made more revenue in 2007?" [--json]   one question
python -m dvd_agent.cli                                                     a conversation: after a long
                                                                            list, type "page 2", "next page",
                                                                            "20 per page"; empty line quits
"""
import argparse
import json
import sys
import uuid

from .agent.graph import ask
from .llm import ensure_embed_model_loaded
from .tracing import init_mlflow


def show(res: dict, as_json: bool) -> None:
    if as_json:
        json.dump(res, sys.stdout, indent=2, default=str)
        print()
        return
    print(f"[{res['status']}]")
    if res.get("standalone_question"):
        print(f"(understood as: {res['standalone_question']})")
    print(res["answer"], "\n")
    if res["plan"] and res["plan"].get("page_request"):   # pages are printed straight from the rows
        print("Steps:", " -> ".join(f"{s['node']}({s['seconds']}s)" for s in res["steps"]))
        return
    print("Sources:")
    for c in res["citations"]:
        if c["type"] == "sql":
            page = f", page {c['page']} of {c['pages']}" if c.get("pages", 1) > 1 else ""
            print(f"  [{c['id']}] {c['row_count']} rows{page}: {c['sql']}")
        else:
            print(f"  [{c['id']}] film {c['film_id']} {c['title']}")
    failed = [q for q in res["sql"] if "error" in q]
    if failed:
        print(f"\n{len(failed)} failed SQL attempt(s), fixed on retry:" if res["citations"] else "\nFailed SQL:")
        for q in failed:
            print(f"  {q['error'][:150]}")
    print("\nSteps:", " -> ".join(f"{s['node']}({s['seconds']}s)" for s in res["steps"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="omit for a conversation (paging: 'page 2', '20 per page')")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-trace", action="store_true")
    a = ap.parse_args()
    if not a.no_trace:
        init_mlflow()
    ensure_embed_model_loaded()
    conversation = uuid.uuid4().hex
    if a.question:
        show(ask(a.question, conversation), a.json)
        return
    print('Ask a question. After a long list: "page 2", "pages 2, 3", "next page", "20 per page". '
          "Empty line quits.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(ask(q, conversation), a.json)


if __name__ == "__main__":
    main()
