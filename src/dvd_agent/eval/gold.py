"""Print every eval line with the result of its gold_sql, to review the expected answers.

python -m dvd_agent.eval.gold --split dev
"""
from __future__ import annotations

import argparse
import sys

from . import cases
from .metrics import GoldError, gold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    bad = 0
    for r in cases.load(a.split):
        print(f"line {r['line']} [{r['expect']}] {r['question']}")
        if r["expect"] != "answer":
            print(f"    {r.get('why', '')} {r.get('contains', '')}".rstrip())
            continue
        try:
            g = gold(r)
            print(f"    texts: {g['texts']}" if g["texts"] else "", end="")
            print(f"    numbers: {g['numbers']}" if g["numbers"] else "")
        except GoldError as e:
            bad += 1
            print(f"    GOLD ERROR: {e}")
    if bad:
        raise SystemExit(f"{bad} gold_sql problem(s)")


if __name__ == "__main__":
    main()
