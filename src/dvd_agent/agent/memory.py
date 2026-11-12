"""Conversation memory: the last few turns of each conversation, so a follow-up ("what actors are in this
movie?") can be rewritten into a standalone question before the agent runs.

Only text is kept (the question as the agent understood it, and the answer), in process memory, for
STATE_TTL_SECONDS after the last turn. Rows are never stored: answers run their SQL again.
"""
from __future__ import annotations

import re
import threading
import time

from .paging import STATE_TTL_SECONDS

MAX_TURNS = 3            # turns given to the rewrite step
MAX_ANSWER_CHARS = 1500  # per answer; page 1 of a long list is cut here

_lock = threading.Lock()
_turns: dict[str, dict] = {}   # conversation_id -> {"turns": [...], "at": time}


def add(conversation_id: str | None, question: str, answer: str, status: str) -> None:
    if not conversation_id:
        return
    with _lock:
        now = time.time()
        for k in [k for k, v in _turns.items() if now - v["at"] > STATE_TTL_SECONDS]:
            del _turns[k]
        c = _turns.setdefault(conversation_id, {"turns": [], "at": now})
        c["turns"] = (c["turns"] + [{"question": question, "answer": answer, "status": status}])[-MAX_TURNS:]
        c["at"] = now


def history(conversation_id: str | None) -> str:
    """The last turns as text for the LLM, oldest first; empty for a new conversation."""
    if not conversation_id:
        return ""
    with _lock:
        c = _turns.get(conversation_id)
        if not c or time.time() - c["at"] > STATE_TTL_SECONDS:
            return ""
        turns = list(c["turns"])
    out = []
    for t in turns:
        answer = re.sub(r"\s*\[(?:[QS]\d+(?:\s*,\s*)?)+\]", "", t["answer"]).strip()   # drop [Q1] citations
        if len(answer) > MAX_ANSWER_CHARS:
            answer = answer[:MAX_ANSWER_CHARS] + " …"
        if t["status"] == "not_found":
            answer = "(no answer: not in the data)"
        out.append(f"User: {t['question']}\nAssistant: {answer}")
    return "\n\n".join(out)


def clear(conversation_id: str | None) -> None:
    with _lock:
        _turns.pop(conversation_id or "", None)
