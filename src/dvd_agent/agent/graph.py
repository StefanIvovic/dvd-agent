"""LangGraph agent.

        ┌──────────┐
        │ analyze  │  DSPy: does the question need film search (plot/theme)?
        └────┬─────┘
             ├──── film search needed ───► film_search (LlamaIndex: BM25 + pgvector)
             ▼                                  │
        ┌──────────┐ ◄──────────────────────────┘
        │   sql    │  DSPy text→SQL, guard, run as dvd_reader; errors go back to the LLM (SQL_RETRIES)
        └────┬─────┘
             ▼            no rows / cannot answer ──► finalize (not_found)
        ┌──────────┐
        │ answer   │  DSPy ChainOfThought: cited answer
        └────┬─────┘
             ▼
        ┌──────────┐  DSPy: grounded/complete? ── no & budget left ──► refine (sql with follow_up) ─► answer
        │ assess   │
        └────┬─────┘
             ▼
          finalize ──► END
"""
from __future__ import annotations

import operator
import re
import time
from functools import lru_cache
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from ..llm import configure_dspy
from . import schema, tools
from .signatures import DvdQAProgram, load_program


class AgentState(TypedDict, total=False):
    question: str
    plan: dict
    films: list[dict]
    results: Annotated[list[dict], operator.add]   # every SQL run (successful and failed), all rounds
    cannot_answer: str
    sql_reasoning: Annotated[list[str], operator.add]   # ChainOfThought scratchpads; traces only
    sources_text: str
    citation_map: dict
    answer: str
    answer_reasoning: str
    assessment: dict
    refinements: int
    status: str                  # answered | partial | not_found (set by finalize)
    draft_answer: str            # LLM answer before finalize; kept for traces, never shown if not_found
    steps: Annotated[list[dict], operator.add]


def _step(name: str, t0: float, **info) -> dict:
    return {"steps": [{"node": name, "seconds": round(time.time() - t0, 2), **info}]}


def _ok(results: list[dict]) -> list[dict]:
    return [r for r in results if "error" not in r]


def _summary(results: list[dict]) -> str:
    """Earlier queries for the SQL step: errors in full, results as row counts + first rows."""
    out = []
    for r in results:
        if "error" in r:
            out.append(f"SQL: {r['sql']}\nERROR: {r['error']}")
        else:
            head = "; ".join(str(row) for row in r["rows"][:3])
            out.append(f"SQL: {r['sql']}\nRESULT: {r['row_count']} rows. First rows: {head[:600]}")
    return "\n\n".join(out)


def build_graph(program: DvdQAProgram | None = None):
    configure_dspy()
    program = program or load_program()
    settings = get_settings()

    def analyze(state: AgentState) -> dict:
        t0 = time.time()
        p = program.analyze(question=state["question"])
        plan = {"needs_film_search": bool(p.needs_film_search),
                "film_search_queries": (p.film_search_queries or [state["question"]])[:2]}
        return {"plan": plan, "refinements": 0, **_step("analyze", t0, plan=plan)}

    def route(state: AgentState) -> str:
        return "film_search" if state["plan"]["needs_film_search"] else "sql"

    def film_search(state: AgentState) -> dict:
        t0 = time.time()
        films = tools.search_films(state["plan"]["film_search_queries"])
        return {"films": films, **_step("film_search", t0, n=len(films), titles=[f["title"] for f in films])}

    def run_sql_round(state: AgentState, previous: str) -> dict:
        """Generate SQL, run it, and on errors regenerate with the error messages (SQL_RETRIES times)."""
        question = state["question"]
        db_schema = schema.describe(question)
        film_hits = tools.format_film_hits(state.get("films", []))
        results, reasoning, cannot = [], [], ""
        for attempt in range(settings.sql_retries + 1):
            p = program.sql(question=question, db_schema=db_schema, film_hits=film_hits,
                            previous_attempts=previous)
            reasoning.append(p.reasoning)
            queries = [q for q in (p.queries or []) if q and q.strip()][:3]
            cannot = (p.cannot_answer or "").strip() if not queries else ""
            if not queries:
                break
            round_results = [tools.run_sql(q, question) for q in queries]
            results += round_results
            if all("error" not in r for r in round_results):
                break
            previous = (previous + "\n\n" if previous else "") + _summary(round_results)
        return {"results": results, "sql_reasoning": reasoning, "cannot_answer": cannot}

    def sql(state: AgentState) -> dict:
        t0 = time.time()
        out = run_sql_round(state, "")
        return {**out, **_step("sql", t0, queries=[r["sql"] for r in out["results"]],
                               errors=[r["error"] for r in out["results"] if "error" in r] or None,
                               cannot_answer=out["cannot_answer"] or None)}

    def after_sql(state: AgentState) -> str:
        return "answer" if _ok(state.get("results", [])) or state.get("films") else "finalize"

    def answer(state: AgentState) -> dict:
        t0 = time.time()
        text, cmap = tools.format_sources(state.get("results", []), state.get("films", []))
        p = program.answer(question=state["question"], sources=text)
        return {"answer": p.answer, "answer_reasoning": p.reasoning, "sources_text": text, "citation_map": cmap,
                **_step("answer", t0)}

    def assess(state: AgentState) -> dict:
        t0 = time.time()
        r = program.assess(question=state["question"], sources=state["sources_text"], answer=state["answer"])
        a = {"grounded": bool(r.grounded), "complete": bool(r.complete), "answer_found": bool(r.answer_found),
             "missing": r.missing or "", "follow_up": r.follow_up or ""}
        return {"assessment": a, **_step("assess", t0, **a)}

    def after_assess(state: AgentState) -> str:
        a = state["assessment"]
        ok = a["grounded"] and a["complete"] and a["answer_found"]
        if not ok and state.get("refinements", 0) < settings.max_refinements:
            return "refine"
        return "finalize"

    def refine(state: AgentState) -> dict:
        t0 = time.time()
        a = state["assessment"]
        need = a["follow_up"] or a["missing"]
        previous = _summary(state.get("results", [])) + f"\n\nThe answer from these results is incomplete: {need}"
        out = run_sql_round(state, previous)
        return {**out, "refinements": state.get("refinements", 0) + 1,
                **_step("refine", t0, need=need, queries=[r["sql"] for r in out["results"]],
                        errors=[r["error"] for r in out["results"] if "error" in r] or None)}

    def finalize(state: AgentState) -> dict:
        """Only grounded content reaches the user.

        answered  - self-check: grounded and complete
        partial   - grounded but incomplete, or the answer's opening says the data lacks something
        not_found - no usable results, or not grounded / no answer after the retry: fixed message
        """
        t0 = time.time()
        a = state.get("assessment") or {}
        has_sources = bool(_ok(state.get("results", [])) or state.get("films"))
        draft = state.get("answer", "")
        not_in_src = tools.lead_not_in_sources_match(draft)
        cannot = state.get("cannot_answer", "")
        if not has_sources or not a.get("grounded") or not a.get("answer_found"):
            status = "not_found"
            final = NOT_FOUND_MESSAGE + (f"\n\n{cannot}" if cannot and not has_sources else "")
            failed = [r for r in state.get("results", []) if "error" in r]
            reasons = [r for r, hit in (
                ("cannot_answer", not has_sources and bool(cannot)),
                ("sql_failed", not has_sources and bool(failed)),
                ("no_sources", not has_sources and not cannot and not failed),
                ("not_grounded", has_sources and not a.get("grounded")),
                ("no_answer_found", has_sources and not a.get("answer_found"))) if hit]
        elif not a["complete"] or not_in_src:
            status, final = "partial", draft + "\n\n" + PARTIAL_NOTE
            reasons = [r for r, hit in (("assess_incomplete", not a["complete"]),
                                        ("says_not_in_sources", bool(not_in_src))) if hit]
        else:
            status, final, reasons = "answered", draft, []
        reason = ",".join(reasons)
        _tag_trace(state["question"], status, reason, not_in_src, a, state.get("refinements", 0), cannot)
        return {"status": status, "answer": final, "draft_answer": draft,
                **_step("finalize", t0, status=status, reason=reason or None, not_in_sources=not_in_src or None)}

    g = StateGraph(AgentState)
    for name, fn in [("analyze", analyze), ("film_search", film_search), ("sql", sql), ("answer", answer),
                     ("assess", assess), ("refine", refine), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", route, ["film_search", "sql"])
    g.add_edge("film_search", "sql")
    g.add_conditional_edges("sql", after_sql, ["answer", "finalize"])
    g.add_edge("answer", "assess")
    g.add_conditional_edges("assess", after_assess, ["refine", "finalize"])
    g.add_edge("refine", "answer")
    g.add_edge("finalize", END)
    return g.compile()


NOT_FOUND_MESSAGE = ("I couldn't find data in the DVD rental database to answer this question reliably, "
                     "so I'm not giving an answer.")
PARTIAL_NOTE = ("_Note: part of this question could not be answered from the database; "
                "the answer above covers only what the data supports._")


def _tag_trace(question: str, status: str, reason: str, not_in_sources: str, assessment: dict,
               refinements: int, cannot_answer: str) -> None:
    """Tag the MLflow trace so unanswered questions can be found: tags.answer_status = 'not_found'.

    status_reason (comma-separated if several):
      not_found: cannot_answer | sql_failed | no_sources | not_grounded | no_answer_found
      partial:   assess_incomplete | says_not_in_sources (not_in_sources = the sentence that matched)
    """
    try:
        import mlflow

        mlflow.update_current_trace(tags={
            "answer_status": status,
            "status_reason": reason,
            "not_in_sources": not_in_sources[:250],
            "cannot_answer": cannot_answer[:250],
            "grounded": str(assessment.get("grounded")),
            "complete": str(assessment.get("complete")),
            "refinements": str(refinements),
            "question": question[:250],
            "missing": (assessment.get("missing") or "")[:500],
        })
    except Exception:  # tracing off or no active trace: never break the answer
        pass


@lru_cache
def get_agent():
    return build_graph()


def cited_sources(answer: str, cmap: dict) -> list[dict]:
    ids = dict.fromkeys(re.findall(r"\[([QS]\d+)\]", answer))
    for grp in re.findall(r"\[((?:[QS]\d+\s*,\s*)+[QS]\d+)\]", answer):   # grouped: [Q1, S3]
        for i in re.split(r"\s*,\s*", grp):
            ids.setdefault(i, None)
    return [{"id": i, **cmap[i]} for i in ids if i in cmap]


def ask(question: str) -> dict[str, Any]:
    state = get_agent().invoke({"question": question})
    status = state.get("status", "answered")
    return {
        "question": question,
        "status": status,   # answered | partial | not_found
        "answer": state.get("answer", ""),
        "citations": [] if status == "not_found"
        else cited_sources(state.get("answer", ""), state.get("citation_map", {})),
        # every query that ran, with its rows (the UI shows them; errors included for transparency)
        "sql": [{k: r[k] for k in ("sql", "columns", "rows", "row_count", "truncated", "error") if k in r}
                for r in state.get("results", [])],
        "films": [{k: f[k] for k in ("film_id", "title", "description", "score")} for f in state.get("films", [])],
        "plan": state.get("plan"),
        "assessment": {k: v for k, v in (state.get("assessment") or {}).items()
                       if status != "not_found" or k in ("grounded", "complete", "answer_found")},
        "steps": state.get("steps", []),
    }
