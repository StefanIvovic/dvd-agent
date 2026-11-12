"""LangGraph agent.

     contextualize   DSPy, only for a follow-up in a conversation: rewrite it as a standalone question
             ▼
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
        │ assess   │  false premise not corrected (premise_conflict) ──► revise (same sources) ─► answer
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
from . import memory, paging, schema, tools
from .signatures import DvdQAProgram, load_program


class AgentState(TypedDict, total=False):
    question: str                # standalone question the agent answers (the rewrite of a follow-up)
    message: str                 # what the user typed, if it was rewritten
    history: str                 # earlier turns of the conversation (agent/memory.py), empty if none
    page_size: int               # rows per page if the question asked for one ("20 per page"), else PAGE_SIZE
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
    paged: dict                  # {"qid", "result"}: the cited result with more than one page, if any
    review: str                  # self-check note for a rewrite (premise_conflict), else empty
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

    def contextualize(state: AgentState) -> dict:
        """A follow-up ("what actors are in this movie?") becomes a standalone question. No LLM call
        without history."""
        if not state.get("history"):
            return {}
        t0 = time.time()
        message = state["question"]
        q = (program.contextualize(conversation=state["history"], message=message).question or "").strip()
        q = q or message
        return {"question": q, "message": message, **_step("contextualize", t0, question=q)}

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
            round_results = [tools.run_sql(q, question, state.get("page_size")) for q in queries]
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
        p = program.answer(question=state["question"], sources=text, review=state.get("review", ""))
        return {"answer": p.answer, "answer_reasoning": p.reasoning, "sources_text": text, "citation_map": cmap,
                **_step("answer", t0)}

    def assess(state: AgentState) -> dict:
        t0 = time.time()
        r = program.assess(question=state["question"], sources=state["sources_text"], answer=state["answer"])
        a = {"grounded": bool(r.grounded), "complete": bool(r.complete), "answer_found": bool(r.answer_found),
             "missing": r.missing or "", "follow_up": r.follow_up or "",
             "premise_conflict": (r.premise_conflict or "").strip()}
        return {"assessment": a, **_step("assess", t0, **a)}

    def after_assess(state: AgentState) -> str:
        a = state["assessment"]
        ok = a["grounded"] and a["complete"] and a["answer_found"]
        if state.get("refinements", 0) < settings.max_refinements:
            if not ok:
                return "refine"
            if a.get("premise_conflict"):   # right data, but the answer goes along with a false premise
                return "revise"
        return "finalize"

    def _review(a: dict) -> str:
        c = a.get("premise_conflict")
        return (f"The question states something the sources contradict: {c}. Say so in the first sentence, "
                "with the correct value and its citation, then answer.") if c else ""

    def revise(state: AgentState) -> dict:
        """Rewrite the answer (same sources, no new SQL) with the self-check's note."""
        t0 = time.time()
        review = _review(state["assessment"])
        return {"review": review, "refinements": state.get("refinements", 0) + 1,
                **_step("revise", t0, review=review)}

    def refine(state: AgentState) -> dict:
        t0 = time.time()
        a = state["assessment"]
        need = a["follow_up"] or a["missing"]
        previous = _summary(state.get("results", [])) + f"\n\nThe answer from these results is incomplete: {need}"
        out = run_sql_round(state, previous)
        return {**out, "refinements": state.get("refinements", 0) + 1, "review": _review(a),
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
        paged = None
        if status != "not_found":   # a cited result with more pages: say so, so the user can ask for them
            cmap = state.get("citation_map", {})
            hit = next((c for c in cited_sources(draft, cmap) if c["type"] == "sql" and c["pages"] > 1), None)
            if hit:
                paged = {"qid": hit["id"], "result": hit}
                final = final + "\n\n" + paging.footer(hit, hit["id"])
        reason = ",".join(reasons)
        _tag_trace(state["question"], status, reason, not_in_src, a, state.get("refinements", 0), cannot,
                   state.get("message", ""))
        return {"status": status, "answer": final, "draft_answer": draft, "paged": paged,
                **_step("finalize", t0, status=status, reason=reason or None, not_in_sources=not_in_src or None)}

    g = StateGraph(AgentState)
    for name, fn in [("contextualize", contextualize), ("analyze", analyze), ("film_search", film_search), ("sql", sql), ("answer", answer),
                     ("assess", assess), ("refine", refine), ("revise", revise), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "analyze")
    g.add_conditional_edges("analyze", route, ["film_search", "sql"])
    g.add_edge("film_search", "sql")
    g.add_conditional_edges("sql", after_sql, ["answer", "finalize"])
    g.add_edge("answer", "assess")
    g.add_conditional_edges("assess", after_assess, ["refine", "revise", "finalize"])
    g.add_edge("refine", "answer")
    g.add_edge("revise", "answer")
    g.add_edge("finalize", END)
    return g.compile()


NOT_FOUND_MESSAGE = ("I couldn't find data in the DVD rental database to answer this question reliably, "
                     "so I'm not giving an answer.")
PARTIAL_NOTE = ("_Note: part of this question could not be answered from the database; "
                "the answer above covers only what the data supports._")


def _tag_trace(question: str, status: str, reason: str, not_in_sources: str, assessment: dict,
               refinements: int, cannot_answer: str, message: str = "") -> None:
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
            "user_message": message[:250],   # what the user typed, when a follow-up was rewritten
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


def ask(question: str, conversation_id: str | None = None) -> dict[str, Any]:
    """Answers a question. With a conversation_id, "page 2" / "20 per page" pages the last listed result
    (agent/paging.py: the query runs again, the rows are printed directly, no LLM)."""
    cmd = paging.parse_command(question)
    if cmd is not None and conversation_id:
        return {"question": question, "conversation_id": conversation_id,
                **_traced_page_request(cmd, conversation_id, question)}
    llm_question, size = paging.split_question(question)
    state = get_agent().invoke({"question": llm_question, "history": memory.history(conversation_id),
                                **({"page_size": size} if size else {})})
    status = state.get("status", "answered")
    memory.add(conversation_id, state["question"], state.get("answer", ""), status)
    paged = state.get("paged")
    if paged:
        paging.remember(conversation_id, paged["result"], paged["qid"], question)
    else:
        paging.forget(conversation_id)   # a new question without pages: "page 2" no longer refers to the old list
    return {
        "question": question,
        # the follow-up as the agent understood it; None if the question was used as typed
        "standalone_question": state["question"] if state.get("message") and state["question"] != llm_question
        else None,
        "conversation_id": conversation_id,
        "status": status,   # answered | partial | not_found
        "answer": state.get("answer", ""),
        "citations": [] if status == "not_found"
        else cited_sources(state.get("answer", ""), state.get("citation_map", {})),
        # every query that ran, with its rows (the UI shows them; errors included for transparency)
        "sql": [{k: r[k] for k in ("sql", "columns", "rows", "row_count", "page", "pages", "page_size", "error")
                 if k in r}
                for r in state.get("results", [])],
        "films": [{k: f[k] for k in ("film_id", "title", "description", "score")} for f in state.get("films", [])],
        "plan": state.get("plan"),
        "assessment": {k: v for k, v in (state.get("assessment") or {}).items()
                       if status != "not_found" or k in ("grounded", "complete", "answer_found")},
        "steps": state.get("steps", []),
    }


def _traced_page_request(cmd: dict, conversation_id: str, question: str) -> dict:
    try:
        import mlflow

        @mlflow.trace(name="page_request", span_type="CHAIN")
        def run(question: str, command: dict) -> dict:
            return paging.answer_command(command, conversation_id)
        return run(question, cmd)
    except ImportError:
        return paging.answer_command(cmd, conversation_id)
