"""Streamlit front end for the API.  streamlit run src/dvd_agent/ui/streamlit_app.py

Two pages: Chat (POST /ask on the FastAPI service at API_URL) and Manual (ui/manual.md).
The API returns [Q1]/[S1] citations and the SQL behind every answer. The chat shows the answer
with the citations as superscripts, the data when it adds something to the answer, and the SQL
under "How this was computed". Developer mode adds the steps, the MLflow trace, failed SQL
attempts and how a follow-up was understood.
"""
import re
from pathlib import Path

import httpx
import pandas as pd
import sqlglot
import streamlit as st

from dvd_agent.config import get_settings

API = get_settings().api_url
EXAMPLES = ["Which film category brought in the most revenue in April 2007?",
            "Which actor appeared in the most films about a lumberjack?",
            "List all Sci-Fi films.",
            "How much revenue did we make in January 2006?"]
LABELS = {"answered": ("answered", "green"), "partial": ("partly answered", "orange"),
          "not_found": ("not in the data", "red")}
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
CITATION = re.compile(r"\s?\[((?:[QS]\d+\s*,\s*)*[QS]\d+)\]")


def sup(n: int) -> str:
    return str(n).translate(SUP)


def with_superscripts(answer: str, numbers: dict[str, int]) -> str:
    """[Q1] -> ¹, [Q1, S2] -> ¹˒², numbered in the order the sources are listed below the answer."""
    def repl(m: re.Match) -> str:
        ids = [i for i in re.split(r"\s*,\s*", m.group(1)) if i in numbers]
        return "˒".join(sup(numbers[i]) for i in ids)
    return CITATION.sub(repl, answer)


def pretty_sql(sql: str) -> str:
    try:
        return sqlglot.transpile(sql, read="postgres", write="postgres", pretty=True)[0]
    except Exception:
        return sql


def chart(df: pd.DataFrame) -> None:
    """Bar chart for (label, number) results with 2-30 rows."""
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.endswith("_id")]
    txt = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]
    if len(num) >= 1 and txt and 2 <= len(df) <= 30:
        st.bar_chart(df.set_index(txt[0])[num[0]], horizontal=True)


def show(d: dict) -> None:
    dev = st.session_state.get("dev", False)
    label, color = LABELS.get(d["status"], (d["status"], "gray"))
    st.markdown(f":{color}-badge[{label}]")
    if dev and d.get("standalone_question"):
        st.caption(f"Understood as: {d['standalone_question']}")
    citations = d.get("citations", [])
    numbers = {c["id"]: n for n, c in enumerate(citations, 1)}
    st.markdown(with_superscripts(d["answer"], numbers))

    page_reply = any(s.get("node") == "page" for s in d.get("steps", []))   # the rows are in the answer
    films = [c for c in citations if c["type"] != "sql"]
    for c in citations:
        if c["type"] != "sql":
            continue
        n = sup(numbers[c["id"]])
        page = f", page {c['page']} of {c['pages']}" if c.get("pages", 1) > 1 else ""
        # one row is a single value the answer already states; developer mode shows it anyway
        if not page_reply and (c["row_count"] > 1 or dev):
            df = pd.DataFrame(c["rows"], columns=c["columns"])
            # open when the table adds to the answer (several columns, short); a one-column list or a
            # long table is mostly what the answer already lists
            adds = len(df.columns) > 1 and len(df) <= 30
            with st.expander(f"Data {n} · {c['row_count']} rows{page}", expanded=adds or dev):
                st.dataframe(df, hide_index=True)
                chart(df)
        with st.expander(f"How this was computed {n}", expanded=dev and not page_reply):
            st.code(pretty_sql(c["sql"]), language="sql")
    if films:
        with st.expander("Film matches " + "˒".join(sup(numbers[c["id"]]) for c in films)):
            for c in films:
                st.markdown(f"{sup(numbers[c['id']])} **{c['title']}** — {c['description']}")

    if dev:
        failed = [q for q in d.get("sql", []) if "error" in q]
        if failed:
            with st.expander(f"{len(failed)} failed SQL attempt(s)" + (", fixed on retry" if citations else "")):
                for q in failed:
                    st.code(pretty_sql(q["sql"]), language="sql")
                    st.caption(q["error"])
        trace = (f" · [MLflow trace]({d['trace_url']})" if d.get("trace_url")
                 else f" · MLflow trace {d['trace_id']}" if d.get("trace_id") else "")
        st.caption(" → ".join(f"{s['node']} {s['seconds']}s" for s in d.get("steps", [])) + trace)


def chat() -> None:
    st.title("DVD Rental Agent")
    try:
        s = httpx.get(f"{API}/stats", timeout=5).json()
        st.caption(f"{s['films']} films · {s['with_embedding']} embedded for search")
    except Exception as e:
        st.error(f"API not reachable at {API}: start it with `uvicorn dvd_agent.api.main:app --port 8001` ({e})")
        st.stop()

    for q, d in st.session_state.history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            show(d)

    q = (st.chat_input("Ask about films, actors, customers, rentals, payments…")   # pinned to the bottom
         or st.session_state.pop("picked", None))
    if not st.session_state.history and not q:
        st.caption("Try one of these:")
        for col, ex in zip(st.columns(len(EXAMPLES)), EXAMPLES):
            if col.button(ex, use_container_width=True):
                st.session_state.picked = ex
                st.rerun()   # redraw without the buttons, then ask
    if q:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Working (local model, this can take a minute)…"):
                try:
                    r = httpx.post(f"{API}/ask", json={"question": q, "conversation_id": st.session_state.conversation},
                                   timeout=600)
                    r.raise_for_status()
                    d = r.json()
                    st.session_state.conversation = d.get("conversation_id")
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.stop()
            show(d)
        st.session_state.history.append((q, d))


def manual() -> None:
    st.markdown((Path(__file__).parent / "manual.md").read_text(encoding="utf-8"))


st.set_page_config(page_title="DVD Rental Agent", layout="wide")
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.conversation = None   # sent back so follow-ups and "page 2" use this conversation
page = st.navigation([st.Page(chat, title="Chat", icon="💬", default=True),
                      st.Page(manual, title="Manual", icon="📖")])
with st.sidebar:
    if st.button("New conversation", use_container_width=True):
        st.session_state.history, st.session_state.conversation = [], None
        st.rerun()
    st.toggle("Developer mode", key="dev", help="Show the SQL, steps, MLflow trace and failed SQL attempts")
    st.caption("After a long list, ask “page 2”, “pages 2, 3”, “next page”, “last page” or “20 per page”.")
    with st.expander("Health"):
        try:
            h = httpx.get(f"{API}/health", timeout=10)
            st.json(h.json().get("detail", h.json()) if h.status_code != 200 else h.json())
        except Exception as e:
            st.error(str(e))
page.run()
