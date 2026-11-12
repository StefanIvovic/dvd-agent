"""Streamlit front end for the API.  streamlit run src/dvd_agent/ui/streamlit_app.py

Calls POST /ask on the FastAPI service (API_URL), shows the answer, every SQL result as a table and
a bar chart when a result has one label column and one numeric column. Long lists are paged by the
API; "page 2" / "20 per page" in the same conversation page the last list.
"""
import httpx
import pandas as pd
import streamlit as st

from dvd_agent.config import get_settings

API = get_settings().api_url
EXAMPLES = ["Which film category brought in the most revenue in April 2007?",
            "Which actor appeared in the most films about a lumberjack?",
            "List all Sci-Fi films.",
            "How much revenue did we make in January 2006?"]
LABELS = {"answered": ("answered", "green"), "partial": ("partly answered", "orange"),
          "not_found": ("not in the data", "red")}

st.set_page_config(page_title="DVD Rental Agent", layout="wide")
st.title("DVD Rental Agent")

try:
    s = httpx.get(f"{API}/stats", timeout=5).json()
    st.caption(f"{s['films']} films · {s['with_embedding']} embedded for search · API {API}")
except Exception as e:
    st.error(f"API not reachable at {API}: start it with `uvicorn dvd_agent.api.main:app --port 8001` ({e})")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.conversation = None   # sent back so "page 2" / "20 per page" page the last list
with st.sidebar:
    if st.button("New conversation", use_container_width=True):
        st.session_state.history, st.session_state.conversation = [], None
        st.rerun()
    st.caption("After a long list, ask “page 2”, “pages 2, 3”, “next page”, “last page” or “20 per page”.")
    with st.expander("Health"):
        try:
            h = httpx.get(f"{API}/health", timeout=10)
            st.json(h.json().get("detail", h.json()) if h.status_code != 200 else h.json())
        except Exception as e:
            st.error(str(e))


def chart(df: pd.DataFrame) -> None:
    """Bar chart for (label, number) results with 2-30 rows."""
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.endswith("_id")]
    txt = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]
    if len(num) >= 1 and txt and 2 <= len(df) <= 30:
        st.bar_chart(df.set_index(txt[0])[num[0]], horizontal=True)


def show(d: dict) -> None:
    label, color = LABELS.get(d["status"], (d["status"], "gray"))
    st.markdown(f":{color}-badge[{label}]")
    st.markdown(d["answer"])
    page_reply = any(s.get("node") == "page" for s in d.get("steps", []))  # rows are already in the answer
    for c in d.get("citations", []):
        if c["type"] == "sql":
            page = f" · page {c['page']} of {c['pages']}" if c.get("pages", 1) > 1 else ""
            with st.expander(f"[{c['id']}] {c['row_count']} rows{page}", expanded=not page_reply):
                st.code(c["sql"], language="sql")
                if not page_reply:
                    df = pd.DataFrame(c["rows"], columns=c["columns"])
                    st.dataframe(df, hide_index=True)
                    chart(df)
        else:
            st.markdown(f"**[{c['id']}]** {c['title']} — {c['description']}")
    trace = (f" · [MLflow trace]({d['trace_url']})" if d.get("trace_url")
             else f" · MLflow trace {d['trace_id']}" if d.get("trace_id") else "")
    st.caption(" → ".join(f"{s['node']} {s['seconds']}s" for s in d.get("steps", [])) + trace)
    failed = [q for q in d.get("sql", []) if "error" in q]
    if failed:
        with st.expander(f"{len(failed)} failed SQL attempt(s)"):
            for q in failed:
                st.code(q["sql"], language="sql")
                st.caption(q["error"])


for q, d in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        show(d)

picked = None
if not st.session_state.history:
    st.caption("Try one of these:")
    for col, ex in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        if col.button(ex, use_container_width=True):
            picked = ex

if q := (st.chat_input("Ask about films, actors, customers, rentals, payments…") or picked):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("Working (local model, this can take a minute)…"):
            try:
                r = httpx.post(f"{API}/ask", json={"question": q,
                                                   "conversation_id": st.session_state.conversation}, timeout=600)
                r.raise_for_status()
                d = r.json()
                st.session_state.conversation = d.get("conversation_id")
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()
        show(d)
    st.session_state.history.append((q, d))
