"""Streamlit front end for the API.  streamlit run src/dvd_agent/ui/streamlit_app.py

Calls POST /ask on the FastAPI service (API_URL), shows the answer, every SQL result as a table and
a bar chart when a result has one label column and one numeric column.
"""
import httpx
import pandas as pd
import streamlit as st

from dvd_agent.config import get_settings

API = get_settings().api_url
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
    for c in d["citations"]:
        if c["type"] == "sql":
            with st.expander(f"[{c['id']}] {c['row_count']} rows", expanded=True):
                st.code(c["sql"], language="sql")
                df = pd.DataFrame(c["rows"], columns=c["columns"])
                st.dataframe(df, hide_index=True)
                chart(df)
        else:
            st.markdown(f"**[{c['id']}]** {c['title']} — {c['description']}")
    st.caption(" → ".join(f"{s['node']} {s['seconds']}s" for s in d["steps"])
               + (f" · MLflow trace {d['trace_id']}" if d.get("trace_id") else ""))
    failed = [q for q in d["sql"] if "error" in q]
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

if q := st.chat_input("Ask about films, actors, customers, rentals, payments…"):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("Working (local model, this can take a minute)…"):
            try:
                r = httpx.post(f"{API}/ask", json={"question": q}, timeout=600)
                r.raise_for_status()
                d = r.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()
        show(d)
    st.session_state.history.append((q, d))
