"""Small text helpers: conversation memory, citations, UI formatting, not-found detection."""
from dvd_agent.agent import memory
from dvd_agent.agent.graph import cited_sources
from dvd_agent.agent.tools import lead_not_in_sources_match
from dvd_agent.ui.format import pretty_sql, sup, with_superscripts


def test_memory_keeps_last_three_turns_without_citations():
    memory.clear("m1")
    for i in range(1, 5):
        memory.add("m1", f"question {i}", f"answer {i} [Q1]", "answered")
    h = memory.history("m1")
    assert "question 1" not in h and "question 4" in h
    assert "[Q1]" not in h
    memory.clear("m1")
    assert memory.history("m1") == ""


def test_memory_hides_not_found_drafts():
    memory.clear("m2")
    memory.add("m2", "who directed it?", "some text", "not_found")
    assert "not in the data" in memory.history("m2") and "some text" not in memory.history("m2")
    memory.clear("m2")


def test_memory_without_conversation():
    memory.add(None, "q", "a", "answered")
    assert memory.history(None) == ""


def test_cited_sources_in_order_and_only_known_ids():
    cmap = {"Q1": {"type": "sql"}, "Q2": {"type": "sql"}, "S1": {"type": "film"}}
    ids = [c["id"] for c in cited_sources("A [Q2] and [Q1, S1], not [Q9]", cmap)]
    assert ids == ["Q2", "Q1", "S1"]


def test_superscripts():
    numbers = {"Q1": 1, "S1": 2}
    assert sup(12) == "¹²"
    assert with_superscripts("34 rentals [Q1].", numbers) == "34 rentals¹."
    assert with_superscripts("both [Q1, S1]", numbers) == "both¹˒²"
    assert with_superscripts("unknown [Q7]", numbers) == "unknown"


def test_pretty_sql():
    assert pretty_sql("SELECT a, b FROM t WHERE a = 1") == "SELECT\n  a,\n  b\nFROM t\nWHERE\n  a = 1"
    assert pretty_sql("not sql at all (((") == "not sql at all ((("


def test_not_in_sources_detection():
    assert lead_not_in_sources_match("The data does not include box office figures.")
    assert not lead_not_in_sources_match("Store 2 made more revenue in 2007.")
