"""Paging: reading "page 2" style requests, rewriting SQL per page, and (db) pages that cover every row."""
import pytest

from dvd_agent.agent import paging


@pytest.mark.parametrize("text, expected", [
    ("page 2", {"pages": [2], "rel": None, "size": None}),
    ("Page 2 please", {"pages": [2], "rel": None, "size": None}),
    ("show me page two", {"pages": [2], "rel": None, "size": None}),
    ("pages 2, 5", {"pages": [2, 5], "rel": None, "size": None}),
    ("pages 3 to 5", {"pages": [3, 4, 5], "rel": None, "size": None}),
    ("pages 3-5", {"pages": [3, 4, 5], "rel": None, "size": None}),
    ("next page", {"pages": None, "rel": "next", "size": None}),
    ("more", {"pages": None, "rel": "next", "size": None}),
    ("previous page", {"pages": None, "rel": "prev", "size": None}),
    ("first page", {"pages": None, "rel": "first", "size": None}),
    ("last page", {"pages": None, "rel": "last", "size": None}),
    ("20 per page", {"pages": None, "rel": None, "size": 20}),
    ("pages 2-4, 20 per page", {"pages": [2, 3, 4], "rel": None, "size": 20}),
])
def test_parse_command(text, expected):
    assert paging.parse_command(text) == expected


@pytest.mark.parametrize("text", [
    "List all customers",
    "What is page 2 of the report about?",       # mentions a page, but is a question
    "Which store made more revenue in 2007?",
])
def test_questions_are_not_commands(text):
    assert paging.parse_command(text) is None


def test_split_question_takes_the_page_size_out():
    assert paging.split_question("List all customers, 20 per page") == ("List all customers", 20)
    assert paging.split_question("List all films") == ("List all films", None)


def test_page_sql():
    assert paging.page_sql("SELECT title FROM film ORDER BY title", 2, 50) == \
        "SELECT title FROM film ORDER BY title LIMIT 50 OFFSET 50"
    # a LIMIT the query already has is the end of the list: page 3 of 50 within LIMIT 120 is 20 rows
    assert paging.page_sql("SELECT title FROM film ORDER BY title LIMIT 120", 3, 50) == \
        "SELECT title FROM film ORDER BY title LIMIT 20 OFFSET 100"
    # WITH TIES can't take an OFFSET: wrapped
    assert paging.page_sql("SELECT title FROM film ORDER BY length FETCH FIRST 3 ROWS WITH TIES", 1, 50) == \
        "SELECT * FROM (SELECT title FROM film ORDER BY length FETCH FIRST 3 ROWS WITH TIES) AS _paged LIMIT 50 OFFSET 0"


def test_stable_sql_adds_every_column_as_tie_breaker():
    assert paging.stable_sql("SELECT title, length FROM film ORDER BY length", 2) == \
        "SELECT title, length FROM film ORDER BY length, 1, 2"


@pytest.mark.parametrize("total, size, pages", [(61, 50, 2), (50, 50, 1), (51, 50, 2), (0, 50, 1)])
def test_pages_for(total, size, pages):
    assert paging.pages_for(total, size) == pages


def test_conversation_state():
    paging.remember("c1", {"sql": "SELECT 1", "page_size": 50, "page": 1}, "Q1", "List all")
    assert paging.recall("c1")["sql"] == "SELECT 1"
    paging.forget("c1")
    assert paging.recall("c1") is None
    assert paging.recall(None) is None


# ------------------------------------------------------------------ against the database
SCIFI = ("SELECT f.title FROM film f JOIN film_category fc USING (film_id) JOIN category c USING (category_id) "
         "WHERE c.name = 'Sci-Fi' ORDER BY f.title")


@pytest.mark.db
def test_pages_cover_every_row_once():
    first = paging.run_paged(SCIFI, 20)
    assert (first["row_count"], first["pages"], len(first["rows"])) == (61, 4, 20)
    titles = [r["title"] for p in range(1, 5) for r in paging.run_paged(first["sql"], 20, p, first_run=False)["rows"]]
    assert len(titles) == 61 == len(set(titles))


@pytest.mark.db
def test_short_result_is_one_page():
    r = paging.run_paged("SELECT name FROM category ORDER BY name", 50)
    assert (r["row_count"], r["pages"], len(r["rows"])) == (16, 1, 16)


@pytest.mark.db
def test_page_number_is_clamped():
    assert paging.run_paged(SCIFI, 50, 99)["page"] == 2
