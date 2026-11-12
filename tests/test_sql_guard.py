"""The guard in front of every SQL statement the LLM (or an MCP client) writes."""
import pytest

from dvd_agent.agent.sql_guard import UnsafeSQL, check, clean

TABLES = {"film", "actor", "staff", "customer", "rental", "payment"}
BLOCKED = {("staff", "password"), ("customer", "email")}


def ok(sql: str) -> str:
    return check(sql, TABLES, BLOCKED)


@pytest.mark.parametrize("sql, message", [
    ("DELETE FROM film", "only SELECT"),
    ("UPDATE film SET title = 'x'", "only SELECT"),
    ("DROP TABLE film", "only SELECT"),
    ("SELECT title INTO new_table FROM film", "read-only"),
    ("SELECT 1; SELECT 2", "exactly one"),
    ("SELECT pg_sleep(5)", "pg_sleep is not allowed"),
    ("SELECT pg_read_file('/etc/passwd')", "not allowed"),
    ("SELECT set_config('x', 'y', false)", "not allowed"),
    ("SELECT * FROM pg_shadow", "unknown table"),
    ("SELECT table_name FROM information_schema.tables", "public schema"),
    ("SELECT password FROM staff", "staff.password is not available"),
    ("SELECT s.password FROM staff AS s", "staff.password is not available"),
    ("SELECT email FROM customer", "customer.email is not available"),
    ("SELECT * FROM staff", "instead of *"),
    ("SELECT s.* FROM staff AS s", "instead of *"),
    ("", "empty SQL"),
    ("SELEC title FROM film", "SQL syntax error"),
])
def test_rejects(sql, message):
    with pytest.raises(UnsafeSQL, match=message):
        ok(sql)


@pytest.mark.parametrize("sql", [
    "SELECT title FROM film",
    "SELECT count(*) FROM staff",                         # count(*) is not a column list
    "SELECT first_name FROM staff",                       # allowed column of a table with hidden ones
    "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
    "SELECT * FROM film",                                 # no hidden columns in film here
    "SELECT * FROM (SELECT first_name FROM staff) AS s",  # * over a subquery that lists its columns
])
def test_allows(sql):
    assert ok(sql)


def test_strips_markdown_fences_and_semicolon():
    assert clean("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert ok("```sql\nSELECT title FROM film;\n```") == "SELECT title FROM film"


@pytest.mark.parametrize("sql", [
    "SELECT title FROM film ORDER BY length DESC LIMIT 3",
    "SELECT title FROM film ORDER BY length DESC FETCH FIRST 3 ROWS ONLY",
])
def test_ordered_top_n_keeps_ties(sql):
    assert ok(sql).endswith("ORDER BY length DESC FETCH FIRST 3 ROWS WITH TIES")


@pytest.mark.parametrize("sql", [
    "SELECT title FROM film LIMIT 3",                          # no order: nothing to tie on
    "SELECT title FROM film ORDER BY 1 LIMIT 3 OFFSET 3",      # explicit paging stays as written
])
def test_other_limits_unchanged(sql):
    assert "TIES" not in ok(sql)
