"""The real boundary: the dvd_reader role, even when SQL gets past the guard."""
import pytest

from dvd_agent.db import postgres as pg

pytestmark = pytest.mark.db


@pytest.mark.parametrize("sql", [
    "DELETE FROM rental WHERE rental_id = -1",
    "UPDATE film SET title = title WHERE film_id = -1",
    "CREATE TABLE x (a int)",
])
def test_reader_cannot_write(sql):
    with pytest.raises(Exception, match="read-only|permission denied"):
        pg.run_readonly(sql)


@pytest.mark.parametrize("sql", [
    "SELECT password FROM staff",
    "SELECT username FROM staff",
    "SELECT phone FROM address",
    "SELECT address FROM address",
    "SELECT * FROM staff_list",
])
def test_reader_cannot_read_hidden_columns(sql):
    with pytest.raises(Exception, match="permission denied"):
        pg.run_readonly(sql)


def test_reader_can_read():
    r = pg.run_readonly("SELECT count(*) AS n FROM film")
    assert r.rows == [{"n": 1000}]


def test_statement_timeout():
    with pytest.raises(Exception, match="timeout|cancel"):
        pg.run_readonly("SELECT pg_sleep(11)")
