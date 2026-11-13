"""The SQL-step metric: a first query passes only if its result holds what gold_sql returns."""
from decimal import Decimal

from dvd_agent.optimize import hits_gold


def res(rows, count=None):
    return {"sql": "q", "columns": list(rows[0]) if rows else [], "rows": rows,
            "row_count": len(rows) if count is None else count}


def test_numbers_and_texts_found():
    r = res([{"first_name": "ELEANOR", "last_name": "HUNT", "rentals": 46}])
    assert hits_gold([r], ["Eleanor Hunt"], [46], 1) == (True, "")


def test_decimal_and_rounding_within_tolerance():
    r = res([{"store_id": 2, "revenue": Decimal("33726.774")}])
    assert hits_gold([r], [], [33726.77], 1)[0]


def test_a_wrong_number_fails():
    r = res([{"n": 50}])
    ok, why = hits_gold([r], [], [46], 1)
    assert not ok and "46" in why


def test_a_missing_name_fails():
    r = res([{"title": "ACADEMY DINOSAUR"}])
    ok, why = hits_gold([r], ["Ace Goldfinger"], [], 1)
    assert not ok and "Ace Goldfinger" in why


def test_a_result_far_bigger_than_gold_does_not_count():
    """Returning every customer contains the top 1, but doesn't answer the question."""
    rows = [{"name": f"C{i}", "total": i} for i in range(599)]
    ok, why = hits_gold([res(rows[:50], count=599)], ["C598"], [598], 1)
    assert not ok and "small enough" in why


def test_a_few_extra_rows_are_fine():
    rows = [{"store_id": 1, "revenue": 33489.47}, {"store_id": 2, "revenue": 33726.77}]
    assert hits_gold([res(rows)], [], [33726.77], 1)[0]


def test_values_may_come_from_two_queries():
    a, b = res([{"n": 1000}]), res([{"n": 599}])
    assert hits_gold([a, b], [], [1000, 599], 1)[0]


def test_errors_only_fail_with_the_error():
    ok, why = hits_gold([{"sql": "q", "error": "UndefinedColumn: x"}], [], [1], 1)
    assert not ok and why.startswith("error: UndefinedColumn")


def test_booleans_are_not_numbers():
    assert not hits_gold([res([{"active": True}])], [], [1], 1)[0]


def test_a_month_column_answers_which_month():
    import datetime
    for v in (4, Decimal("4"), "2007-04", "2007-04-01", datetime.date(2007, 4, 1)):
        assert hits_gold([res([{"month": v, "total": 28559.46}])], ["April"], [28559.46], 1)[0], v


def test_a_number_outside_a_month_column_is_not_a_month():
    assert not hits_gold([res([{"store_id": 4, "total": 1.0}])], ["April"], [1.0], 1)[0]
    assert not hits_gold([res([{"month": 5, "total": 1.0}])], ["April"], [1.0], 1)[0]


def test_the_agent_loads_the_tuned_program_by_default(monkeypatch):
    from dvd_agent.agent import signatures
    from dvd_agent.config import get_settings

    monkeypatch.setattr(get_settings(), "compiled_program_path", "")
    tuned = signatures.load_program()
    assert len(tuned.sql.predict.demos) == 4
    assert all(len(p.demos) == 0 for name, p in tuned.named_predictors() if not name.startswith("sql"))

    monkeypatch.setattr(get_settings(), "compiled_program_path", "none")
    assert len(signatures.load_program().sql.predict.demos) == 0


def test_a_missing_program_path_is_an_error(monkeypatch, tmp_path):
    import pytest
    from dvd_agent.agent import signatures
    from dvd_agent.config import get_settings

    monkeypatch.setattr(get_settings(), "compiled_program_path", str(tmp_path / "nope.json"))
    with pytest.raises(FileNotFoundError):
        signatures.load_program()
