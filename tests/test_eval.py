"""The eval's own logic: line validation and grading. A wrong grader would make every score wrong."""
from pathlib import Path

import pytest

from dvd_agent.eval import cases, metrics

ROOT = Path(__file__).resolve().parents[1]


def test_eval_files_are_valid(monkeypatch):
    monkeypatch.chdir(ROOT)   # cases.load reads data/eval/ from the repo root
    assert len(cases.load("dev")) > 50
    assert len(cases.load("test")) > 50   # also checks that no test question is in dev


@pytest.mark.parametrize("row, problem", [
    ({"question": "q", "expect": "answer"}, "needs 'gold_sql'"),
    ({"question": "q", "expect": "maybe"}, "'expect' must be one of"),
    ({"question": "q", "expect": "correction"}, "needs 'contains'"),
    ({"question": "q", "expect": "pages", "gold_sql": "x", "text_col": "t", "context": ["a"]}, "takes no 'context'"),
    ({"question": "q", "expect": "answer", "gold_sql": "x", "text_col": "t", "context": []}, "non-empty list"),
])
def test_invalid_lines(row, problem):
    assert any(problem in p for p in cases._check_row(row))


def test_numbers():
    assert metrics.numbers_in("1,234.5 and -3") == [1234.5, -3.0]
    assert metrics.numeric_match("total $30,683.13", 30683.13)
    assert not metrics.numeric_match("about 30,000", 30683.13)


@pytest.mark.parametrize("expect, status, passed", [
    ("not_found", "not_found", True),
    ("not_found", "answered", False),     # made up an answer
])
def test_not_found_grading(expect, status, passed):
    assert metrics.check({"expect": expect}, "some answer", status)[0] is passed


def test_citation_check():
    assert metrics.has_citation("34 rentals [Q1].")
    assert not metrics.has_citation("34 rentals.")
