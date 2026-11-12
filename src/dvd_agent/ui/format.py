"""Text formatting for the Streamlit UI (kept apart from the app so it can be tested without Streamlit)."""
import re

import sqlglot

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
