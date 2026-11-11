"""DSPy signatures and the optimizable program used by the LangGraph agent."""
from __future__ import annotations

from pathlib import Path

import dspy

from ..config import get_settings


class AnalyzeQuestion(dspy.Signature):
    """Plan how to answer a question about a DVD rental business database (films, actors, categories,
    customers, stores, staff, inventory, rentals, payments).

    Film search finds films by what they are about (plot, themes, characters, setting) using the
    title and description text, e.g. "films about a robot" or "movies set in a shark tank". It is
    not needed for questions that name a title, actor, category, rating or a number."""

    question: str = dspy.InputField()
    needs_film_search: bool = dspy.OutputField(desc="true only if films must be found by plot/theme/description")
    film_search_queries: list[str] = dspy.OutputField(desc="1-2 short queries describing the plot/theme; [] if no film search")


class GenerateSQL(dspy.Signature):
    """Write PostgreSQL SELECT queries that answer the question from the database schema.

    - Use only tables and columns in the schema. Prefer base tables over the *_list views.
    - Revenue/sales = SUM(payment.amount). A rental links to a film and a store through inventory
      (rental.inventory_id -> inventory.film_id, inventory.store_id). A store's revenue is the payments
      for rentals of its inventory (payment -> rental -> inventory.store_id), as in the sales_by_store
      view; customer.store_id is only the customer's home store. film.rental_rate is the price of one rental, rental_duration is in days,
      length is in minutes. Mind the date ranges in the schema: there may be no data for the period asked.
    - Match names and titles case-insensitively (ILIKE). Return readable columns (names, titles) plus the
      numbers the answer needs; round money to 2 decimals.
    - For "top N" / "most" questions, include ties: use RANK() or ORDER BY ... FETCH FIRST N ROWS WITH TIES.
    - film_hits are only the top matches of a plot/theme search, not every matching film. If the theme
      is a concrete word or phrase that would appear in the text (robot, shark, Berlin, "mad scientist"),
      filter with film.description ILIKE '%word%' (and title if relevant) so all films are covered. Use
      film_id IN (...) with the relevant film_hits only for vaguer themes (e.g. "films about space travel").
    - Write 1-3 queries; one is best when it can answer everything. Each query is a single SELECT
      (WITH ... SELECT is fine), no semicolons, no comments.
    - If the database cannot answer the question at all (e.g. box-office gross, critic reviews, viewer
      star ratings, streaming), return no queries and say why in cannot_answer.
    - If previous_attempts shows errors, fix them; if it says what is missing, write queries for that."""

    question: str = dspy.InputField()
    db_schema: str = dspy.InputField(desc="tables/views with columns, PK, FK (->), enum and known values, date ranges")
    film_hits: str = dspy.InputField(desc="films found by plot/theme search (JSON), or []")
    previous_attempts: str = dspy.InputField(desc="earlier queries with their errors or results summary; empty on the first try")
    queries: list[str] = dspy.OutputField(desc="SQL SELECT statements; [] if the database cannot answer")
    cannot_answer: str = dspy.OutputField(desc="why the database cannot answer; empty if queries are given")


class AnswerQuestion(dspy.Signature):
    """Answer the question about the DVD rental database using ONLY the provided sources.
    Cite every claim with its source id in square brackets, e.g. [Q1] for a SQL result or [S2] for a
    film search hit. Use the numbers exactly as in the results; state units (USD, days, minutes) and
    periods. When results are tied, say so and list the tied items. If a result reached the row limit
    or shows fewer rows than it has, say the list is partial. If the sources do not contain the answer,
    say what is missing instead of guessing. If the question asks for an exact figure the data does not
    have (e.g. the revenue for a month with no payments), say so first, then give the closest related
    figure with its citation.
    Be concise: a direct answer first, then supporting detail."""

    question: str = dspy.InputField()
    sources: str = dspy.InputField(desc="[Q#] SQL statements with result tables, [S#] film search hits")
    answer: str = dspy.OutputField(desc="markdown answer with [Q#]/[S#] citations")


class AssessAnswer(dspy.Signature):
    """Check a draft answer against its sources, claim by claim.

    - 'grounded' is false if ANY factual claim is not supported by the cited sources. Numbers, names
      and titles must match the result rows. A statement that the data does not contain some
      information is grounded.
    - 'complete' is false if part of the question is unanswered, or a SQL result does not actually
      answer what was asked (wrong filter, wrong period, missing grouping).
    - 'answer_found' is false if the sources contain no information that answers the question at all
      (e.g. the answer only says the information is not available). It is true if the answer says the
      exact figure is not available but gives a related figure from the sources; 'complete' is then false.
    - If not grounded or not complete, say in follow_up what a new query should compute."""

    question: str = dspy.InputField()
    sources: str = dspy.InputField()
    answer: str = dspy.InputField()
    grounded: bool = dspy.OutputField()
    complete: bool = dspy.OutputField()
    answer_found: bool = dspy.OutputField()
    missing: str = dspy.OutputField(desc="what is missing or unsupported; empty if nothing")
    follow_up: str = dspy.OutputField(desc="what a new SQL query should compute, if not grounded or not complete; else empty")


class DvdQAProgram(dspy.Module):
    """Container for every LLM step, so DSPy can optimize, save and load them together."""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.Predict(AnalyzeQuestion)
        # ChainOfThought: a hidden scratchpad before the SQL / the answer (qwen's thinking is off)
        self.sql = dspy.ChainOfThought(GenerateSQL)
        self.answer = dspy.ChainOfThought(AnswerQuestion)
        self.assess = dspy.Predict(AssessAnswer)


def load_program() -> DvdQAProgram:
    program = DvdQAProgram()
    path = Path(get_settings().compiled_program_path)
    if path.exists():
        program.load(str(path))
    return program
