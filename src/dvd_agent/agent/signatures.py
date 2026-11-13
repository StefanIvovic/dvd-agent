"""DSPy signatures and the optimizable program used by the LangGraph agent."""
from __future__ import annotations

from pathlib import Path

import dspy

from ..config import get_settings


class ContextualizeQuestion(dspy.Signature):
    """Rewrite the user's latest message in a conversation about a DVD rental database as a standalone
    question that can be answered without the conversation.

    - Replace references to earlier turns ("this movie", "them", "that store", "the second one", "the
      same for 2006", "and for Horror?") with the names, titles, values or filters they refer to,
      taken from the conversation.
    - If the message is already a complete question, or starts a new topic, return it unchanged.
    - Keep the user's wording, numbers and premises as they are, even if they look wrong. Do not answer
      the question, and add no facts, filters or limits the user did not ask for."""

    conversation: str = dspy.InputField(desc="earlier turns, oldest first")
    message: str = dspy.InputField(desc="the user's latest message")
    question: str = dspy.OutputField(desc="the standalone question")


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
      If the period asked is outside a table's date range, the data does not cover it: do not query it
      as if it were zero, and do not wrap sums or counts in COALESCE(..., 0). Query the date range of
      the table instead (min/max of its date column) so the answer can say what the data covers.
    - Durations between timestamps: EXTRACT(EPOCH FROM (end_ts - start_ts)) / 86400 for days (or / 3600
      for hours). Do not use EXTRACT(DAY FROM interval): it drops the hours.
    - Match names and titles case-insensitively (ILIKE). Return readable columns (names, titles) plus the
      numbers the answer needs; round money to 2 decimals.
    - For "top N" / "most" / "least" questions, include ties: use RANK() or ORDER BY <ranking value>
      FETCH FIRST N ROWS WITH TIES. WITH TIES keeps only rows equal on every ORDER BY column, so order
      by the ranking value alone (no name or title tie-breakers). Never use LIMIT N or FETCH ... ROWS
      ONLY for them: that drops tied rows.
    - For lists ("list all ...", "which films ..."), return every matching row: do not add a LIMIT (the
      user gets the rows page by page), ORDER BY a readable column (e.g. title or last name), and select
      only the columns the question needs (titles or names, not descriptions unless asked).
    - film_hits are only the top matches of a plot/theme search, not every matching film. If the theme
      is a concrete word or phrase that would appear in the text (robot, shark, Berlin, "mad scientist"),
      filter with film.description ILIKE '%word%' (and title if relevant) so all films are covered. Use
      film_id IN (...) with the relevant film_hits only for vaguer themes (e.g. "films about space travel").
      Take the ILIKE patterns from the wording the film_hits descriptions actually use for the theme
      (e.g. a setting phrase such as 'in The Gulf of Mexico'), and do not OR in loosely related words that
      would match other films.
    - If the question states a fact as given (a count such as "her 60 films", a total, a date, a store or
      item that should exist), add a query that checks it (e.g. count her films), so the answer can
      confirm or correct it.
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
    periods. When results are tied, say so and list the tied items. For "top N" / "N shortest" style
    questions, if items beyond the Nth have the same value as the Nth, the top N is cut inside a tie:
    say so and list every tied item (or say how many are tied if there are many), not just N of them.
    A period outside the date range of the data has no data; say that, do not report it as 0. If a result is paged ("page 1 shown"),
    give the total row count and, for a list, list every row of page 1; a page note is added for the
    user automatically, so do not call the list partial or incomplete. If the sources do not contain the answer,
    say what is missing instead of guessing. If the question asks for an exact figure the data does not
    have (e.g. the revenue for a month with no payments), say so first, then give the closest related
    figure with its citation. If the question states something the results contradict (a count, a
    date, a name, a store or item that does not exist), say so first and give the correct value.
    Be concise: a direct answer first, then supporting detail."""

    question: str = dspy.InputField()
    sources: str = dspy.InputField(desc="[Q#] SQL statements with result tables, [S#] film search hits")
    review: str = dspy.InputField(desc="problems a reviewer found in an earlier draft; fix them. Empty on the first draft")
    answer: str = dspy.OutputField(desc="markdown answer with [Q#]/[S#] citations")


class AssessAnswer(dspy.Signature):
    """Check a draft answer against its sources, claim by claim.

    - 'grounded' is false if ANY factual claim is not supported by the cited sources. Numbers, names
      and titles must match the result rows. A statement that the data does not contain some
      information is grounded.
    - 'complete' is false if part of the question is unanswered, or a SQL result does not actually
      answer what was asked (wrong filter, wrong period, missing grouping). When the results show a
      tie, listing all tied items is complete, even if the question asked for a single one. A paged
      result ("page 1 shown") is complete when the answer gives the total and lists page 1: the user
      can ask for every other page, so never ask a new query for the remaining rows or pages. A count or
      list of films by plot, theme or setting is complete when its query filters the description on the
      wording the descriptions use for that theme (e.g. ILIKE '%in A U-Boat%' for "set on a submarine"):
      the database has no setting or theme field, so that is the best answer it can give. Do not mark it
      incomplete for that reason, and do not ask for broader or related keywords.
    - 'answer_found' is false if the sources contain no information that answers the question at all
      (e.g. the answer only says the information is not available). It is true if the answer says the
      exact figure is not available but gives a related figure from the sources; 'complete' is then false.
    - If not grounded or not complete, say in follow_up what a new query should compute.
    - 'premise_conflict': if the question states as fact a number, date, name or the existence of
      something (e.g. "her 60 films", "store 3", "since 2010") that the sources contradict, and the
      answer does not point that out, write what the question says and what the sources show (e.g.
      "the question says 60 films; [Q2] counts 42"). Only when a source shows a different value: empty
      if the question states no such fact, no source checks it, the sources agree with it, or the
      answer already corrects it."""

    question: str = dspy.InputField()
    sources: str = dspy.InputField()
    answer: str = dspy.InputField()
    grounded: bool = dspy.OutputField()
    complete: bool = dspy.OutputField()
    answer_found: bool = dspy.OutputField()
    missing: str = dspy.OutputField(desc="what is missing or unsupported; empty if nothing")
    follow_up: str = dspy.OutputField(desc="what a new SQL query should compute, if not grounded or not complete; else empty")
    premise_conflict: str = dspy.OutputField(desc="a fact stated in the question that the sources contradict and the answer did not correct; else empty")


class DvdQAProgram(dspy.Module):
    """Container for every LLM step, so DSPy can optimize, save and load them together."""

    def __init__(self):
        super().__init__()
        self.contextualize = dspy.Predict(ContextualizeQuestion)
        self.analyze = dspy.Predict(AnalyzeQuestion)
        # ChainOfThought: a hidden scratchpad before the SQL / the answer (qwen's thinking is off)
        self.sql = dspy.ChainOfThought(GenerateSQL)
        self.answer = dspy.ChainOfThought(AnswerQuestion)
        self.assess = dspy.Predict(AssessAnswer)


PACKAGED_PROGRAM = Path(__file__).with_name("compiled_program.json")


def load_program() -> DvdQAProgram:
    """The program the agent runs: COMPILED_PROGRAM_PATH, or the packaged tuned one by default.
    "none" gives the untuned prompts. A path that doesn't exist is an error, not a silent fallback."""
    program = DvdQAProgram()
    setting = get_settings().compiled_program_path.strip()
    if setting.lower() == "none":
        return program
    path = Path(setting) if setting else PACKAGED_PROGRAM
    if not path.exists():
        raise FileNotFoundError(f"COMPILED_PROGRAM_PATH {path} not found (use 'none' for the untuned prompts)")
    program.load(str(path))
    return program
