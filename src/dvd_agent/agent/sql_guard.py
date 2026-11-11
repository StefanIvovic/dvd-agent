"""Checks the LLM's SQL before it runs, and gives errors the LLM can act on.

The database is the real boundary (dvd_reader: SELECT only, read-only, 10 s timeout). This guard
rejects anything that is not a single read query early, with a clear message, blocks hidden
columns, checks table names and caps the row count.
"""
from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

_WRITE_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
                exp.Command, exp.Set, exp.Transaction, exp.Commit, exp.Rollback, exp.Copy, exp.Grant)
_BLOCKED_FUNCS = re.compile(r"^(pg_sleep\w*|pg_read\w*|pg_ls_dir|pg_stat_file|pg_terminate_backend|"
                            r"pg_cancel_backend|set_config|dblink\w*|lo_\w+|pg_reload_conf|"
                            r"pg_advisory\w*|txid_\w+|query_to_xml\w*)$", re.I)


class UnsafeSQL(ValueError):
    """The SQL was rejected before running; the message is shown to the LLM for a retry."""


def clean(sql: str) -> str:
    sql = re.sub(r"^\s*```\w*|```\s*$", "", sql.strip()).strip()
    return sql.rstrip(";").strip()


def check(sql: str, known_tables: set[str], blocked_columns: set[tuple[str, str]], row_limit: int) -> str:
    """Returns the SQL to run (with a LIMIT), or raises UnsafeSQL."""
    sql = clean(sql)
    if not sql:
        raise UnsafeSQL("empty SQL")
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQL(f"SQL syntax error: {str(e)[:300]}") from None
    if len(statements) != 1:
        raise UnsafeSQL("write exactly one SQL statement")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise UnsafeSQL("only SELECT queries are allowed")
    for node in tree.walk():
        if isinstance(node, _WRITE_NODES) or (isinstance(node, exp.Select) and node.args.get("into")):
            raise UnsafeSQL("only read-only SELECT queries are allowed")
        if isinstance(node, (exp.Anonymous, exp.Func)):
            name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            if _BLOCKED_FUNCS.match(name or ""):
                raise UnsafeSQL(f"function {name} is not allowed")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = {}
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if t.db and t.db.lower() not in ("public",):
            raise UnsafeSQL(f"only tables in the public schema are allowed (got {t.db}.{t.name})")
        if name in cte_names:
            continue
        if name not in known_tables:
            raise UnsafeSQL(f"unknown table {t.name}; tables: {', '.join(sorted(known_tables))}")
        tables[t.alias_or_name.lower()] = name
    used = set(tables.values())

    for table, column in blocked_columns:
        if table not in used:
            continue
        for col in tree.find_all(exp.Column):
            if col.name.lower() == column and (not col.table or tables.get(col.table.lower()) == table):
                raise UnsafeSQL(f"column {table}.{column} is not available")
        for star in tree.find_all(exp.Star):
            parent = star.parent
            if isinstance(parent, exp.Func):   # count(*)
                continue
            qualifier = parent.table.lower() if isinstance(parent, exp.Column) and parent.table else ""
            if qualifier:
                hit = tables.get(qualifier) == table
            else:  # bare *: only tables read directly by this SELECT (not a CTE or subquery that lists columns)
                sel = star.find_ancestor(exp.Select)
                hit = sel is not None and any(
                    t.name.lower() == table and t.find_ancestor(exp.Select) is sel for t in sel.find_all(exp.Table))
            if hit:
                raise UnsafeSQL(f"list the columns you need instead of * ({table} has columns that are not available)")

    limit = tree.args.get("limit")
    current = None
    if limit is not None:
        try:
            current = int(limit.expression.name)
        except (AttributeError, ValueError, TypeError):
            current = None
    if current is None or current > row_limit:
        tree = tree.limit(row_limit)
    return tree.sql(dialect="postgres")
