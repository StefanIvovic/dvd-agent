# Eval questions

| File | Use |
|---|---|
| `dev.jsonl` | Tune against these freely. Scores here are optimistic. |
| `test.jsonl` | Held out. Run it only to report a score, never to decide what to fix. If a test failure leads to a code change, move that question to `dev.jsonl` and write a new one. |

```
python -m dvd_agent.eval.gold --split dev     # show what each gold_sql returns (review before running)
python -m dvd_agent.eval.run --split dev
python -m dvd_agent.eval.run --split test --run-name test-v1
```

## Line format

```json
{"question": "Which customer has spent the most?", "expect": "answer", "gold_sql": "SELECT ... AS name, sum(amount) AS v ...", "text_col": "name", "number_col": "v"}
{"question": "Who directed Ace Goldfinger?", "expect": "not_found", "why": "no director column"}
{"question": "Why did store 3 make the least revenue?", "expect": "correction", "contains": ["only two", "no store 3"]}
{"context": ["What is the most rented movie?"], "question": "Which actors are in it?", "expect": "answer", "gold_sql": "...", "text_col": "name"}
{"question": "List all customers of store 2.", "expect": "pages", "gold_sql": "SELECT first_name || ' ' || last_name AS name FROM customer WHERE store_id = 2", "text_col": "name", "follow_ups": [{"say": "20 per page", "pages": [1], "size": 20}, {"say": "pages 3, 5", "pages": [3, 5]}]}
```

| Field | Required | Meaning |
|---|---|---|
| `question` | yes | What the agent is asked. |
| `expect` | yes | `answer`: the agent should answer; saying "not found" fails.<br>`not_found`: the database doesn't have it; inventing an answer fails.<br>`correction`: false premise; the agent should say not found or correct it.<br>`pages`: a list question, then `follow_ups` in the same conversation. |
| `gold_sql` | `answer`, `pages` | Run at eval time as `dvd_reader`, so expected values follow the data. At most 20 rows for `answer`; the full list for `pages`. |
| `text_col` | `answer`*, `pages` | For `pages`: the one column that identifies a row (title, name). For `answer`: column(s) whose value in every gold row must appear in the answer (case-insensitive). Ties are rows too: use `FETCH FIRST n ROWS WITH TIES`. |
| `number_col` | `answer`* | Column(s) whose value in every gold row must appear as a number in the answer. |
| `tolerance` | no | Relative tolerance for numbers (default 0.005; 0.01 absolute for small values). |
| `contains` | `correction` | At least one phrase must appear, e.g. "only two". |
| `follow_ups` | no (`pages`) | Paging requests: `say` is sent, `pages` are the page numbers it should return, `size` the rows per page from then on (default: unchanged). |
| `context` | no (not `pages`) | Earlier questions, asked first in the same conversation; only `question` is graded. Its gold_sql must stand alone (e.g. find the most rented film itself). |
| `size` | no (`pages`) | Rows per page the question itself asks for (default 50). |
| `id`, `why` | no | Notes; not scored. |

\* at least one of `text_col` / `number_col`.

A `pages` line passes when page 1 (in the answer) and every requested page say "Page p of P" with the total
(only when there is more than one page), have the right number of rows, hold only rows of the gold list
(in the answer text too), and pages of the same size don't overlap. Pages that cover a whole page size must
together be the gold list. The order is the agent's: any order passes.

## What a run does

- Every gold_sql is checked first; a broken one stops the run.
- Every line is asked once; `not_found` and `correction` lines 3 times (`--repeats`; `--repeat-all` repeats every line).
- MLflow gets one table (each run: pass/fail, reason, the agent's SQL) and these totals:

| Metric | Meaning |
|---|---|
| `answer_pass_rate` | `answer` lines answered correctly |
| `false_refusal_rate` | `answer` lines where the agent said "not found" |
| `made_up_rate` | `not_found` lines where the agent invented an answer |
| `correction_pass_rate` | false premises refused or corrected |
| `followup_pass_rate` | lines with `context` (follow-up questions) answered correctly |
| `pages_pass_rate` | `pages` lines (list and every follow-up page) correct |
| `sql_error_rate` | runs where at least one SQL attempt failed (and was retried) |
| `*_consistency` | same result on every repeat |
