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
```

| Field | Required | Meaning |
|---|---|---|
| `question` | yes | What the agent is asked. |
| `expect` | yes | `answer`: the agent should answer; saying "not found" fails.<br>`not_found`: the database doesn't have it; inventing an answer fails.<br>`correction`: false premise; the agent should say not found or correct it. |
| `gold_sql` | `answer` | Run at eval time as `dvd_reader`, so expected values follow the data. At most 20 rows. |
| `text_col` | `answer`* | Column(s) whose value in every gold row must appear in the answer (case-insensitive). Ties are rows too: use `FETCH FIRST n ROWS WITH TIES`. |
| `number_col` | `answer`* | Column(s) whose value in every gold row must appear as a number in the answer. |
| `tolerance` | no | Relative tolerance for numbers (default 0.005; 0.01 absolute for small values). |
| `contains` | `correction` | At least one phrase must appear, e.g. "only two". |
| `id`, `why` | no | Notes; not scored. |

\* at least one of `text_col` / `number_col`.

## What a run does

- Every gold_sql is checked first; a broken one stops the run.
- Every line is asked once; `not_found` and `correction` lines 3 times (`--repeats`).
- MLflow gets one table (each run: pass/fail, reason, the agent's SQL) and these totals:

| Metric | Meaning |
|---|---|
| `answer_pass_rate` | `answer` lines answered correctly |
| `false_refusal_rate` | `answer` lines where the agent said "not found" |
| `made_up_rate` | `not_found` lines where the agent invented an answer |
| `correction_pass_rate` | false premises refused or corrected |
| `sql_error_rate` | runs where at least one SQL attempt failed (and was retried) |
| `*_consistency` | same result on every repeat |
