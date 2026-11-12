# Manual

Ask questions in plain English about a DVD rental business. The agent writes SQL, runs it against
the database (read-only), and answers from the results. If the data doesn't have the answer, it says so
instead of guessing.

## What you can ask

The database has **films** (title, description, length, rating, rental price, language), **actors**,
**categories**, **customers**, **stores**, **staff**, **inventory**, **rentals** and **payments**.

- Rentals run from May 2005 to February 2006. Payments only cover **February–May 2007**.
- Films can be found by what they are about: *“films about a shark”*, *“movies set in a submarine”*.
- Not in the data: box-office gross, critic or viewer reviews, directors, streaming.
- Hidden from the agent: staff usernames and passwords, street addresses, postal codes and phone numbers.

## Status badge

| Badge | Meaning |
|---|---|
| :green-badge[answered] | The question is fully answered from the data. |
| :orange-badge[partly answered] | Part of the question isn't in the data; the answer covers what is. |
| :red-badge[not in the data] | The data can't answer it, so the agent gives no answer rather than guess. |

## Sources ¹ ²

Every fact in an answer points to the result it came from: ¹ is the first query result below the answer,
² the second, and so on. A film found by plot or theme search is listed under **Film matches**.

## Data

The rows the answer is based on, with a bar chart when there is a label and a number per row. It is open
when it adds to the answer (a short table with several columns), folded for a one-column list or a long
table the answer already lists, and left out when the result is a single value the answer states.

## How this was computed

The SQL query that ran against the database. The agent's queries are read-only and limited to 10 seconds.

## Long lists

Lists longer than 50 rows are split into pages; the answer shows page 1 and the page count. Then ask:

- “page 2”, “pages 2, 5”, “pages 3 to 5”
- “next page”, “previous page”, “last page”
- “20 per page” (or in the question: “List all customers, 20 per page”)

Each page runs the query again, so nothing is cached, and pages are printed straight from the database.

## Follow-up questions

The agent remembers the last 3 questions and answers of a conversation for 30 minutes, so you can ask
*“who acts in it?”* or *“and in 2006?”*. **New conversation** (sidebar) starts fresh.

## Developer mode

The sidebar toggle shows what happened behind each answer:

- **Understood as**: how a follow-up question was rewritten into a standalone question
- **How this was computed** opened, and **Data** even for a single value
- **Failed SQL attempts**: queries that failed and were fixed on retry
- **Steps**: analyze → film search → sql → answer → assess (self-check) → finalize, with timings
- **MLflow trace**: every prompt, query and result of the answer

## Health

The sidebar panel checks the database, the film search embeddings and the language model.
