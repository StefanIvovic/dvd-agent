# DVD Rental Agent

Ask questions in plain English about the [dvdrental](https://neon.com/postgresql/getting-started/sample-database)
sample database (films, actors, customers, stores, rentals, payments) and get answers with the SQL behind them.
It writes read-only SQL, finds films by plot or theme (hybrid BM25 + vector search), checks its own answer,
and says "not in the data" instead of guessing.

Stack: LangGraph, DSPy (few-shot tuned SQL step), LlamaIndex, PostgreSQL (pgvector + pg_search), MLflow, Prefect, FastAPI, Streamlit, MCP.

## How it works

1. A follow-up ("who acts in it?") is rewritten into a standalone question from the conversation.
2. Plot or theme questions search the film descriptions (BM25 + vectors) for matching films.
3. The model writes SQL from the live schema; a guard checks it and it runs as a read-only role. Errors go back to the model.
4. The answer cites every SQL result it uses ([Q1], [S1]); a self-check compares it with the results and asks for
   another query if it is incomplete, or corrects a false premise in the question.
5. Answers the data can't support become "not in the data". Each step is traced in MLflow.

## Requirements

- Docker
- Python 3.11+
- [LM Studio](https://lmstudio.ai) for the embedding model (bge-m3, runs on CPU), and for the chat model if you run it locally
- For a local chat model: a 24 GB GPU and 32 GB+ RAM. Otherwise an Anthropic or OpenAI API key.

## Setup

Download the [sample database](https://neon.com/postgresql/getting-started/sample-database) into `database/`
before the first `docker compose up` (Postgres restores it only when its data volume is new):

```bash
python -c "import io, urllib.request, zipfile; zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://neon.com/postgresqltutorial/dvdrental.zip').read())).extractall('database')"
```

```bash
docker compose up -d     # Postgres :5434 (restores database/dvdrental.tar on first start), MLflow :5001, Prefect :4201

python -m venv .venv
.venv\Scripts\activate                # Windows
source .venv/bin/activate             # macOS / Linux
pip install -e .

cp .env.example .env                  # Windows: copy .env.example .env
```

Download the models in LM Studio:

```bash
lms get https://huggingface.co/gpustack/bge-m3-GGUF                  # embeddings (Q8_0), always needed
lms get https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF   # chat (Q4_K_M), only for the local setup
```

Embed the film descriptions for search (about 2 minutes on CPU):

```bash
python -m dvd_agent.sync.flows
```

Using Claude Code or Codex? The agent instructions are in [AGENTS.md](AGENTS.md).

## Chat model

**Local (tested):** LM Studio with `qwen3.8-27b` (Q4_K_M) on an RTX 3090 (24 GB) with 40 GB RAM. This is the default in `.env.example`.

```bash
lms load qwen3.8-27b --gpu max -c 30976 --identifier qwen3.8-27b
```

- In LM Studio's settings for the model, turn off **Keep model in memory** and **Try mmap()**. Otherwise LM Studio holds about 30 GB of RAM.
- bge-m3 is loaded on the CPU automatically at startup (`EMBED_GPU_OFFLOAD=off`).
- Use `127.0.0.1`, not `localhost`, in `.env` (on Windows `localhost` adds a ~2 s delay per connection).

**Anthropic or OpenAI:** comment out the LM Studio block in `.env` and uncomment one of the others.

## Ask

```bash
python -m dvd_agent.cli "Which film category brought in the most revenue in April 2007?"
uvicorn dvd_agent.api.main:app --port 8001          # API, docs at http://127.0.0.1:8001/docs
streamlit run src/dvd_agent/ui/streamlit_app.py     # UI at http://127.0.0.1:8501 (needs the API running)
```

Long lists are paged, 50 rows per page: the answer shows page 1 and the page count. Then ask "page 2",
"pages 2, 5", "next page", "last page" or "20 per page" (also in the question: "List all customers, 20 per
page"). Each page runs the query again (no cached rows) and is printed straight from the database. In the CLI,
start without a question for a conversation: `python -m dvd_agent.cli`.

Follow-up questions use the last 3 turns of the conversation ("which of them are rated R?"). The Streamlit UI has
a Manual page, and a developer mode that shows the SQL, the steps and the MLflow trace of each answer.

API: `POST /ask` with `{"question": "...", "conversation_id": "..."}` (send back the `conversation_id` of the
previous answer for follow-ups and paging); also `/schema`, `/stats`, `/sync`, `/health`. Traces of every question are in MLflow at http://127.0.0.1:5001.

The agent's SQL runs as the `dvd_reader` role: SELECT only, read-only transactions, 10 s timeout.
It has no access to staff usernames and passwords, street addresses, postal codes or phone numbers
(column grants in `sql/init.sql`).

## MCP server

The agent as tools for Claude Code, Claude Desktop or any MCP client: `ask` (the agent, with its
self-check and follow-ups), `page` (more pages of a list), `run_sql` (your own read-only SELECT),
`search_films` and `schema`.

```bash
docker compose up -d --build mcp        # HTTP at http://127.0.0.1:8002/mcp
claude mcp add --transport http dvd-agent http://127.0.0.1:8002/mcp
```

Or without Docker, started by the client over stdio:

```bash
claude mcp add dvd-agent -- <path to repo>/.venv/Scripts/python -m dvd_agent.mcp_server   # macOS/Linux: .venv/bin/python
```

The container uses LM Studio on the host; bge-m3 must already be loaded there (`lms ps`). It is published on
127.0.0.1 only. To reach it from other machines, publish the port wider and set `MCP_TOKEN` in `.env`
(clients send `Authorization: Bearer <token>`).

## Data changes

Questions always run against the live data, and new tables or columns are picked up within 5 minutes.
Film embeddings follow edits: new or changed films are embedded by the sync flow, and deleted films lose theirs.

```bash
python -m dvd_agent.sync.flows --serve     # Prefect deployment, runs every 10 minutes (UI: http://127.0.0.1:4201)
```

## Tests

```bash
pip install -e ".[dev]"
pytest          # ~20 s, no LLM; tests marked db need Postgres running and are skipped without it
```

## Evaluate

```bash
python -m dvd_agent.eval.run --split dev
python -m dvd_agent.eval.run --split test
```

About 15-20 minutes per split with the local model.

The questions are in `data/eval/` ([format](data/eval/README.md)); expected values come from SQL run at eval time.

## Optimization

The SQL step's prompt carries 4 worked examples, picked by DSPy `BootstrapFewShot` from dev
questions the model got right (`src/dvd_agent/agent/compiled_program.json`, loaded by default;
`COMPILED_PROGRAM_PATH=none` runs the untuned prompts).

Final answers were already near 100%, because the self-check sends a wrong first query back for
another round. So the target was the SQL step on its own, by execution accuracy: does the first
query's result hold what `gold_sql` returns? 47 dev and 44 test questions (expect=answer, no
follow-ups), the same saved inputs for every program, 2 runs each.

| | Untuned | Tuned |
|---|---|---|
| SQL right on first try, dev | 43, 45 / 47 | 46, 46 / 47 |
| SQL right on first try, test | 42, 43 / 44 | 44, 43 / 44 |
| Full agent eval, test | 80 / 80 | 80 / 80 |
| Full agent eval, dev | 87 / 88 | 86 / 88 (d40 was noise: 3/3 on re-run, tuned and untuned) |
| Mean seconds per question, test / dev | 10.9 / 11.6 | 9.0 / 10.2 (one run each) |

- On test, about one more question right on the first try. The clearest case: one example uses
  `active = 1`, and the unseen "inactive customers" question stopped failing with `active = true`
  (0/2 to 2/2).
- Dev gains are partly in-sample: one example is a dev question.
- No change in final answers, which were already right; d38 (submarine count) is still wrong.
- Promoted because test didn't get worse and the first query got better. One run per full eval,
  so the speed figure is a hint, not a measurement.

The metric first failed correct queries that returned a month as 4 or `2007-04` rather than
"April" (d08, t07). It now reads month columns; every gold query passes it, and unit tests cover
the cases it must fail (a wrong count, a missing name, all 599 customers for "the top one").

```bash
python -m dvd_agent.optimize inputs  --split dev     # then --split test: inputs saved once
python -m dvd_agent.optimize measure --split dev     # first-try accuracy, untuned
python -m dvd_agent.optimize compile                 # BootstrapFewShot on dev only
python -m dvd_agent.optimize measure --split test --program artifacts/candidate_program.json
COMPILED_PROGRAM_PATH=artifacts/candidate_program.json python -m dvd_agent.eval.run --split test
```

## License

MIT
