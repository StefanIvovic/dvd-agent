# DVD Rental Agent

Ask questions in plain English about the [dvdrental](https://neon.com/postgresql/getting-started/sample-database)
sample database (films, actors, customers, stores, rentals, payments) and get answers with the SQL behind them.
It writes read-only SQL, finds films by plot or theme (hybrid BM25 + vector search), checks its own answer,
and says "not in the data" instead of guessing.

Stack: LangGraph, DSPy, LlamaIndex, PostgreSQL (pgvector + pg_search), MLflow, Prefect, FastAPI, Streamlit.

## Requirements

- Docker
- Python 3.11+
- [LM Studio](https://lmstudio.ai) for the embedding model (bge-m3, runs on CPU), and for the chat model if you run it locally
- For a local chat model: a 24 GB GPU and 32 GB+ RAM. Otherwise an Anthropic or OpenAI API key.

## Setup

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

API: `POST /ask` with `{"question": "...", "conversation_id": "..."}` (send back the `conversation_id` of the
previous answer to page its list); also `/schema`, `/stats`, `/sync`, `/health`. Traces of every question are in MLflow at http://127.0.0.1:5001.

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

## Evaluate

```bash
python -m dvd_agent.eval.run --split dev
python -m dvd_agent.eval.run --split test
```

The questions are in `data/eval/` ([format](data/eval/README.md)); expected values come from SQL run at eval time.

## License

MIT
