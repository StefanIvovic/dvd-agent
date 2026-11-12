# Running this project (for coding agents)

Follow README.md for setup. Notes that matter when you drive it:

- Services: `docker compose up -d`. Check with `docker compose ps` (postgres must be healthy) and
  `curl http://127.0.0.1:5001` (MLflow). The first start restores `database/dvdrental.tar`; to restore
  again, `docker compose down -v` (deletes the data) and start again.
- LM Studio must serve bge-m3 (embeddings) for every setup. Check: `lms ps`. The app loads it on the CPU
  at startup if missing. Never load a model with a JIT request: it can unload the other model.
- Use 127.0.0.1, not localhost, for LM Studio and the services.
- `.env` is required: copy `.env.example` and pick one chat-model block. Ask the user for API keys; don't
  copy secrets from other files or make them up. Never commit `.env`.
- Before asking questions: `python -m dvd_agent.sync.flows` (embeds the films). Done when
  `GET /stats` or the flow log shows `with_embedding` = `films`.
- Smoke test: `python -m dvd_agent.cli "Which store made more revenue in 2007?"` should answer
  store 2 with $30,683.13 and a [Q1] citation.
- Health: `GET /health` reports postgres, embeddings_current, llm and embed_model_loaded.
- The agent's SQL runs as `dvd_reader` (read-only). Don't weaken that role or the guard in
  `agent/sql_guard.py` to make a question work.
- MCP server: `docker compose up -d --build mcp`, then `http://127.0.0.1:8002/mcp`. The container can't
  load models: bge-m3 must be loaded in LM Studio on the host first (it refuses to embed otherwise,
  rather than trigger a JIT load). Ask the user for `MCP_TOKEN`; don't make one up.
- Evals take about 1 h with a local 27B model (`python -m dvd_agent.eval.run --split dev`). Ask before
  running them; with a cloud model they cost money.
- Do not change files in data/eval/ to make an eval pass.
