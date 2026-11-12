# The agent as an MCP server (Streamable HTTP on :8002). Used by the "mcp" service in docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd --create-home app
USER app

EXPOSE 8002
CMD ["python", "-m", "dvd_agent.mcp_server", "--http", "--host", "0.0.0.0", "--port", "8002"]
