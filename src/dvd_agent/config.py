from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # the project's .env (also when started from another folder), then ./.env
    model_config = SettingsConfigDict(env_file=(Path(__file__).resolve().parents[2] / ".env", ".env"),
                                      extra="ignore")

    llm_provider: str = "lmstudio"      # lmstudio | openai | anthropic
    llm_base_url: str = "http://127.0.0.1:1234/v1"   # not localhost: Windows tries ::1 first (+2.3 s per connection)
    llm_model: str = "qwen3.8-27b"
    llm_api_key: str = "lm-studio"
    llm_max_tokens: int | None = None   # default: 4000 for LM Studio (thinking off), 16000 for OpenAI/Anthropic
    llm_effort: str = ""                # OpenAI/Anthropic: low | medium | high (Anthropic also xhigh | max); "" = default
    llm_temperature: float = 0.1
    # sent explicitly so LM Studio's built-in defaults never apply
    llm_top_p: float = 0.9
    llm_top_k: int = 20
    llm_min_p: float = 0.0
    llm_repeat_penalty: float = 1.0      # >1 penalizes legitimately repeated tokens in SQL/tables
    llm_reasoning_effort: str = "none"   # LM Studio: "none" disables qwen thinking; "" = model default
    llm_serialize: bool = False          # one LM Studio call at a time; fallback if the embed model can't be pinned
    embed_base_url: str = "http://127.0.0.1:1234/v1"  # LM Studio serves bge-m3 whatever the chat provider
    embed_autoload: bool = True          # explicitly load the embedding model at startup if missing
    embed_gpu_offload: str = "off"       # lms --gpu: "off" (CPU), "max" (GPU) or 0-1
    embed_context: int = 2048            # film texts are short; larger contexts cost VRAM even on CPU
    embed_model: str = "text-embedding-bge-m3"
    embed_dim: int = 1024

    # owner role: embedding sync. reader role: every SQL statement the agent writes (SELECT only, read-only, 10 s)
    postgres_dsn: str = "postgresql://dvd:dvd@127.0.0.1:5434/dvdrental"
    postgres_reader_dsn: str = "postgresql://dvd_reader:dvd_reader@127.0.0.1:5434/dvdrental"
    mlflow_tracking_uri: str = "http://127.0.0.1:5001"
    mlflow_experiment: str = "dvd-agent"
    mlflow_ui_url: str = ""                  # MLflow address for trace links, if it differs from the tracking URI (Docker)
    api_url: str = "http://127.0.0.1:8001"   # used by the Streamlit UI
    mcp_token: str = ""                      # MCP over HTTP: if set, clients must send "Authorization: Bearer <token>"
    mcp_allowed_hosts: str = ""              # MCP over HTTP: extra Host header values, comma-separated (e.g. "myhost:8002")

    page_size: int = 50                  # rows per page; the answer step sees page 1, the user can ask for any page
    sql_retries: int = 2                 # regenerate the SQL with the error message this many times
    film_search_k: int = 10
    max_refinements: int = 1             # extra SQL round when the self-check says incomplete
    schema_cache_seconds: int = 300      # schema text is re-read from information_schema after this
    sync_interval_minutes: int = 10
    # The DSPy program with tuned prompts. Empty: the one in the package (agent/compiled_program.json,
    # few-shot SQL examples, see README "Optimization"). "none": the untuned prompts. Or a path.
    compiled_program_path: str = ""

    @property
    def max_tokens(self) -> int:
        if self.llm_provider == "lmstudio":
            return self.llm_max_tokens or 4000
        # cloud models think before answering and the thinking counts toward max_tokens: never below 16000
        return max(self.llm_max_tokens or 0, 16000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
