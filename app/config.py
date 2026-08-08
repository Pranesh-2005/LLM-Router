import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Put .env into os.environ so LiteLLM can see the provider keys.

    pydantic-settings only maps the fields declared below; provider keys like
    GROQ_API_KEY are read by LiteLLM straight from the environment, so they'd
    be invisible without this. Real env vars always win over the file.
    ponytail: 6 lines instead of a python-dotenv dependency.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

# NVIDIA hands out one key but LiteLLM looks for NVIDIA_NIM_API_KEY. Alias it so
# the .env only ever holds the secret once.
if os.environ.get("NVIDIA_API_KEY"):
    os.environ.setdefault("NVIDIA_NIM_API_KEY", os.environ["NVIDIA_API_KEY"])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- candidate pool -----------------------------------------------------
    # Adding a provider = add its model id here + set its API key env var.
    # LiteLLM resolves provider/auth from the id prefix, so no code change.
    router_models: str = (
        "gpt-5.4-nano,gpt-5.4-mini,gpt-5.5,"
        "claude-haiku-4-5,claude-sonnet-5,"
        "gemini/gemini-3.5-flash,"
        "groq/llama-3.3-70b-versatile"
    )
    # Small, cheap model used by the guardrail/intent agent.
    classifier_model: str = "gpt-5.4-nano"
    # Fallback chain used when the picked model errors out.
    fallback_models: str = "gpt-5.4-mini,gemini/gemini-3.5-flash"

    # Per-model base URLs for OpenAI-compatible endpoints (vLLM, LM Studio,
    # LiteLLM proxy, any self-hosted /v1). Format: "model=url,model=url".
    # Providers with a known prefix (groq/, deepseek/, together_ai/, ollama/,
    # openrouter/) need nothing here -- LiteLLM already knows their endpoint.
    model_api_bases: str = ""

    default_priority: str = "balanced"  # cheap | balanced | quality

    # --- reliability --------------------------------------------------------
    num_retries: int = 3          # litellm does exponential backoff between these
    request_timeout: float = 60.0
    classifier_timeout: float = 6.0

    # --- caching ------------------------------------------------------------
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.93
    semantic_cache_ttl: int = 900
    semantic_cache_size: int = 2000
    idempotency_ttl: int = 86400
    intent_cache_size: int = 4096

    # --- rate limiting (token bucket, per API key / client ip) --------------
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 20

    # --- misc ---------------------------------------------------------------
    api_keys: str = ""            # comma list; empty => auth disabled
    max_query_chars: int = 32000
    log_level: str = "INFO"

    @property
    def model_pool(self) -> list[str]:
        return [m.strip() for m in self.router_models.split(",") if m.strip()]

    @property
    def fallbacks(self) -> list[str]:
        return [m.strip() for m in self.fallback_models.split(",") if m.strip()]

    @property
    def api_bases(self) -> dict[str, str]:
        out = {}
        for pair in self.model_api_bases.split(","):
            model, _, url = pair.partition("=")
            if model.strip() and url.strip():
                out[model.strip()] = url.strip()
        return out

    @property
    def allowed_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()
