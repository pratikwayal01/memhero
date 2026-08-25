import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


@dataclass(frozen=True)
class Config:
    api_key: str = field(default_factory=lambda: os.environ["OPENAI_API_KEY"])
    base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    chat_model: str = field(default_factory=lambda: os.environ.get("MEMHERO_CHAT_MODEL", "gpt-4o-mini"))
    aux_model: str = field(default_factory=lambda: os.environ.get("MEMHERO_AUX_MODEL") or os.environ.get("MEMHERO_CHAT_MODEL", "gpt-4o-mini"))
    embed_model: str = field(default_factory=lambda: os.environ.get("MEMHERO_EMBED_MODEL", "text-embedding-3-small"))
    embed_api_key: str | None = field(default_factory=lambda: os.environ.get("MEMHERO_EMBED_API_KEY"))
    embed_base_url: str | None = field(default_factory=lambda: os.environ.get("MEMHERO_EMBED_BASE_URL"))
    embed_provider: str = field(default_factory=lambda: os.environ.get("MEMHERO_EMBED_PROVIDER", "api"))  # "api" or "local"
    local_model: str = field(default_factory=lambda: os.environ.get("MEMHERO_LOCAL_MODEL", "all-MiniLM-L6-v2"))
    vector_dims: int = field(default_factory=lambda: int(os.environ.get("MEMHERO_VECTOR_DIMS", "0")))  # 0 = auto
    database_url: str = field(default_factory=lambda: os.environ.get("MEMHERO_DATABASE_URL", "postgresql://memhero:memhero@localhost:5434/memhero"))

    langfuse_host: str = field(default_factory=lambda: os.environ.get("LANGFUSE_HOST", "http://localhost:3000"))
    langfuse_public_key: str | None = field(default_factory=lambda: os.environ.get("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str | None = field(default_factory=lambda: os.environ.get("LANGFUSE_SECRET_KEY"))

    # memory tuning
    retrieve_k: int = 8
    retrieve_min_sim: float = 0.25
    dup_sim: float = 0.65          # candidate vs existing -> treat as same topic for reconcile
    cap_per_user: int = 500
    skip_retrieval_under_words: int = field(default_factory=lambda: int(os.environ.get("MEMHERO_SKIP_RETRIEVAL_UNDER_WORDS", "0")))


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
