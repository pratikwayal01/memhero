"""LLM calls: chat, embeddings (API or local with LRU cache), fact extraction, reconcile. All traced via langfuse.openai."""

import functools
import json
import re
import time

import openai
import langfuse.openai  # noqa: F401  # patches openai globally with tracing
from openai import OpenAI

from .config import get_config

_client: OpenAI | None = None
_aux_client: OpenAI | None = None
_embed_client: OpenAI | None = None
_local_model: object | None = None  # sentence_transformers.SentenceTransformer


def client() -> OpenAI:
    global _client
    if _client is None:
        cfg = get_config()
        _client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=90.0)
    return _client


def aux_client() -> OpenAI:
    """Separate client for extraction/reconcile/judging — hits a different rpm quota bucket."""
    global _aux_client
    if _aux_client is None:
        cfg = get_config()
        _aux_client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=90.0)
    return _aux_client


def embed_client() -> OpenAI:
    """Dedicated client for embeddings — can point to a different provider (e.g. Gemini embeds + Groq chat)."""
    global _embed_client
    if _embed_client is None:
        cfg = get_config()
        _embed_client = OpenAI(
            api_key=cfg.embed_api_key or cfg.api_key,
            base_url=cfg.embed_base_url or cfg.base_url,
            timeout=90.0,
        )
    return _embed_client


def _retry(fn, *args, **kwargs):
    """Backoff on 429 and stalled/dropped connections."""
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError:
            wait = min(30 * (attempt + 1), 60)
            print(f"[memhero] rate limited, waiting {wait}s", flush=True)
            time.sleep(wait)
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            wait = 5
            print(f"[memhero] {type(e).__name__}, retrying in {wait}s", flush=True)
            time.sleep(wait)
    return fn(*args, **kwargs)


def _get_local_model():
    """Lazy-load sentence-transformers model once."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        cfg = get_config()
        _local_model = SentenceTransformer(cfg.local_model)
    return _local_model


def embed_dim() -> int:
    """Return embedding vector dimension for current provider (0 until first embed call)."""
    cfg = get_config()
    if cfg.vector_dims > 0:
        return cfg.vector_dims
    # auto-detect by running a single embed
    v = embed(["dim test"])[0]
    return len(v)


@functools.lru_cache(maxsize=4096)  # same text → same vector, zero recompute
def _embed_cached(text: str, provider: str) -> tuple[float, ...] | None:
    """Cached wrapper — provider param disambiguates local vs api vectors."""
    cfg = get_config()
    if provider == "local":
        model = _get_local_model()
        return tuple(float(x) for x in model.encode(text))
    resp = _retry(embed_client().embeddings.create, model=cfg.embed_model, input=[text])
    return tuple(float(x) for x in resp.data[0].embedding)


def embed(texts: list[str]) -> list[list[float]]:
    cfg = get_config()
    provider = cfg.embed_provider
    results = []
    for t in texts:
        v = _embed_cached(t, provider)
        if v is not None:
            results.append(list(v))
        else:
            # cache miss — shouldn't happen with current impl, but safe fallback
            if provider == "local":
                results.append(_get_local_model().encode(t).tolist())
            else:
                resp = _retry(embed_client().embeddings.create, model=cfg.embed_model, input=[t])
                results.append(resp.data[0].embedding)
    return results


def chat(messages: list[dict], temperature: float = 0.7) -> str:
    return chat_with(get_config().chat_model, messages, temperature)


def aux_chat(messages: list[dict], temperature: float = 0) -> str:
    """Auxiliary LLM call (extraction, reconcile, judging) on the aux-model quota bucket."""
    out = chat_with(get_config().aux_model, messages, temperature, c=aux_client())
    # thinking models emit <thought>/<think> blocks before the answer
    return re.sub(r"<(?:thought|think)>.*?(?:</(?:thought|think)>|\Z)", "", out, flags=re.DOTALL).strip()


def chat_with(model: str, messages: list[dict], temperature: float = 0.7, c: OpenAI | None = None) -> str:
    c = c or client()
    resp = _retry(
        c.chat.completions.create,
        model=model, messages=messages, temperature=temperature,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


# -- extraction ---------------------------------------------------------------

EXTRACT_SYSTEM = """You extract durable facts about the user from a conversation exchange.

Rules:
- Only extract stable, reusable facts about the user (identity, preferences, relationships, work, plans, possessions, skills).
- Preserve temporal qualifiers VERBATIM ("moved to Bangalore in July 2026", "lived in Delhi until March").
- Do not extract: transient state ("I'm tired"), assistant statements, opinions about the assistant.
- Never extract secrets: passwords, API keys, tokens, card numbers, government IDs. Skip them entirely.
- Assign a short lowercase slot name if the fact belongs to a known category (location, job, diet, relationship, pet, allergy, name, age, hobby). Use null for uncategorized facts.
- Return a JSON array of objects: [{"content": "...", "slot": "location"|null}, ...].
- Empty array [] if nothing worth remembering."""


def extract_facts(user_message: str, assistant_reply: str) -> list[dict]:
    out = aux_chat(
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"USER: {user_message}\nASSISTANT: {assistant_reply}"},
        ],
        temperature=0,
    )
    try:
        facts = json.loads(out)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", out, re.DOTALL)
        facts = json.loads(m.group()) if m else []
    result = []
    for f in facts:
        if isinstance(f, str):
            # backward compat: bare string → fact with no slot
            f = {"content": f, "slot": None}
        if isinstance(f, dict) and f.get("content") and guard_secret(str(f["content"])):
            slot = str(f.get("slot") or "").strip().lower() or None
            # ponytail: validate slot format — alphanumeric + hyphens, 1-40 chars
            if slot and not re.match(r"^[a-z0-9_-]{1,40}$", slot):
                slot = None
            result.append({"content": f["content"].strip(), "slot": slot})
    return result


_SECRET_PATTERNS = [
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),                 # card numbers
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),                    # long hex
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),               # api keys
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                     # aws keys
    re.compile(r"\b[A-Za-z0-9+/=]{40,}\b"),                  # base64 blobs
]


def guard_secret(text: str) -> bool:
    """True if safe to store."""
    return not any(p.search(text) for p in _SECRET_PATTERNS)


# -- reconcile ----------------------------------------------------------------

RECONCILE_SYSTEM = """You maintain a user's memory store of discrete facts.

You get NEW candidate facts and the user's EXISTING similar memories (id + content).
Decide per candidate what to do and return ONLY a JSON array of operations:

[{"op": "ADD", "content": "..."}]                                  - genuinely new fact
[{"op": "UPDATE", "id": "<existing-id>", "content": "..."}]        - refines/extends one existing memory in place
[{"op": "SUPERSEDE", "id": "<existing-id>", "content": "..."}]     - replaces an outdated/conflicting memory (e.g. moved cities, changed jobs)
[{"op": "DELETE", "id": "<existing-id>", "reason": "..."}]         - existing memory is now known false or the candidate says to forget it
[{"op": "SKIP", "reason": "..."}]                                  - duplicate / trivial / already covered

Rules:
- UPDATE when both can merge into one richer fact; SUPERSEDE only when they conflict and the candidate is newer information.
- Facts that merely coexist (works in X, lives in Y) are NOT conflicts - keep both.
- Preserve temporal qualifiers verbatim in output content.
- One operation per candidate, same order as candidates."""


def reconcile(candidates: list[str], matches: list[list]) -> list[dict]:
    """candidates: new facts; matches: parallel list of [(id, content), ...] nearest existing memories."""
    lines = []
    for i, cand in enumerate(candidates):
        existing = "\n".join(f"  [{mid}] {content}" for mid, content in matches[i]) or "  (none)"
        lines.append(f"CANDIDATE {i}: {cand}\nEXISTING:\n{existing}")
    out = aux_chat(
        [
            {"role": "system", "content": RECONCILE_SYSTEM},
            {"role": "user", "content": "\n\n".join(lines)},
        ],
        temperature=0,
    )
    try:
        ops = json.loads(out)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", out, re.DOTALL)
        ops = json.loads(m.group()) if m else []
    valid = {"ADD", "UPDATE", "SUPERSEDE", "DELETE", "SKIP"}
    return [o for o in ops if isinstance(o, dict) and o.get("op") in valid]


# -- slot detection for retrieval ---------------------------------------------
# Query keywords → slot name. Small; grows with eval needs.
SLOT_ALIASES = {
    "location": ["live", "living", "stay", "stays", "staying", "city", "town", "where do i", "address", "moved"],
    "job": ["job", "work", "working", "company", "employer", "career", "office", "profession"],
    "diet": ["diet", "eat", "eating", "food", "meal", "vegetarian", "vegan", "allergy", "allergic",
             "dinner", "lunch", "breakfast", "snack", "recipe", "restaurant", "cook", "dish", "cuisine"],
    "relationship": ["relationship", "married", "wife", "husband", "girlfriend", "boyfriend", "partner", "spouse"],
    "pet": ["pet", "dog", "cat", "animal"],
    "name": ["name", "called", "my name is", "i'm called"],
    "age": ["age", "old", "years old", "born", "birthday"],
    "hobby": ["hobby", "hobbies", "like to", "enjoy", "interest", "play", "sport", "game"],
}


def detect_slots(query: str) -> list[str]:
    """Return slot names triggered by query keywords. Cheap, no embedding."""
    q = query.lower()
    return [slot for slot, aliases in SLOT_ALIASES.items() if any(a in q for a in aliases)]
