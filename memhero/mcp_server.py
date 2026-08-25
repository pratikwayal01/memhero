"""MCP server: expose the memory store as tools for any MCP-capable LLM client.

Run: uv run memhero-mcp   (stdio transport)
"""

import os

from mcp.server import MCPServer

from .service import ChatService, store


def _uid(user_id: str | None) -> str:
    return user_id or os.environ.get("MEMHERO_MCP_USER", "mcp-user")


mcp = MCPServer(
    name="memhero",
    instructions=(
        "Persistent long-term memory for the user across conversations. "
        "Call remember() when the user shares durable facts. Call recall() when "
        "context about the user would help answer. Secrets are never stored."
    ),
)
svc = ChatService()


@mcp.tool()
def remember(facts: list[str], user_id: str | None = None) -> dict:
    """Store durable facts about the user (third-person, e.g. "The user lives in Pune").
    Duplicates are merged, outdated facts superseded automatically. Secrets rejected."""
    return svc.remember_many(_uid(user_id), facts)


@mcp.tool()
def recall(query: str, k: int = 8, user_id: str | None = None) -> list[str]:
    """Retrieve stored memories relevant to a query. Empty list if nothing relevant."""
    from . import llm
    from .config import get_config
    cfg = get_config()
    vec = llm.embed([query])[0]
    return [m.content for m in store().search(_uid(user_id), vec, k, cfg.retrieve_min_sim)]


@mcp.tool()
def forget(query: str, user_id: str | None = None) -> list[str]:
    """Delete memories matching a natural-language request ("forget my old address").
    Returns ids of deleted memories."""
    return svc.forget(_uid(user_id), query)


@mcp.tool()
def memories(user_id: str | None = None) -> list[dict]:
    """List all stored memories for the user."""
    return [{"id": m.id, "content": m.content, "updated_at": m.updated_at}
            for m in store().list_memories(_uid(user_id))]


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
