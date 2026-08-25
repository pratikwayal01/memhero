.PHONY: up down restart logs ps sync ui api cli mcp eval test clean kill

kill:          ## force-kill server on port 8765
	fuser -k 8765/tcp || true

up:            ## start db + langfuse
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f --tail=50

ps:
	docker compose ps

sync:
	uv sync

ui:            ## html ui on :8765
	uv run memhero-http

api:           ## fastapi on :8000
	uv run memhero-api

cli:           ## uv run make cli USER=you
	uv run memhero-cli $(USER)

mcp:
	uv run memhero-mcp

eval:          ## full ON vs OFF eval (~15 min on free tier)
	uv run python -m eval.run_eval

test:
	uv run pytest -q

clean:         ## WARNING: destroys all data volumes (memories + langfuse)
	docker compose down -v
