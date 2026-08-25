"""HTTP API: POST /chat, memory management endpoints."""

from collections import defaultdict

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from .config import get_config
from .service import ChatService, store

app = FastAPI(title="memhero")
svc = ChatService()


class ChatIn(BaseModel):
    user_id: str
    message: str
    use_memory: bool = True


@app.post("/chat")
def chat(body: ChatIn, background: BackgroundTasks):
    result = svc.turn(body.user_id, body.message, use_memory=body.use_memory)
    # extraction already ran in its own thread via turn(); nothing to schedule here.
    return result


@app.get("/users/{user_id}/memories")
def list_memories(user_id: str):
    return [{"id": m.id, "content": m.content, "updated_at": m.updated_at}
            for m in store().list_memories(user_id)]


class ForgetIn(BaseModel):
    user_id: str
    query: str


@app.post("/forget")
def forget(body: ForgetIn):
    deleted = svc.forget(body.user_id, body.query)
    return {"deleted": deleted}


@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: str, user_id: str):
    store().delete(memory_id, user_id, "DELETE", {"via": "api"})
    return {"ok": True}


@app.get("/healthz")
def healthz():
    cfg = get_config()
    n = store().count("%")  # global count
    return {"ok": True, "memories_total": n, "chat_model": cfg.chat_model}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
