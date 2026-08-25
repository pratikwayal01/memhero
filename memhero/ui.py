"""Gradio UI: chat + live memory sidebar. Same ChatService."""

import gradio as gr

from .service import ChatService, store

svc = ChatService()


def _memories_table(user_id: str):
    rows = [(m.updated_at, m.content) for m in store().list_memories(user_id)]
    return gr.Dataframe(value=rows or [], wrap=True), f"{len(rows)} memories"


def _chat(user_id, message, history):
    if not message.strip():
        return history, "", ""
    out = svc.turn(user_id, message, background=False)
    svc.learn(user_id, message, out["reply"])  # inline (dedup-safe since turn also background=False)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": out["reply"]},
    ]
    recalled = "\n".join(f"• {r}" for r in out["retrieved"])
    table, count = _memories_table(user_id)
    return history, table, f"recalled:\n{recalled or '—'}\n\nreply latency: {out['inline_ms']} ms"


def _forget(user_id, query):
    if query.strip():
        svc.forget(user_id, query)
    table, count = _memories_table(user_id)
    return table, count


def _clear_user(user_id):
    store().clear_user(user_id)
    return *_memories_table(user_id), []


def main():
    with gr.Blocks(title="memhero") as demo:
        with gr.Row():
            user_id = gr.Textbox(label="user id", value="demo-user", scale=3)
            count = gr.Markdown("0 memories", scale=1)
        with gr.Row():
            with gr.Column(scale=2):
                chat = gr.Chatbot(height=480, label="chat")
                msg = gr.Textbox(placeholder="say something about yourself…", autofocus=True)
                state = gr.State([])
                clear_btn = gr.Button("clear conversation")
            with gr.Column(scale=1):
                recall = gr.Markdown("recalled:\n—")
                mem_table = gr.Dataframe(headers=["updated", "memory"], wrap=True, label="stored memories")
                forget_q = gr.Textbox(placeholder="forget my old address…")
                forget_btn = gr.Button("forget")

        msg.submit(_chat, [user_id, msg, state], [chat, mem_table, recall]).then(
            lambda: "", None, msg
        )
        clear_btn.click(_clear_user, user_id, [mem_table, count, chat])
        forget_btn.click(_forget, [user_id, forget_q], [mem_table, count])
        demo.load(_memories_table, user_id, [mem_table, count])

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
