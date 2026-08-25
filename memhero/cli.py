"""CLI REPL over the same ChatService. Extraction runs after reply (hidden by user think time)."""

import sys

from .service import ChatService, store


def main():
    user_id = sys.argv[1] if len(sys.argv) > 1 else "cli-user"
    svc = ChatService()
    print(f"memhero cli | user: {user_id} | /memories /forget <text> /quit")
    while True:
        try:
            msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/memories":
            for m in store().list_memories(user_id):
                print(f"  [{m.updated_at}] {m.content}")
            print(f"  total: {store().count(user_id)}")
            continue
        if msg.startswith("/forget "):
            deleted = svc.forget(user_id, msg[len("/forget "):])
            print(f"  forgot {len(deleted)} memories")
            continue
        out = svc.turn(user_id, msg)
        svc.learn(user_id, msg, out["reply"])  # inline here; think-time hides it in real use
        print(out["reply"])
        if out["retrieved"]:
            print(f"  [recalled {len(out['retrieved'])}]")


if __name__ == "__main__":
    main()
