"""Stdlib HTTP server + HTML UI for memhero. No deps beyond stdlib + memhero.

Run: uv run memhero-http  →  http://localhost:8765
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .service import ChatService, store

HOST = "0.0.0.0"
PORT = 8765
MAX_BODY = 1_000_000

INDEX_HTML = (Path(__file__).parent / "templates" / "index.html").read_text()
svc = ChatService()

# Verify DB connectivity at startup
try:
    store()
except Exception as e:
    import sys
    print(f"Database not reachable: {e}", file=sys.stderr)
    print("Run: docker compose up -d", file=sys.stderr)
    sys.exit(1)


def _drain(user_id: str, conversation: str, message: str, reply: str):
    try:
        svc.learn(user_id, message, reply)
        # also drain any pending extractions
        svc.drain_pending(user_id)
    except Exception as e:
        print(f"[drain] {type(e).__name__}: {e}")


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            return None

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/" or url.path == "/index.html":
            html = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        if url.path == "/api/memories":
            uid = parse_qs(url.query).get("user_id", ["demo"])[0]
            mems = [
                {"id": str(m.id), "content": m.content, "updated": str(m.updated_at)}
                for m in store().list_memories(uid)
            ]
            self._send({"memories": mems, "count": len(mems)})
            return
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        body = self._read_body()
        if body is None:
            return self._send({"error": "invalid or missing body"}, 400)

        if url.path == "/api/chat":
            uid = body.get("user_id", "demo")
            msg = body.get("message", "")
            with_mem = body.get("with_memory", True)
            t0 = time.perf_counter()
            out = svc.turn(uid, msg, use_memory=with_mem, conversation=uid, background=False)
            threading.Thread(target=_drain, args=(uid, uid, msg, out["reply"]), daemon=True).start()
            wall_ms = round((time.perf_counter() - t0) * 1000)
            mems = [
                {"id": str(m.id), "content": m.content, "updated": str(m.updated_at)}
                for m in store().list_memories(uid)
            ]
            self._send({
                "answer": out["reply"],
                "retrieved": out["retrieved"],
                "store": mems,
                "times": {"inline_ms": out["inline_ms"], "memory_ms": out["memory_ms"], "wall_ms": wall_ms},
            })
            return

        if url.path == "/api/forget":
            uid = body.get("user_id", "demo")
            mid = body.get("mem_id")
            if mid:
                s = store()
                old_mem = next((m for m in s.list_memories(uid) if str(m.id) == str(mid)), None)
                if old_mem:
                    s.delete(mid, uid, "FORGET", {"direct": True})
            else:
                query = body.get("query", "")
                svc.forget(uid, query) if query.strip() else []
            mems = [
                {"id": str(m.id), "content": m.content, "updated": str(m.updated_at)}
                for m in store().list_memories(uid)
            ]
            self._send({"deleted": 1 if mid else 0, "memories": mems, "count": len(mems)})
            return

        if url.path == "/api/reset":
            uid = body.get("user_id", "demo")
            store().clear_user(uid)
            self._send({"ok": True, "count": 0})
            return

        self._send({"error": "not found"}, 404)

    def log_message(self, *a):
        pass


def main():
    import sys
    print(f"memhero UI → http://localhost:{PORT}")
    try:
        ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"Port {PORT} already in use. Kill it: fuser -k {PORT}/tcp", file=sys.stderr)
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()