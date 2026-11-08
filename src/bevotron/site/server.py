"""`bevotron site`: a local page over configs/models.json.

Read-only: it serves the page and the registry's numbers, with hardware cost worked out
from each run's config. Nothing runs a model. Binds 127.0.0.1 and answers only requests
addressed to localhost.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from bevotron.config import RunConfig
from bevotron.eval.latency import DEFAULT_LINK_MB_PER_S
from bevotron.site.costs import hardware_cost

STATIC = Path(__file__).parent / "static"


def state(registry: dict) -> dict:
    """Everything the page draws: registry entries plus shape and hardware cost."""
    models = []
    for entry in registry["models"]:
        config = RunConfig.from_dict(entry["config"])
        m = config.model
        models.append(
            {
                **{k: v for k, v in entry.items() if k != "config"},
                "scheme": config.quant.scheme,
                "mode": config.quant.mode,
                "vocab_size": config.tokenizer.vocab_size,
                "shape": f"{m.n_layer}L {m.d_model}d {m.n_head}H, ctx {m.context_len}",
                "train_tokens": config.train.total_tokens,
                "cost": hardware_cost(config),
            }
        )
    return {
        "models": models,
        "hf_repo": registry.get("hf_repo"),
        "samples_note": registry.get("samples_note", ""),
        "link_mb_per_s": DEFAULT_LINK_MB_PER_S,
    }


def make_handler(registry: dict, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    body = json.dumps(state(registry)).encode()

    class Handler(BaseHTTPRequestHandler):
        server_version = "bevotron-site"

        def log_message(self, format, *args):  # keep the terminal for errors only
            pass

        def _send(self, code: int, data: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            if self.headers.get("Host", "") not in allowed_hosts:
                self._json(403, {"error": "this site only answers requests to localhost"})
                return
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(200, body, "application/json")
            else:
                self._json(404, {"error": "not found"})

    return Handler


def serve(registry: dict, port: int = 8642) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(registry, port))
    print(f"bevotron site on http://localhost:{port}  ({len(registry['models'])} models)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
