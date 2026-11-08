"""`bevotron site`: a local web page over configs/models.json.

Binds 127.0.0.1 only, answers only requests addressed to localhost, and takes JSON POSTs
only, so other sites open in the browser cannot drive it. Generation and evals run as
jobs on one worker thread; the page polls them.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from bevotron.config import RunConfig
from bevotron.eval.latency import DEFAULT_LINK_MB_PER_S
from bevotron.site.costs import hardware_cost
from bevotron.site.engine import QUICK_FRACTION, Engine, Jobs

STATIC = Path(__file__).parent / "static"
MAX_NEW_TOKENS = 512
MAX_SLOTS = 6


def state(engine: Engine) -> dict:
    """Everything the page draws: registry entries plus cost, availability, local evals."""
    evals = engine.local_evals()
    models = []
    for entry in engine.registry["models"]:
        config = RunConfig.from_dict(entry["config"])
        m = config.model
        models.append(
            {
                **{k: v for k, v in entry.items() if k != "config"},
                "scheme": config.quant.scheme,
                "mode": config.quant.mode,
                "vocab_size": config.tokenizer.vocab_size,
                "shape": f"{m.n_layer}L {m.d_model}d {m.n_head}H, ctx {m.context_len}",
                "batch_tokens": config.train.batch_tokens,
                "train_tokens": config.train.total_tokens,
                "cost": hardware_cost(config),
                "available": engine.availability(entry),
                "local_evals": evals.get(entry["id"], {}),
            }
        )
    return {
        "models": models,
        "hf_repo": engine.registry.get("hf_repo"),
        "device": engine.device,
        "source": engine.source,
        "link_mb_per_s": DEFAULT_LINK_MB_PER_S,
        "quick_fraction": QUICK_FRACTION,
    }


def submit_generate(engine: Engine, jobs: Jobs, body: dict) -> dict:
    slots = body.get("slots") or []
    if not 1 <= len(slots) <= MAX_SLOTS:
        raise ValueError(f"pick 1 to {MAX_SLOTS} models")
    for slot in slots:
        if slot.get("id") not in engine.models:
            raise ValueError(f"unknown model {slot.get('id')!r}")
    params = {
        "prompt": str(body.get("prompt", ""))[:2000],
        "seed": int(body.get("seed", 0)),
        "temperature": min(max(float(body.get("temperature", 0.7)), 0.0), 2.0),
        "top_k": max(int(body.get("top_k", 50)), 1),
        "max_new_tokens": min(max(int(body.get("max_new_tokens", 150)), 1), MAX_NEW_TOKENS),
    }

    def run(job: dict) -> list:
        for i, slot in enumerate(slots):
            name = engine.models[slot["id"]]["name"]
            job["progress"], job["message"] = i / len(slots), f"generating: {name}"
            try:
                out = engine.generate(slot["id"], quant=bool(slot.get("quant", True)), **params)
            except Exception as e:  # one missing model should not hide the others
                out = {"id": slot["id"], "error": f"{type(e).__name__}: {e}"}
            out["slot"] = i
            job["partial"].append(out)
        return job["partial"]

    label = " vs ".join(engine.models[s["id"]]["name"] for s in slots)
    return jobs.submit("generate", label, run)


def submit_evals(engine: Engine, jobs: Jobs, body: dict) -> list[dict]:
    ids = body.get("ids") or []
    scope = body.get("scope", "quick")
    if scope not in ("quick", "full"):
        raise ValueError("scope must be quick or full")
    unknown = [i for i in ids if i not in engine.models]
    if not ids or unknown:
        raise ValueError(f"pick models to evaluate (unknown: {unknown})")
    submitted = []
    for model_id in ids:

        def run(job: dict, model_id: str = model_id) -> dict:
            def progress(fraction: float, message: str) -> None:
                job["progress"], job["message"] = fraction, message

            return engine.evaluate(model_id, scope, progress=progress)

        label = f"{scope} eval: {engine.models[model_id]['name']}"
        submitted.append(jobs.submit("eval", label, run))
    return submitted


def make_handler(engine: Engine, jobs: Jobs, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "bevotron-site"

        def log_message(self, format, *args):  # keep the terminal for errors only
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _local(self) -> bool:
            if self.headers.get("Host", "") in allowed_hosts:
                return True
            self._json(403, {"error": "this site only answers requests to localhost"})
            return False

        def do_GET(self) -> None:
            if not self._local():
                return
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(200, state(engine))
            elif path == "/api/jobs":
                self._json(200, jobs.recent())
            elif path.startswith("/api/jobs/"):
                job = jobs.get(path.rsplit("/", 1)[1])
                self._json(200, job) if job else self._json(404, {"error": "no such job"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._local():
                return
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": "send JSON"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/generate":
                    self._json(200, submit_generate(engine, jobs, body))
                elif path == "/api/eval":
                    self._json(200, submit_evals(engine, jobs, body))
                else:
                    self._json(404, {"error": "not found"})
            except (ValueError, TypeError, KeyError) as e:
                self._json(400, {"error": str(e)})

    return Handler


def serve(engine: Engine, port: int = 8642) -> None:
    jobs = Jobs()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(engine, jobs, port))
    available = sum(engine.availability(m) != "missing" for m in engine.registry["models"])
    print(
        f"bevotron site on http://localhost:{port}  ({len(engine.models)} models, "
        f"{available} runnable here, device {engine.device}, source {engine.source})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
