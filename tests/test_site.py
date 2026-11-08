"""Comparison site: registry, hardware cost, Hub weight format, engine and HTTP guards."""

import http.client
import importlib
import json
import math
import socket
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from bevotron.config import RunConfig, apply_overrides, load_raw
from bevotron.site import registry
from bevotron.site.costs import hardware_cost

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"

TINY = {
    "model.n_layer": 2,
    "model.d_model": 64,
    "model.n_head": 4,
    "model.n_kv_head": 4,
    "model.context_len": 32,
    "tokenizer.vocab_size": 257,
    "train.tokens": "32K",
    "train.batch_tokens": 128,
}


def tiny_config(**overrides) -> RunConfig:
    return RunConfig.from_dict(apply_overrides(load_raw(BASELINE), {**TINY, **overrides}))


def write_run(runs: Path, ablation: str, overrides: dict, config: RunConfig, ckpt: bool):
    folder = runs / config.run_id()
    folder.mkdir(parents=True)
    config.save(folder / "config.yaml")
    label = ",".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items()) or "baseline"
    meta = {
        "run_id": config.run_id(),
        "training_key": config.training_key(),
        "ablation": ablation,
        "label": label,
        "overrides": overrides,
    }
    (folder / "run.json").write_text(json.dumps(meta))
    (folder / "eval.json").write_text(json.dumps({"bpb": 0.5, "perplexity": 4.0, "extra": 1}))
    if ckpt:
        (folder / "ckpt.pt").write_bytes(b"")
    return config.run_id()


def test_registry_links_ptq_runs_to_baseline_weights_and_keeps_edits(tmp_path):
    base = tiny_config()
    ptq = tiny_config(**{"quant.scheme": "w4a8kv4"})
    vocab = tiny_config(**{"tokenizer.vocab_size": 300})
    b = write_run(tmp_path, "precision", {}, base, ckpt=True)
    p = write_run(tmp_path, "precision", {"quant.scheme": "w4a8kv4"}, ptq, ckpt=True)
    v = write_run(tmp_path, "vocab", {"tokenizer.vocab_size": 300}, vocab, ckpt=True)

    built = registry.build(tmp_path, repo="org/bevo")
    by_id = {m["id"]: m for m in built["models"]}
    assert by_id[p]["weights"] == b  # PTQ shares the baseline's training key
    assert by_id[p]["baseline"] == b and by_id[v]["baseline"] is None
    assert by_id[p]["name"] == "w4a8kv4" and by_id[v]["name"] == "0k vocab"
    assert by_id[b]["name"] == "precision baseline (int8, 0k vocab)"
    assert by_id[p]["hf"] == f"https://huggingface.co/org/bevo/tree/main/{p}"
    assert by_id[p]["results"] == {"bpb": 0.5, "perplexity": 4.0}
    assert not any(m["uploaded"] for m in built["models"])

    by_id[p]["name"], by_id[p]["uploaded"] = "hand-edited", True
    rebuilt = {m["id"]: m for m in registry.build(tmp_path, "org/bevo", built)["models"]}
    assert rebuilt[p]["name"] == "hand-edited" and rebuilt[p]["uploaded"]
    moved = {m["id"]: m for m in registry.build(tmp_path, "org/other", built)["models"]}
    assert not moved[p]["uploaded"]  # a new repo has not received the upload


def test_hardware_cost_counts_streamed_bytes_per_token():
    cfg = RunConfig.load(BASELINE)  # int8, 50k vocab, 8 x 512, 512-token context
    c = hardware_cost(cfg)
    assert c["weights_mb"] == 25_165_824 / 1e6
    assert c["kv_cache_mb"] == 2 * 8 * 16 * 32 * 512 / 1e6
    assert c["output_table_mb"] == 50_000 * 512 / 1e6
    assert math.isclose(c["stream_mb_chip_output"], c["stream_mb_host_output"] + 25.6)
    assert math.isclose(c["ms_per_token"], c["stream_mb_chip_output"] / 150 * 1e3)
    assert c["macs_per_token"] == 25_165_824 + 50_000 * 512 + 2 * 8 * 512 * 512
    w4 = hardware_cost(
        RunConfig.from_dict(apply_overrides(load_raw(BASELINE), {"quant.scheme": "w4a4kv4"}))
    )
    assert math.isclose(w4["stream_mb_chip_output"], c["stream_mb_chip_output"] / 2)
    assert (w4["weight_bits"], w4["act_bits"], w4["kv_bits"]) == (4, 4, 4)


# ---- engine: needs torch ----


@pytest.fixture
def tiny_hub(tmp_path, monkeypatch):
    """A tiny trained-looking run served through the Hub code path from tmp files."""
    torch = pytest.importorskip("torch")
    from bevotron.models import Bevo
    from bevotron.site import hub
    from bevotron.tokenizer.bpe import BPETokenizer

    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    tok_path = tmp_path / "tokenizer.json"
    BPETokenizer(vocab, []).save(tok_path)
    config = tiny_config()
    torch.manual_seed(0)
    model = Bevo(config)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "step": 7, "optimizer": {}}, ckpt)
    weights = tmp_path / "model.safetensors"
    hub.export_weights(ckpt, weights)
    entry = {
        "id": "tiny",
        "name": "tiny",
        "ablation": "test",
        "change": "baseline",
        "baseline": "tiny",
        "weights": "tiny",
        "hf_repo": "org/bevo",
        "hf_path": "tiny",
        "hf": registry.hf_url("org/bevo", "tiny"),
        "uploaded": True,
        "notes": "",
        "config": config.to_dict(),
        "results": {"bpb": 1.0},
    }
    monkeypatch.setattr(hub, "fetch", lambda e, owner: (tok_path, weights))
    return {"registry": {"hf_repo": "org/bevo", "models": [entry]}, "model": model, "tmp": tmp_path}


def test_exported_weights_drop_tied_head_and_reload_exactly(tiny_hub):
    import torch
    from safetensors.torch import load_file

    from bevotron.site.engine import Engine

    saved = load_file(str(tiny_hub["tmp"] / "model.safetensors"))
    assert "lm_head.weight" not in saved and "wte.weight" in saved
    engine = Engine(
        tiny_hub["registry"], device="cpu", source="hub", evals_path=tiny_hub["tmp"] / "e.json"
    )
    assert engine.availability(tiny_hub["registry"]["models"][0]) == "hub"
    loaded = engine.load("tiny", quant=False).model
    for name, tensor in tiny_hub["model"].state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    assert loaded.lm_head.weight is loaded.wte.weight  # re-tied after loading


def test_engine_generates_reproducibly_and_scores_a_quick_eval(tiny_hub, monkeypatch):
    import numpy as np

    prepare = importlib.import_module("bevotron.data.prepare")
    from bevotron.eval.quality import bits_per_byte
    from bevotron.site.engine import Engine

    evals = tiny_hub["tmp"] / "evals.json"
    engine = Engine(tiny_hub["registry"], device="cpu", source="hub", evals_path=evals)
    a = engine.generate("tiny", "Once", quant=True, seed=3, max_new_tokens=12)
    b = engine.generate("tiny", "Once", quant=True, seed=3, max_new_tokens=12)
    assert a["text"] == b["text"] and a["text"].startswith("Once")
    assert a["scheme"] == "int8" and a["prompt_tokens"] == 4 and a["new_tokens"] <= 12
    assert engine.generate("tiny", "Once", quant=False, max_new_tokens=4)["scheme"] == "fp32"

    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=40 * 32 + 1).astype(np.uint16)
    monkeypatch.setattr(prepare, "val_split", lambda config: (data, 1000, 4000))
    seen = []
    result = engine.evaluate("tiny", "quick", progress=lambda f, msg: seen.append(f))
    assert result["windows"] == 2 and result["tokens"] == 64  # 5% of 40 windows
    assert math.isclose(result["bpb"], bits_per_byte(result["val_loss"], 1000, 4000))
    assert result["scheme"] == "int8" and seen[-1] == 1.0
    assert json.loads(evals.read_text())["tiny"]["quick"]["bpb"] == result["bpb"]
    full = engine.evaluate("tiny", "full")
    assert full["windows"] == 40


# ---- HTTP guards ----


class FakeEngine:
    def __init__(self, reg):
        self.registry, self.models = reg, {m["id"]: m for m in reg["models"]}
        self.device, self.source = "cpu", "auto"

    def availability(self, entry):
        return "missing"

    def local_evals(self):
        return {}


@pytest.fixture
def server():
    from bevotron.site.engine import Jobs
    from bevotron.site.server import make_handler

    cfg = tiny_config()
    entry = {
        "id": "tiny",
        "name": "tiny",
        "ablation": "test",
        "baseline": "tiny",
        "weights": "tiny",
        "hf_repo": "org/bevo",
        "hf_path": "tiny",
        "uploaded": False,
        "config": cfg.to_dict(),
        "results": {},
    }
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = ThreadingHTTPServer(
        ("127.0.0.1", port),
        make_handler(FakeEngine({"hf_repo": "org/bevo", "models": [entry]}), Jobs(), port),
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield port
    srv.shutdown()
    srv.server_close()


def request(port, method, path, body=None, host=None, content_type="application/json"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": host or f"localhost:{port}"}
    if body is not None:
        headers["Content-Type"] = content_type
    conn.request(method, path, body=body, headers=headers)
    r = conn.getresponse()
    return r.status, json.loads(r.read() or b"null") if "json" in r.getheader(
        "Content-Type", ""
    ) else None


def test_server_serves_state_and_page_to_localhost(server):
    status, state = request(server, "GET", "/api/state")
    assert status == 200 and state["models"][0]["id"] == "tiny"
    assert state["models"][0]["cost"]["weight_bits"] == 8 and "config" not in state["models"][0]
    assert request(server, "GET", "/")[0] == 200


def test_server_rejects_other_hosts_non_json_and_bad_payloads(server):
    assert request(server, "GET", "/api/state", host="evil.example")[0] == 403
    assert request(server, "POST", "/api/eval", "{}", host=f"attacker.test:{server}")[0] == 403
    assert request(server, "POST", "/api/eval", "ids=tiny", content_type="text/plain")[0] == 415
    status, body = request(server, "POST", "/api/generate", json.dumps({"slots": []}))
    assert status == 400 and "pick" in body["error"]
    status, body = request(server, "POST", "/api/eval", json.dumps({"ids": ["nope"]}))
    assert status == 400 and "nope" in body["error"]
    assert (
        request(server, "POST", "/api/eval", json.dumps({"ids": ["tiny"], "scope": "x"}))[0] == 400
    )
    assert request(server, "GET", "/api/jobs/999")[0] == 404
