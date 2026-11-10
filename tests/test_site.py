"""Comparison site: registry, saved samples, hardware cost, Hub weight format, HTTP guards."""

import http.client
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


def write_report(path: Path, base: str, ptq: str) -> Path:
    path.write_text(
        '# Report\n\n## Samples\n\nPrompt "Once",\nseed 0.\n\n### Precision\n\n'
        f"#### bevotron sample {base}\n\n```text\nOnce fp.\n\nSecond paragraph.\n```\n\n"
        f"#### bevotron sample {base} --quant\n\n```text\nOnce int8.\n```\n\n"
        f"#### bevotron sample {ptq} --quant\n\n```text\nOnce w4.\n```\n\n## After\n"
    )
    return path


def test_registry_links_ptq_runs_to_baseline_weights_and_keeps_edits(tmp_path):
    runs = tmp_path / "runs"
    base = tiny_config()
    ptq = tiny_config(**{"quant.scheme": "w4a8kv4"})
    vocab = tiny_config(**{"tokenizer.vocab_size": 300})
    b = write_run(runs, "precision", {}, base, ckpt=True)
    p = write_run(runs, "precision", {"quant.scheme": "w4a8kv4"}, ptq, ckpt=True)
    v = write_run(runs, "vocab", {"tokenizer.vocab_size": 300}, vocab, ckpt=True)
    report = write_report(tmp_path / "report.md", b, p)

    built = registry.build(runs, repo="org/bevo", report=report)
    by_id = {m["id"]: m for m in built["models"]}
    assert by_id[p]["weights"] == b  # PTQ shares the baseline's training key
    assert by_id[p]["baseline"] == b and by_id[v]["baseline"] is None
    assert by_id[p]["name"] == "w4a8kv4" and by_id[v]["name"] == "0k vocab"
    assert by_id[b]["name"] == "precision baseline (int8, 0k vocab)"
    assert by_id[p]["hf"] == f"https://huggingface.co/org/bevo/tree/main/{p}"
    assert by_id[p]["results"] == {"bpb": 0.5, "perplexity": 4.0}
    assert not any(m["uploaded"] for m in built["models"])

    assert built["samples_note"] == 'Prompt "Once", seed 0.'
    assert by_id[b]["samples"] == [
        {"quant": False, "text": "Once fp.\n\nSecond paragraph."},
        {"quant": True, "text": "Once int8."},
    ]
    assert by_id[p]["samples"] == [{"quant": True, "text": "Once w4."}]
    assert by_id[v]["samples"] == []

    by_id[p]["name"], by_id[p]["uploaded"] = "hand-edited", True
    rebuilt = {m["id"]: m for m in registry.build(runs, "org/bevo", built, report)["models"]}
    assert rebuilt[p]["name"] == "hand-edited" and rebuilt[p]["uploaded"]
    moved = {m["id"]: m for m in registry.build(runs, "org/other", built, report)["models"]}
    assert not moved[p]["uploaded"]  # a new repo has not received the upload


def test_registry_without_a_report_has_no_samples(tmp_path):
    write_run(tmp_path, "precision", {}, tiny_config(), ckpt=True)
    built = registry.build(tmp_path, "org/bevo", report=tmp_path / "missing.md")
    assert built["samples_note"] == "" and built["models"][0]["samples"] == []


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


def test_exported_weights_drop_tied_head_and_reload_exactly(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    from bevotron.models import Bevo
    from bevotron.site import hub

    torch.manual_seed(0)
    model = Bevo(tiny_config())
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "step": 7, "optimizer": {}}, ckpt)
    hub.export_weights(ckpt, tmp_path / "model.safetensors")

    saved = load_file(str(tmp_path / "model.safetensors"))
    assert "lm_head.weight" not in saved and "wte.weight" in saved
    fresh = Bevo(tiny_config())
    missing, unexpected = fresh.load_state_dict(saved, strict=False)
    assert missing == ["lm_head.weight"] and not unexpected
    assert fresh.lm_head.weight is fresh.wte.weight  # still tied, so it loads the saved table
    for name, tensor in model.state_dict().items():
        assert torch.equal(fresh.state_dict()[name], tensor), name


# ---- HTTP ----


@pytest.fixture
def server():
    from bevotron.site.server import make_handler

    entry = {
        "id": "tiny",
        "name": "tiny",
        "ablation": "test",
        "baseline": "tiny",
        "weights": "tiny",
        "hf_repo": "org/bevo",
        "hf_path": "tiny",
        "uploaded": False,
        "config": tiny_config().to_dict(),
        "results": {},
        "samples": [{"quant": False, "text": "Once"}],
    }
    reg = {"hf_repo": "org/bevo", "samples_note": "note", "models": [entry]}
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(reg, port))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield port
    srv.shutdown()
    srv.server_close()


def request(port, method, path, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, headers={"Host": host or f"localhost:{port}"})
    r = conn.getresponse()
    body = r.read()
    return r.status, json.loads(body) if "json" in r.getheader("Content-Type", "") else None


def test_server_serves_state_and_page_to_localhost(server):
    status, state = request(server, "GET", "/api/state")
    assert status == 200 and state["samples_note"] == "note"
    model = state["models"][0]
    assert model["id"] == "tiny" and model["samples"][0]["text"] == "Once"
    assert model["cost"]["weight_bits"] == 8 and "config" not in model
    assert request(server, "GET", "/")[0] == 200
    assert request(server, "GET", "/api/jobs")[0] == 404


def test_server_rejects_other_hosts_and_writes(server):
    assert request(server, "GET", "/api/state", host="evil.example")[0] == 403
    assert request(server, "GET", "/", host=f"attacker.test:{server}")[0] == 403
    assert request(server, "POST", "/api/state")[0] == 501  # read-only
