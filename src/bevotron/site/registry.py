"""configs/models.json: every run, the ablation it belongs to, its scores, its saved
samples, and where its weights live on the Hugging Face Hub. `bevotron registry` rebuilds it
from ./runs and results/report.md; names and notes edited by hand survive a rebuild."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bevotron.config import RunConfig

REGISTRY = Path("configs/models.json")
RUNS = Path("runs")
REPORT = Path("results/report.md")
DEFAULT_REPO = "LonghornSilicon/bevo-ablations"
RESULT_KEYS = (
    "bpb",
    "bpb_fp",
    "perplexity",
    "perplexity_fp",
    "val_loss",
    "val_loss_fp",
    "judge_mean",
    "checkpoint_step",
)
KEEP = ("name", "notes")  # hand-edited fields a rebuild leaves alone


def hf_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/tree/main/{path}"


SAMPLE_HEADING = re.compile(r"^#### bevotron sample (\w+)( --quant)?", re.M)
SAMPLE_BLOCK = re.compile(r"```text\n(.*?)\n```", re.S)


def read_samples(report: Path = REPORT) -> tuple[str, dict[str, list[dict]]]:
    """The report's saved samples: (how they were made, {run id: [{quant, text}]})."""
    if not report.exists() or "## Samples" not in report.read_text():
        return "", {}
    section = report.read_text().split("## Samples", 1)[1]
    note = " ".join(section.split("###", 1)[0].split())
    samples: dict[str, list[dict]] = {}
    for heading in SAMPLE_HEADING.finditer(section):
        block = SAMPLE_BLOCK.search(section, heading.end())
        if block:
            samples.setdefault(heading.group(1), []).append(
                {"quant": bool(heading.group(2)), "text": block.group(1)}
            )
    return note, samples


def short_name(overrides: dict[str, Any], config: RunConfig, ablation: str = "") -> str:
    """'w4a8kv4 PTQ', '16k vocab'; a baseline is named by its study, scheme and vocabulary."""
    if not overrides:
        detail = f"{config.quant.scheme}, {config.tokenizer.vocab_size // 1000}k vocab"
        return f"{ablation} baseline ({detail})".strip()
    parts = []
    for key, value in overrides.items():
        if key == "train.tokens":
            continue  # follows from the vocabulary in the fixed-text vocab study
        if key == "quant.scheme":
            parts.append(str(value))
        elif key == "quant.mode":
            parts.append(str(value).upper())
        elif key == "tokenizer.vocab_size":
            parts.append(f"{int(value) // 1000}k vocab")
        else:
            parts.append(f"{key.split('.')[-1]}={value}")
    return " ".join(parts) or "baseline"


def build(
    runs: Path = RUNS,
    repo: str = DEFAULT_REPO,
    existing: dict | None = None,
    report: Path = REPORT,
) -> dict:
    """One entry per run folder that has a config and a run.json."""
    old = {m["id"]: m for m in (existing or {}).get("models", [])}
    samples_note, samples = read_samples(report)
    raw = []
    for run_json in sorted(runs.glob("*/run.json")):
        meta = json.loads(run_json.read_text())
        config = RunConfig.load(run_json.with_name("config.yaml"))
        eval_path = run_json.with_name("eval.json")
        result = json.loads(eval_path.read_text()) if eval_path.exists() else {}
        raw.append((meta, config, result, (run_json.parent / "ckpt.pt").exists()))

    baselines = {m["ablation"]: m["run_id"] for m, *_ in raw if m["label"] == "baseline"}
    owners: dict[str, str] = {}  # training key -> run whose folder holds the weights
    for meta, _, _, has_ckpt in sorted(raw, key=lambda r: r[0]["label"] != "baseline"):
        if has_ckpt:
            owners.setdefault(meta["training_key"], meta["run_id"])

    models = []
    for meta, config, result, _ in raw:
        run_id = meta["run_id"]
        overrides = meta.get("overrides") or {}
        entry = {
            "id": run_id,
            "name": short_name(overrides, config, meta["ablation"]),
            "ablation": meta["ablation"],
            "change": ", ".join(f"{k}={v}" for k, v in overrides.items()) or "baseline",
            "baseline": baselines.get(meta["ablation"]),
            "weights": owners.get(meta["training_key"], run_id),
            "hf_repo": repo,
            "hf_path": run_id,
            "hf": hf_url(repo, run_id),
            "uploaded": False,
            "notes": "",
            "config": config.to_dict(),
            "results": {k: result[k] for k in RESULT_KEYS if k in result},
            "samples": samples.get(run_id, []),
        }
        prev = old.get(run_id, {})
        for key in KEEP:
            if prev.get(key):
                entry[key] = prev[key]
        if prev.get("hf_repo") == repo:
            entry["uploaded"] = bool(prev.get("uploaded"))
        models.append(entry)

    models.sort(
        key=lambda m: (
            m["ablation"],
            m["id"] != m["baseline"],
            m["results"].get("bpb", float("inf")),
        )
    )
    return {
        "hf_repo": repo,
        "about": (
            "Every Bevotron run: the ablation it belongs to, the change from that ablation's "
            "baseline, its scores, and its folder on the Hugging Face Hub. `weights` names the "
            "folder holding the checkpoint (PTQ runs share their baseline's). Rebuild with "
            "`bevotron registry`; upload with `bevotron hub push`."
        ),
        "samples_note": samples_note,
        "models": models,
    }


def load(path: Path = REGISTRY) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `bevotron registry` first")
    return json.loads(path.read_text())


def save(registry: dict, path: Path = REGISTRY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2) + "\n")
