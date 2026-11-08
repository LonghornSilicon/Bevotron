"""Move run folders to and from the Hugging Face Hub.

One repo holds every run, one folder per run ID:
    <id>/config.yaml, run.json, eval.json, tokenizer.json   (every run)
    <id>/model.safetensors                                  (runs that own a checkpoint)
PTQ runs share their baseline's weights, so only the folder named by the registry's
`weights` field carries model.safetensors. Weights are saved without optimizer state, and
the output layer is dropped when it is tied to the token table (the model re-ties it).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from bevotron.config import RunConfig
from bevotron.site.registry import RUNS, hf_url

WEIGHTS = "model.safetensors"
TOKENIZER = "tokenizer.json"


def export_weights(ckpt_path: Path, out_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["model"]
    tied = (
        "lm_head.weight" in state
        and state["lm_head.weight"].data_ptr() == state["wte.weight"].data_ptr()
    )
    if tied:
        state = {k: v for k, v in state.items() if k != "lm_head.weight"}
    save_file(
        {k: v.contiguous() for k, v in state.items()},
        str(out_path),
        metadata={"step": str(ckpt.get("step", "")), "tied": str(tied)},
    )


def stage_run(entry: dict, out_dir: Path, runs: Path = RUNS) -> Path:
    """Copy one run's files into out_dir/<id>, ready to upload."""
    from bevotron.data.prepare import resolve_tokenizer_path

    run_dir = runs / entry["id"]
    folder = out_dir / entry["hf_path"]
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "run.json", "eval.json"):
        if (run_dir / name).exists():
            shutil.copy(run_dir / name, folder / name)
    config = RunConfig.from_dict(entry["config"])
    shutil.copy(resolve_tokenizer_path(config), folder / TOKENIZER)
    if entry["weights"] == entry["id"]:
        export_weights(run_dir / "ckpt.pt", folder / WEIGHTS)
    return folder


def push(registry: dict, ids: list[str], repo: str, private: bool = False) -> list[str]:
    """Upload runs (weights owners first) and mark them uploaded in the registry.
    Needs a Hugging Face token with write access (`hf auth login` or HF_TOKEN)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    by_id = {m["id"]: m for m in registry["models"]}
    wanted = [by_id[i] for i in ids]
    for owner in {m["weights"] for m in wanted} - set(ids):
        wanted.append(by_id[owner])  # a PTQ run is useless without its baseline's weights
    wanted.sort(key=lambda m: m["weights"] != m["id"])
    done = []
    for entry in wanted:
        entry["hf_repo"], entry["hf"] = repo, hf_url(repo, entry["hf_path"])
        with tempfile.TemporaryDirectory() as tmp:
            folder = stage_run(entry, Path(tmp))
            api.upload_folder(
                repo_id=repo,
                folder_path=str(folder),
                path_in_repo=entry["hf_path"],
                commit_message=f"{entry['ablation']}: {entry['name']} ({entry['id']})",
            )
        entry["uploaded"] = True
        done.append(entry["id"])
    registry["hf_repo"] = repo
    return done


def fetch(entry: dict, owner: dict) -> tuple[Path, Path]:
    """Download (or reuse the Hub cache for) a run's tokenizer and its owner's weights."""
    from huggingface_hub import hf_hub_download

    repo = entry["hf_repo"]
    tokenizer = hf_hub_download(repo, f"{entry['hf_path']}/{TOKENIZER}")
    weights = hf_hub_download(owner["hf_repo"], f"{owner['hf_path']}/{WEIGHTS}")
    return Path(tokenizer), Path(weights)


def local_files(entry: dict, runs: Path = RUNS) -> tuple[Path, Path] | None:
    """The same files from ./runs and ./artifacts, if this machine trained the run."""
    from bevotron.data.prepare import resolve_tokenizer_path

    ckpt = runs / entry["weights"] / "ckpt.pt"
    tokenizer = resolve_tokenizer_path(RunConfig.from_dict(entry["config"]))
    if ckpt.exists() and tokenizer.exists():
        return tokenizer, ckpt
    return None
