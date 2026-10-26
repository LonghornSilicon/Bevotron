"""Train one run config. Writes runs/<id>/metrics.jsonl, ckpt.pt and eval.json."""

from __future__ import annotations

import json
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from bevotron.config import RunConfig
from bevotron.data.loader import get_batch
from bevotron.data.prepare import load_split, prepare
from bevotron.eval.quality import evaluate_run
from bevotron.models import Bevo
from bevotron.train.schedule import lr_at


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def quick_val_loss(model, data, config: RunConfig, rng, device, ctx) -> float:
    model.eval()
    losses = []
    batch = config.train.batch_tokens // config.model.context_len
    for _ in range(config.train.eval_batches):
        x, y = get_batch(data, batch, config.model.context_len, rng, device)
        with ctx:
            _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def find_reusable_checkpoint(config: RunConfig, runs_root: Path, exclude: Path) -> Path | None:
    """A finished run with the same training key trained identical weights; reuse its checkpoint."""
    key = config.training_key()
    for run_json in sorted(runs_root.glob("*/run.json")):
        folder = run_json.parent
        if folder == exclude:
            continue
        try:
            meta = json.loads(run_json.read_text())
        except json.JSONDecodeError:
            continue
        if (
            meta.get("training_key") == key
            and (folder / "ckpt.pt").exists()
            and (folder / "eval.json").exists()
        ):
            return folder / "ckpt.pt"
    return None


def train(config: RunConfig, run_dir: Path, reuse: bool = True, resume: bool = False) -> dict:
    t = config.train
    device = pick_device(t.device)
    if reuse and config.quant.mode == "ptq":
        existing = find_reusable_checkpoint(config, run_dir.parent, run_dir)
        if existing is not None:
            print(f"reusing checkpoint {existing} (same training key); evaluating only")
            shutil.copyfile(existing, run_dir / "ckpt.pt")
            result = evaluate_run(config, run_dir, device=device)
            result["reused_checkpoint"] = str(existing)
            (run_dir / "eval.json").write_text(json.dumps(result, indent=2))
            return result
    torch.manual_seed(t.seed)
    rng = np.random.default_rng(t.seed)
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    meta = prepare(config)
    train_data, val_data = load_split(config, "train"), load_split(config, "val")
    model = Bevo(config).to(device)
    assert model.non_embedding_params() == config.non_embedding_params(), "param count drift"
    lr, steps = t.lr, t.steps
    if config.quant.mode == "qat":
        from bevotron.quant.qat import apply_qat

        source = find_reusable_checkpoint(config.full_precision_twin(), run_dir.parent, run_dir)
        if source is None:
            raise FileNotFoundError(
                "QAT fine-tunes from the full-precision checkpoint: run the baseline "
                "(or the PTQ row, which shares its training key) first"
            )
        model.load_state_dict(torch.load(source, map_location=device)["model"])
        apply_qat(model, config.quant.scheme)
        lr, steps = config.quant.qat_lr, config.quant.qat_total_tokens // t.batch_tokens
        print(f"QAT: fine-tuning from {source} for {steps:,} steps at lr {lr:.1e}")
    optimizer = model.configure_optimizer(lr, t.weight_decay)

    start_step, best_val = 0, float("inf")
    if resume and (run_dir / "ckpt.pt").exists():
        ckpt = torch.load(run_dir / "ckpt.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"], ckpt["val_loss"]
        print(f"resuming from step {start_step} (val loss {best_val:.4f})")

    batch = t.batch_tokens // config.model.context_len
    metrics = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    model.train()
    for step in range(start_step, steps + 1):
        step_lr = lr_at(step, steps, lr, t.warmup_frac, t.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = step_lr
        if step % t.eval_every == 0 or step == steps:
            val = quick_val_loss(model, val_data, config, rng, device, ctx)
            metrics.write(json.dumps({"step": step, "val_loss": val, "lr": step_lr}) + "\n")
            metrics.flush()
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.to_dict(),
                "step": step,
                "val_loss": val,
            }
            if step > start_step:
                torch.save(state, run_dir / "ckpt.pt")  # latest: what --resume continues from
            if val < best_val:
                best_val = val
                torch.save(state, run_dir / "ckpt_best.pt")  # lowest val loss seen
        if step == steps:
            break
        x, y = get_batch(train_data, batch, config.model.context_len, rng, device)
        with ctx:
            _, loss, _ = model(x, y)
        loss.backward()
        if t.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % t.log_every == 0:
            elapsed = time.time() - t0
            metrics.write(
                json.dumps(
                    {
                        "step": step,
                        "train_loss": loss.item(),
                        "lr": lr,
                        "tokens": (step + 1) * t.batch_tokens,
                        "elapsed_s": round(elapsed, 1),
                    }
                )
                + "\n"
            )
            print(f"step {step}/{steps}  loss {loss.item():.4f}  lr {step_lr:.2e}  {elapsed:.0f}s")
    metrics.close()

    if (run_dir / "ckpt_best.pt").exists():  # score the best model, not just the last one
        shutil.copyfile(run_dir / "ckpt_best.pt", run_dir / "ckpt.pt")
    result = evaluate_run(config, run_dir, device=device)
    result["val_tokens"], result["val_bytes"] = meta.val_tokens, meta.val_bytes
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2))
    return result
