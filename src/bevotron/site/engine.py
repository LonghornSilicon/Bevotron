"""Load registry models on this machine and run them: side-by-side generation and
validation evals. Models come from ./runs when this machine trained them (source "local")
or from the Hugging Face Hub (source "hub"); "auto" prefers local files."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bevotron.config import RunConfig
from bevotron.site import hub

EVALS = Path("results/site_evals.json")
QUICK_FRACTION = 0.05  # quick eval: the first 5% of the validation split, same text per model


@dataclass
class Loaded:
    model: Any
    tokenizer: Any
    config: RunConfig
    source: str


class Engine:
    def __init__(
        self,
        registry: dict,
        device: str = "auto",
        source: str = "auto",
        max_loaded: int = 4,
        evals_path: Path = EVALS,
    ):
        from bevotron.train.loop import pick_device

        self.registry = registry
        self.models = {m["id"]: m for m in registry["models"]}
        self.device = pick_device(device)
        self.source = source
        self.max_loaded = max_loaded
        self.evals_path = evals_path
        self._loaded: OrderedDict[tuple[str, bool], Loaded] = OrderedDict()
        self._lock = threading.Lock()

    # ---- where a model's files come from ----
    def availability(self, entry: dict) -> str:
        if self.source != "hub" and hub.local_files(entry):
            return "local"
        owner = self.models.get(entry["weights"], entry)
        if self.source != "local" and entry.get("uploaded") and owner.get("uploaded"):
            return "hub"
        return "missing"

    def _files(self, entry: dict) -> tuple[Path, Path, str]:
        where = self.availability(entry)
        if where == "local":
            tokenizer, weights = hub.local_files(entry)
            return tokenizer, weights, "local"
        if where == "hub":
            tokenizer, weights = hub.fetch(entry, self.models[entry["weights"]])
            return tokenizer, weights, "hub"
        raise FileNotFoundError(
            f"{entry['id']} is not in ./runs and not uploaded to {entry['hf_repo']}; "
            "run `bevotron hub push` on the machine that trained it"
        )

    # ---- loading ----
    def load(self, model_id: str, quant: bool) -> Loaded:
        """A model in eval mode; quant=True applies the run's PTQ scheme, as eval does."""
        key = (model_id, quant)
        with self._lock:
            if key in self._loaded:
                self._loaded.move_to_end(key)
                return self._loaded[key]
        import torch

        from bevotron.models import Bevo
        from bevotron.quant.ptq import apply_ptq
        from bevotron.tokenizer.bpe import BPETokenizer

        entry = self.models[model_id]
        tokenizer_path, weights_path, source = self._files(entry)
        raw = json.loads(json.dumps(entry["config"]))
        raw["tokenizer"]["path"] = str(tokenizer_path)  # the data cache is keyed by content
        config = RunConfig.from_dict(raw)
        tokenizer = BPETokenizer.load(tokenizer_path)
        if tokenizer.vocab_size != config.tokenizer.vocab_size:
            raise ValueError(f"{model_id}: tokenizer size does not match its config")
        model = Bevo(config)
        if weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(weights_path))
        else:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed = {"lm_head.weight"} if config.model.tie_embeddings else set()
        if unexpected or set(missing) - allowed:
            raise ValueError(f"{model_id}: checkpoint keys do not match: {missing} {unexpected}")
        model.to(self.device).eval()
        if quant and config.quant.scheme != "none":
            apply_ptq(model, config.quant.scheme)
        loaded = Loaded(model, tokenizer, config, source)
        with self._lock:
            self._loaded[key] = loaded
            while len(self._loaded) > self.max_loaded:
                self._loaded.popitem(last=False)
        return loaded

    # ---- generation ----
    def generate(
        self,
        model_id: str,
        prompt: str,
        quant: bool = True,
        seed: int = 0,
        temperature: float = 0.7,
        top_k: int = 50,
        max_new_tokens: int = 150,
    ) -> dict:
        import torch

        m = self.load(model_id, quant)
        ids = m.tokenizer.encode(prompt) if prompt else []
        context = ids or [m.tokenizer.eos_token_id]
        x = torch.tensor([context], device=self.device)
        torch.manual_seed(seed)
        self._sync()
        start = time.perf_counter()
        with torch.no_grad():
            out = m.model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        self._sync()
        seconds = time.perf_counter() - start
        new = out[0, len(context) :].tolist()
        if m.tokenizer.eos_token_id in new:
            new = new[: new.index(m.tokenizer.eos_token_id)]
        scheme = m.config.quant.scheme
        return {
            "id": model_id,
            "quant": quant and scheme != "none",
            "scheme": scheme if quant and scheme != "none" else "fp32",
            "text": m.tokenizer.decode(ids + new),
            "prompt_tokens": len(ids),
            "new_tokens": len(new),
            "seconds": seconds,
            "tokens_per_s": max_new_tokens / seconds if seconds else None,
            "source": m.source,
            "device": self.device,
        }

    # ---- evaluation ----
    def evaluate(self, model_id: str, scope: str = "quick", progress=None) -> dict:
        """Validation loss, perplexity and bits per byte at full precision and under the
        run's scheme. quick scores the first 5% of the validation split; full scores all
        of it, the same number `bevotron run` records in eval.json."""
        from bevotron.data.prepare import val_split
        from bevotron.eval.quality import (
            bits_per_byte,
            eval_batch_size,
            full_val_loss,
            perplexity,
        )

        if scope not in ("quick", "full"):
            raise ValueError("scope must be quick or full")
        report = progress or (lambda fraction, message: None)
        report(0.0, "loading model")
        fp = self.load(model_id, quant=False)
        config = fp.config
        report(0.02, "loading validation data")
        data, val_tokens, val_bytes = val_split(config)
        ctx = config.model.context_len
        total = (len(data) - 1) // ctx
        windows = total if scope == "full" else max(1, int(total * QUICK_FRACTION))
        batch = eval_batch_size(ctx, config.train.batch_tokens)
        quantized = config.quant.scheme != "none"
        passes = 2 if quantized else 1
        start = time.perf_counter()

        def stage(i: int, name: str):
            def update(done: float) -> None:
                report(0.05 + 0.95 * (i + done) / passes, f"{name} {done:.0%}")

            return update

        fp_loss = full_val_loss(
            fp.model, data, ctx, batch, self.device, max_windows=windows, progress=stage(0, "fp32")
        )
        fp.model.eval()
        result = {
            "id": model_id,
            "scope": scope,
            "windows": windows,
            "tokens": windows * ctx,
            "val_loss_fp": fp_loss,
            "perplexity_fp": perplexity(fp_loss),
            "bpb_fp": bits_per_byte(fp_loss, val_tokens, val_bytes),
        }
        q_loss = fp_loss
        if quantized:
            q = self.load(model_id, quant=True)
            q_loss = full_val_loss(
                q.model,
                data,
                ctx,
                batch,
                self.device,
                max_windows=windows,
                progress=stage(1, config.quant.scheme),
            )
            q.model.eval()
        result.update(
            scheme=config.quant.scheme,
            val_loss=q_loss,
            perplexity=perplexity(q_loss),
            bpb=bits_per_byte(q_loss, val_tokens, val_bytes),
            seconds=time.perf_counter() - start,
            device=self.device,
            source=fp.source,
            finished=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self._save_eval(result)
        report(1.0, "done")
        return result

    def local_evals(self) -> dict:
        return json.loads(self.evals_path.read_text()) if self.evals_path.exists() else {}

    def _save_eval(self, result: dict) -> None:
        with self._lock:
            evals = self.local_evals()
            evals.setdefault(result["id"], {})[result["scope"]] = result
            self.evals_path.parent.mkdir(parents=True, exist_ok=True)
            self.evals_path.write_text(json.dumps(evals, indent=2) + "\n")

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()


class Jobs:
    """One worker thread runs jobs in order, so models never compete for the GPU. The page
    polls a job for its progress and partial results."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._jobs: dict[str, dict] = {}
        self._count = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, kind: str, label: str, fn) -> dict:
        with self._lock:
            self._count += 1
            job = {
                "id": str(self._count),
                "kind": kind,
                "label": label,
                "status": "queued",
                "progress": 0.0,
                "message": "queued",
                "partial": [],
                "result": None,
                "error": None,
                "created": time.time(),
            }
            self._jobs[job["id"]] = job
        self._queue.put((job, fn))
        return job

    def get(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def recent(self, n: int = 30) -> list[dict]:
        return sorted(self._jobs.values(), key=lambda j: -j["created"])[:n]

    def _work(self) -> None:
        while True:
            job, fn = self._queue.get()
            job["status"], job["message"] = "running", "starting"
            try:
                job["result"] = fn(job)
                job["status"], job["progress"], job["message"] = "done", 1.0, "done"
            except Exception as e:  # shown on the page; the worker keeps serving jobs
                job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
