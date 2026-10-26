"""LLM-as-judge: the model continues the opening of validation stories; a judge LLM grades
each continuation for grammar, creativity and consistency with the opening (1-10), following
the TinyStories evaluation. Scores are averaged over judge_samples stories.

Needs the `anthropic` package and ANTHROPIC_API_KEY. Generated stories and raw scores are
saved to runs/<id>/samples.jsonl so they can be re-judged or spot-checked by hand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from bevotron.config import RunConfig

RUBRIC = """You are grading a short children's story written by a small language model.
The story opening was given to the model; it wrote the continuation.

Rate the CONTINUATION on three criteria, each an integer from 1 (worst) to 10 (best):
- grammar: is the English correct and fluent?
- creativity: is the continuation interesting rather than generic or repetitive?
- consistency: does it follow logically from the opening (same characters, setting, plot)?

Reply with JSON only: {"grammar": N, "creativity": N, "consistency": N}

OPENING:
{opening}

CONTINUATION:
{continuation}
"""

CRITERIA = ("grammar", "creativity", "consistency")


def parse_scores(text: str) -> dict[str, float]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in judge reply: {text[:200]!r}")
    data = json.loads(match.group())
    scores = {k: float(data[k]) for k in CRITERIA}
    for k, v in scores.items():
        if not 1 <= v <= 10:
            raise ValueError(f"{k} score out of range: {v}")
    return scores


def story_openings(val_tokens, eos_token_id: int, n: int, prompt_tokens: int) -> list[list[int]]:
    """The first `prompt_tokens` tokens of the first n validation stories."""
    openings, start = [], 0
    data = val_tokens
    for i in range(len(data)):
        if data[i] == eos_token_id:
            if i - start >= prompt_tokens:
                openings.append([int(t) for t in data[start : start + prompt_tokens]])
                if len(openings) == n:
                    break
            start = i + 1
    return openings


def generate_samples(config: RunConfig, run_dir: Path, device: str = "cpu") -> list[dict]:
    import torch

    from bevotron.data.prepare import load_split, prepare, resolve_tokenizer_path
    from bevotron.models import Bevo
    from bevotron.quant.ptq import apply_ptq
    from bevotron.tokenizer.bpe import BPETokenizer

    e = config.eval
    meta = prepare(config)
    tok = BPETokenizer.load(resolve_tokenizer_path(config))
    model = Bevo(config).to(device)
    model.load_state_dict(torch.load(run_dir / "ckpt.pt", map_location=device)["model"])
    if config.quant.scheme != "none":
        apply_ptq(model, config.quant.scheme)
    model.eval()
    torch.manual_seed(config.train.seed)
    openings = story_openings(
        load_split(config, "val"), meta.eos_token_id, e.judge_samples, e.judge_prompt_tokens
    )
    samples = []
    for ids in openings:
        prompt = torch.tensor([ids], device=device)
        out = model.generate(
            prompt, e.judge_max_new_tokens, temperature=e.judge_temperature, top_k=50
        )
        continuation = out[0, len(ids) :].tolist()
        if meta.eos_token_id in continuation:
            continuation = continuation[: continuation.index(meta.eos_token_id)]
        samples.append({"opening": tok.decode(ids), "continuation": tok.decode(continuation)})
    return samples


def judge_samples(samples: list[dict], model: str) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic()
    for sample in samples:
        prompt = RUBRIC.format(opening=sample["opening"], continuation=sample["continuation"])
        reply = client.messages.create(
            model=model, max_tokens=100, messages=[{"role": "user", "content": prompt}]
        )
        sample["scores"] = parse_scores(reply.content[0].text)
    return samples


def summarize(samples: list[dict]) -> dict[str, float]:
    scored = [s["scores"] for s in samples if "scores" in s]
    if not scored:
        return {}
    summary = {f"judge_{k}": sum(s[k] for s in scored) / len(scored) for k in CRITERIA}
    summary["judge_mean"] = sum(summary.values()) / len(CRITERIA)
    summary["judge_n"] = len(scored)
    return summary


def judge_run(config: RunConfig, run_dir: Path, device: str = "cpu") -> dict[str, float]:
    samples = generate_samples(config, run_dir, device)
    samples = judge_samples(samples, config.eval.judge_model)
    with open(run_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    summary = summarize(samples)
    eval_path = run_dir / "eval.json"
    result = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    result.update(summary)
    eval_path.write_text(json.dumps(result, indent=2))
    return summary
