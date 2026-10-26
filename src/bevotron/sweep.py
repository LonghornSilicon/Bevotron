"""Turn an ablation YAML into run configs.

Ablation file keys:
  name:      ablation name (results/<name>.csv)
  question:  free text
  overrides: dotted keys applied to every run              (optional)
  pairs:     list of dotted-key dicts, one run each         (optional)
  grid:      dotted key -> list of values, crossed          (optional)
Runs = each entry of pairs (or the empty override) crossed with the grid.
The baseline itself is always run 0 so every ablation compares against it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bevotron.config import RunConfig, apply_overrides


@dataclass(frozen=True)
class Run:
    ablation: str
    overrides: dict[str, Any]
    config: RunConfig

    @property
    def run_id(self) -> str:
        return self.config.run_id()

    @property
    def label(self) -> str:
        if not self.overrides:
            return "baseline"
        return ",".join(f"{k.split('.')[-1]}={v}" for k, v in self.overrides.items())


def expand(ablation: dict[str, Any], baseline: dict[str, Any]) -> list[Run]:
    name = ablation["name"]
    common = ablation.get("overrides", {}) or {}
    pairs = ablation.get("pairs") or [{}]
    grid = ablation.get("grid", {}) or {}
    keys, values = list(grid), list(grid.values())
    runs = [Run(name, {}, RunConfig.from_dict(apply_overrides(baseline, common)))]
    seen = {runs[0].run_id}
    for pair in pairs:
        for combo in itertools.product(*values) if keys else [()]:
            overrides = {**common, **pair, **dict(zip(keys, combo, strict=True))}
            config = RunConfig.from_dict(apply_overrides(baseline, overrides))
            if config.run_id() in seen:  # a variant identical to the baseline
                continue
            seen.add(config.run_id())
            runs.append(Run(name, overrides, config))
    return runs


def load_sweep(ablation_path: str | Path, baseline_path: str | Path) -> list[Run]:
    with open(ablation_path, encoding="utf-8") as f:
        ablation = yaml.safe_load(f)
    with open(baseline_path, encoding="utf-8") as f:
        baseline = yaml.safe_load(f)
    return expand(ablation, baseline)
