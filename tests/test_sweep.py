from pathlib import Path

import yaml

from bevotron.sweep import expand, load_sweep

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs" / "baseline.yaml"
ABLATIONS = ROOT / "configs" / "ablations"


def test_every_ablation_file_expands_and_starts_with_baseline():
    expected = {
        "scaling": 1 + 5 * 5 - 1,  # 512 x 500M would equal the baseline... (504M differs)
        "layers": 1 + 6,
        "hidden_dim": 1 + 2,
        "mlp_ratio": 1 + 2,
        "attention": 1 + 2,
        "vocab_size": 1 + 5,
        "context_length": 1 + 2,
        "precision": 1 + 2,
    }
    for path in sorted(ABLATIONS.glob("*.yaml")):
        runs = load_sweep(path, BASELINE)
        assert runs[0].label == "baseline"
        assert len({r.run_id for r in runs}) == len(runs), path.name
        if path.stem in expected and path.stem != "scaling":
            assert len(runs) == expected[path.stem], path.name
    scaling = load_sweep(ABLATIONS / "scaling.yaml", BASELINE)
    assert len(scaling) == 1 + 25  # 5 sizes x 5 budgets, none equal to the 504M baseline


def test_grid_and_pairs_cross_and_dedupe_against_baseline():
    baseline = yaml.safe_load(BASELINE.read_text())
    ablation = {
        "name": "t",
        "pairs": [{"model.n_layer": 8}, {"model.n_layer": 4}],  # first pair == baseline
        "grid": {"model.mlp_ratio": [4, 2]},
    }
    runs = expand(ablation, baseline)
    labels = [r.label for r in runs]
    assert labels[0] == "baseline"
    assert len(runs) == 1 + 3  # (8,4) is the baseline and is deduped
    assert all(r.config.model.n_layer in (4, 8) for r in runs)


def test_overrides_apply_to_all_runs():
    baseline = yaml.safe_load(BASELINE.read_text())
    runs = expand(
        {"name": "t", "overrides": {"train.tokens": "32M"}, "grid": {"model.n_layer": [2]}},
        baseline,
    )
    assert all(r.config.train.total_tokens == 32_000_000 for r in runs)
