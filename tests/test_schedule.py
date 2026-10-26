from bevotron.train.schedule import lr_at


def test_warmup_then_cosine_to_floor():
    lr, steps = 6e-4, 1000
    values = [lr_at(s, steps, lr, warmup_frac=0.1, min_lr_ratio=0.1) for s in range(steps + 1)]
    assert values[0] < values[50] < values[99]  # warming up
    assert abs(values[100] - lr) < 1e-12  # peak right after warmup
    assert all(a >= b - 1e-12 for a, b in zip(values[100:], values[101:], strict=False))  # decays
    assert abs(values[-1] - lr * 0.1) < 1e-12  # ends at the floor
