import numpy as np
import pytest

from bevotron.eval.judge import parse_scores, story_openings, summarize


def test_parse_scores_accepts_json_with_noise_and_rejects_bad_ranges():
    assert parse_scores('Sure! {"grammar": 8, "creativity": 6, "consistency": 9}') == {
        "grammar": 8.0,
        "creativity": 6.0,
        "consistency": 9.0,
    }
    with pytest.raises(ValueError):
        parse_scores('{"grammar": 11, "creativity": 6, "consistency": 9}')
    with pytest.raises(ValueError):
        parse_scores("no scores here")


def test_story_openings_take_prefixes_of_complete_stories():
    eos = 256
    data = np.array([1, 2, 3, 4, 5, eos, 6, 7, eos, 8, 9, 10, 11, eos], dtype=np.uint16)
    assert story_openings(data, eos, n=5, prompt_tokens=3) == [[1, 2, 3], [8, 9, 10]]


def test_summarize_averages_only_scored_samples():
    samples = [
        {"scores": {"grammar": 8, "creativity": 6, "consistency": 10}},
        {"scores": {"grammar": 6, "creativity": 4, "consistency": 8}},
        {"continuation": "unscored"},
    ]
    s = summarize(samples)
    assert s["judge_grammar"] == 7 and s["judge_mean"] == 7 and s["judge_n"] == 2
