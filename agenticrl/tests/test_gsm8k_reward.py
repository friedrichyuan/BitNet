import pytest

from agenticrl.rewards.gsm8k import compute_score


@pytest.mark.parametrize(
    "response, ground_truth, expected",
    [
        ("She has 3 + 4 = 7 apples.\n#### 7", "7", 1.0),
        ("#### 1,234", "1234", 1.0),
        ("####1234", "1234", 1.0),
        ("#### $1234.00", "1234", 1.0),
        ("The answer is #### -5.", "-5", 1.0),
        ("#### 7\nWait, recheck... #### 8", "7", 0.0),  # last marker wins
        ("The answer is 1234.", "1234", 0.0),  # right number, missing the format
        ("#### 123", "1234", 0.0),
        ("", "1234", 0.0),
    ],
)
def test_strict_score(response, ground_truth, expected):
    assert compute_score("openai/gsm8k", response, ground_truth)["score"] == expected


def test_diagnostics():
    out = compute_score("openai/gsm8k", "so the total is 72 clips", "72")
    assert out == {"score": 0.0, "acc": 0.0, "acc_flexible": 1.0, "has_marker": 0.0}
    out = compute_score("openai/gsm8k", "48 + 24 = 72\n#### 72", "72")
    assert out == {"score": 1.0, "acc": 1.0, "acc_flexible": 1.0, "has_marker": 1.0}


def test_keys_are_constant():
    # verl stacks reward_extra_info per key, so every call must return the same keys.
    keys = {frozenset(compute_score("openai/gsm8k", r, "3")) for r in ["", "#### 3", "3", "#### x"]}
    assert len(keys) == 1
