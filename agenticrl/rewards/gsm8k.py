"""GSM8K rule reward for verl (``reward.custom_reward_function.path`` points at this file).

Reward is 1.0 iff the number after the last ``####`` equals the ground truth, else 0.0. This is
verl's "strict" GSM8K reward, with whitespace / ``$`` / thousands separators / trailing ``.0``
tolerated so the model is not penalized for harmless formatting. Two diagnostics ride along in
``reward_extra_info`` and are logged by verl:

* ``acc_flexible``: last number anywhere in the response is correct (ignores the format);
* ``has_marker``:  the response contains ``#### <number>`` at all.

A large gap between ``acc_flexible`` and ``acc`` means the model solves problems but misses the
output format.
"""

import re

_CLIP_CHARS = 300  # answers sit at the end; bounds regex cost on long responses
_NUMBER = r"-?\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?"
_MARKER_RE = re.compile(r"####\s*(" + _NUMBER + ")")
_ANY_NUMBER_RE = re.compile(_NUMBER)


def _to_number(text: str) -> float | None:
    text = text.replace(",", "").replace("$", "").replace(" ", "")
    try:
        return float(text)
    except ValueError:
        return None


def _matches(pred: str | None, ground_truth: str) -> bool:
    if pred is None:
        return False
    pred_value, gt_value = _to_number(pred), _to_number(ground_truth)
    return pred_value is not None and gt_value is not None and abs(pred_value - gt_value) < 1e-6


def extract_marked_answer(solution_str: str) -> str | None:
    found = _MARKER_RE.findall(solution_str[-_CLIP_CHARS:])
    return found[-1] if found else None


def extract_last_number(solution_str: str) -> str | None:
    found = _ANY_NUMBER_RE.findall(solution_str[-_CLIP_CHARS:])
    return found[-1] if found else None


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    marked = extract_marked_answer(solution_str)
    correct = _matches(marked, ground_truth)
    return {
        "score": 1.0 if correct else 0.0,
        "acc": 1.0 if correct else 0.0,
        "acc_flexible": 1.0 if _matches(extract_last_number(solution_str), ground_truth) else 0.0,
        "has_marker": 1.0 if marked is not None else 0.0,
    }
