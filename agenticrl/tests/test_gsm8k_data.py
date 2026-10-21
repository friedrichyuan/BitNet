import os

import pandas as pd
import pytest

from agenticrl.data.gsm8k import DATA_SOURCE, INSTRUCTION, extract_solution, to_verl_rows

RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "gsm8k", "main")


def test_extract_solution():
    assert extract_solution("so 5 * 4 = 20\n#### 1,020") == "1020"
    assert extract_solution("#### -3") == "-3"


def test_row_format():
    df = pd.DataFrame({"question": ["What is 2+3?"], "answer": ["2+3=5\n#### 5"]})
    row = to_verl_rows(df, "train").iloc[0]
    assert row["data_source"] == DATA_SOURCE
    assert row["prompt"] == [{"role": "user", "content": f"What is 2+3? {INSTRUCTION}"}]
    assert row["reward_model"] == {"style": "rule", "ground_truth": "5"}
    assert row["extra_info"]["index"] == 0


@pytest.mark.skipif(not os.path.isdir(RAW), reason="local GSM8K parquet not present")
def test_local_dataset_converts():
    for split, n in (("train", 7473), ("test", 1319)):
        files = [f for f in os.listdir(RAW) if f.startswith(f"{split}-")]
        out = to_verl_rows(pd.concat([pd.read_parquet(os.path.join(RAW, f)) for f in files]), split)
        assert len(out) == n
        assert out["reward_model"].map(lambda r: r["ground_truth"].lstrip("-").replace(".", "").isdigit()).all()
