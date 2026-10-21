"""Convert the local GSM8K parquet files into verl's RL dataset format.

Usage (from the BitNet repo root):
    python -m agenticrl.data.gsm8k --input_dir datasets/gsm8k/main --output_dir datasets/gsm8k_verl
"""

import argparse
import os
import re

import pandas as pd

DATA_SOURCE = "openai/gsm8k"
# Same instruction as verl's examples/data_preprocess/gsm8k.py, so results are comparable.
INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def extract_solution(answer: str) -> str:
    match = re.search(r"#### (\-?[0-9\.\,]+)", answer)
    assert match is not None, f"no '#### <number>' in GSM8K answer: {answer!r}"
    return match.group(1).replace(",", "")


def to_verl_rows(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for idx, (question, answer) in enumerate(zip(df["question"], df["answer"], strict=True)):
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": f"{question} {INSTRUCTION}"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": extract_solution(answer)},
                "extra_info": {"split": split, "index": idx, "question": question, "answer": answer},
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", default="datasets/gsm8k/main", help="dir with train-*.parquet / test-*.parquet")
    parser.add_argument("--output_dir", default="datasets/gsm8k_verl")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for split in ("train", "test"):
        files = sorted(f for f in os.listdir(args.input_dir) if f.startswith(f"{split}-") and f.endswith(".parquet"))
        assert files, f"no {split}-*.parquet in {args.input_dir}"
        raw = pd.concat([pd.read_parquet(os.path.join(args.input_dir, f)) for f in files], ignore_index=True)
        out = to_verl_rows(raw, split)
        path = os.path.join(args.output_dir, f"{split}.parquet")
        out.to_parquet(path)
        print(f"{split}: {len(out)} rows -> {path}")


if __name__ == "__main__":
    main()
