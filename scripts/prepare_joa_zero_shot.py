"""Remove RoBERTa labels and retain only JoA segment boundaries and text."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.joa_zero_shot import prepare_segment_row, segment_type_counts  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="RoBERTa-labelled JoA JSON")
    parser.add_argument("--output", required=True, help="label-free output JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input, encoding="utf-8") as handle:
        source = json.load(handle)
    rows = [prepare_segment_row(row) for row in source]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[prepared] {len(rows)} articles -> {output}")
    print(f"[segments] {dict(segment_type_counts(rows))}")
    print("[verified] RoBERTa stance attributes were not retained")


if __name__ == "__main__":
    main()
