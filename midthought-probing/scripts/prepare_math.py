"""Optional JSONL dump of a filtered math split."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probing.data import load_math_split, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = load_math_split(split=args.split, max_items=args.max_items)
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
