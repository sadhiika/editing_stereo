#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judges import RubricJudge


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate rubric JSON schema and constraints")
    parser.add_argument("--rubric-path", default="data/rubric.json")
    args = parser.parse_args()

    RubricJudge(rubric_path=args.rubric_path, model="gemini-2.0-flash")
    print(f"Rubric validation OK: {args.rubric_path}")


if __name__ == "__main__":
    main()
