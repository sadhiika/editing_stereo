#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# ensure repo root is on import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biaswipe.database import Database
from database import init_schema
from scripts.pipeline_utils import load_prompts_any_format


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize StereoWipe DB for pipeline use")
    parser.add_argument("--db-path", default="stereowipe.db")
    parser.add_argument("--prompts-path", default="data/prompts.json")
    parser.add_argument("--export-normalized-prompts", default="")
    args = parser.parse_args()

    # Initialize both existing app schema and pipeline schema extensions.
    Database(args.db_path)
    init_schema(args.db_path)

    prompts_path = Path(args.prompts_path)
    if not prompts_path.exists() and Path("data/prompts_openweight_100.json").exists():
        prompts_path = Path("data/prompts_openweight_100.json")

    prompts = load_prompts_any_format(str(prompts_path))
    db = Database(args.db_path)
    db.import_prompts_from_json(prompts)

    if args.export_normalized_prompts:
        out = Path(args.export_normalized_prompts)
        out.write_text(__import__("json").dumps(prompts, indent=2) + "\n")

    print(f"Initialized DB at {args.db_path}")
    print(f"Imported {len(prompts)} prompts from {prompts_path}")


if __name__ == "__main__":
    main()
