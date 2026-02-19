#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biaswipe.database import Database


def normalize_input(data: object, model_name: str) -> List[Tuple[str, str]]:
    """Accept multiple response file formats and return [(prompt_id, response_text)]."""
    rows: List[Tuple[str, str]] = []

    if isinstance(data, dict):
        # format A: {prompt_id: "response text"}
        if all(isinstance(v, str) for v in data.values()):
            for pid, txt in data.items():
                rows.append((str(pid), txt))
            return rows

        # format B: {prompt_id: {model_a: "...", model_b: "..."}}
        for pid, val in data.items():
            if isinstance(val, dict):
                if model_name in val and isinstance(val[model_name], str):
                    rows.append((str(pid), val[model_name]))
                # fallback to first string value
                elif any(isinstance(v, str) for v in val.values()):
                    first_txt = next(v for v in val.values() if isinstance(v, str))
                    rows.append((str(pid), first_txt))
        if rows:
            return rows

    if isinstance(data, list):
        # format C: [{prompt_id|id, response|text}]
        for i, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            pid = item.get("prompt_id") or item.get("id") or f"prompt_{i:03d}"
            txt = item.get("response") or item.get("response_text") or item.get("text")
            if isinstance(txt, str):
                rows.append((str(pid), txt))
        if rows:
            return rows

    raise ValueError("Unsupported response JSON format")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import model responses from JSON into SQLite responses table")
    parser.add_argument("--db-path", default="stereowipe.db")
    parser.add_argument("--file", required=True, help="Path to response JSON file")
    parser.add_argument("--model", required=True, help="Model name key to store rows under")
    parser.add_argument("--provider", default="import")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    p = Path(args.file)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    data = json.loads(p.read_text())
    rows = normalize_input(data, model_name=args.model)
    if args.limit > 0:
        rows = rows[: args.limit]

    db = Database(args.db_path)
    inserted = 0
    for prompt_id, response_text in rows:
        db.insert_or_update_response(
            prompt_id=prompt_id,
            model_name=args.model,
            response_text=response_text,
            provider=args.provider,
            metadata_json=json.dumps({"source_file": str(p.name)}),
        )
        inserted += 1

    print(f"Imported {inserted} responses for model={args.model} from {p}")


if __name__ == "__main__":
    main()
