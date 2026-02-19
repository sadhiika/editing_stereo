#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# ensure repo root is on import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import random
import time
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

from biaswipe.database import Database
from model_clients import available_models, get_model_client
from scripts.pipeline_utils import load_prompts_any_format


def iter_prompts(prompts: Dict[str, Dict[str, str]], limit: int = 0) -> Iterable[Tuple[str, Dict[str, str]]]:
    items = list(prompts.items())
    if limit > 0:
        items = items[:limit]
    return items


def choose_models(model_arg: str) -> List[str]:
    registry = available_models()
    if model_arg == "all":
        return list(registry.keys())
    chosen = [m.strip() for m in model_arg.split(",") if m.strip()]
    unknown = [m for m in chosen if m not in registry]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Known: {sorted(registry.keys())}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect model responses for StereoWipe prompts")
    parser.add_argument("--db-path", default="stereowipe.db")
    parser.add_argument("--prompts-path", default="data/prompts.json")
    parser.add_argument("--models", default="all", help="Comma-separated model list or 'all'")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of prompts per model")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true", help="No network calls; store synthetic responses")
    parser.add_argument("--sleep-ms", type=int, default=250, help="Sleep between requests")
    args = parser.parse_args()

    prompts_path = Path(args.prompts_path)
    if not prompts_path.exists() and Path("data/prompts_openweight_100.json").exists():
        prompts_path = Path("data/prompts_openweight_100.json")

    prompts = load_prompts_any_format(str(prompts_path))
    selected_models = choose_models(args.models)

    db = Database(args.db_path)
    db.import_prompts_from_json(prompts)

    print(f"Loaded {len(prompts)} prompts from {prompts_path}")
    print(f"Collecting responses for models: {', '.join(selected_models)}")

    for model in selected_models:
        provider = available_models()[model][0]
        client = None
        if not args.dry_run:
            client = get_model_client(model)

        progress = tqdm(list(iter_prompts(prompts, args.limit)), desc=f"{model}")
        for prompt_id, prompt in progress:
            prompt_text = prompt["text"]
            try:
                if args.dry_run:
                    response_text = f"[DRY_RUN:{model}] {prompt_text[:120]}"
                    raw = {"dry_run": True, "model": model}
                else:
                    result = client.generate(prompt_text, temperature=args.temperature, max_tokens=args.max_tokens)
                    response_text = result.text
                    raw = result.raw

                db.insert_or_update_response(
                    prompt_id=prompt_id,
                    model_name=model,
                    response_text=response_text,
                    provider=provider,
                    metadata_json=json.dumps(raw)[:50000],
                )
            except Exception as exc:
                progress.write(f"[{model}] failed for {prompt_id}: {exc}")

            time.sleep(max(0.0, args.sleep_ms / 1000.0) + random.random() * 0.1)

    total_saved = len(db.get_responses())
    print(f"Done. Responses currently stored: {total_saved}")


if __name__ == "__main__":
    main()
