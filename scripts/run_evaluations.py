#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from biaswipe.database import Database
from judges import RubricJudge


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rubric-based evaluations over collected responses")
    parser.add_argument("--db-path", default="stereowipe.db")
    parser.add_argument("--rubric-path", default="data/rubric.json")
    parser.add_argument("--judge-model", default="gemini-2.0-flash")
    parser.add_argument("--judge-name", default="gemini_rubric_v1")
    parser.add_argument("--model", default="", help="Optional model filter")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic heuristic judge")
    args = parser.parse_args()

    db = Database(args.db_path)
    judge = RubricJudge(rubric_path=args.rubric_path, model=args.judge_model)

    pending = db.get_pending_rubric_responses(
        judge_name=args.judge_name,
        model_name=args.model or None,
        limit=args.limit or None,
    )

    if not pending:
        print("No pending responses to evaluate.")
        return

    print(f"Evaluating {len(pending)} responses with {args.judge_name} (dry_run={args.dry_run})")

    ok = 0
    fail = 0
    for row in tqdm(pending, desc="rubric-eval"):
        try:
            result = judge.judge(
                prompt_text=row.get("prompt_text") or "",
                response_text=row["response_text"],
                dry_run=args.dry_run,
            )

            db.insert_or_update_rubric_evaluation(
                prompt_id=row["prompt_id"],
                model_name=row["model_name"],
                judge_name=args.judge_name,
                answers_json=json.dumps(result.answers),
                reasoning_json=json.dumps(result.reasoning),
                raw_score=result.raw_score,
                normalized_score=result.normalized_score,
                severity_0_to_5=result.severity_0_to_5,
            )
            ok += 1
        except Exception as exc:
            fail += 1
            tqdm.write(f"Failed {row['model_name']}:{row['prompt_id']} -> {exc}")

    print(f"Completed. Success={ok}, Failed={fail}")


if __name__ == "__main__":
    main()
