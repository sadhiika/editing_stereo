#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biaswipe.database import Database
from metrics import StereoWipeMetrics


CATEGORY_TO_COLUMN = {
    "gender": "gender_score",
    "race": "race_score",
    "race_ethnicity": "race_score",
    "religion": "religion_score",
    "nationality": "nationality_score",
    "profession": "profession_score",
    "age": "age_score",
    "disability": "disability_score",
    "socioeconomic": "socioeconomic_score",
    "lgbtq": "lgbtq_score",
}


def compute_category_scores(db_path: str, judge_name: str, model_name: str) -> Dict[str, float]:
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """
        SELECT p.category, AVG(e.normalized_score) as avg_score
        FROM rubric_evaluations e
        LEFT JOIN prompts p ON p.prompt_id = e.prompt_id
        WHERE e.judge_name = ? AND e.model_name = ?
        GROUP BY p.category
        """,
        (judge_name, model_name),
    )
    rows = cur.fetchall()
    conn.close()

    out: Dict[str, float] = {}
    for r in rows:
        category = r["category"]
        col = CATEGORY_TO_COLUMN.get(category)
        if col:
            out[col] = float(r["avg_score"] or 0.0)
    return out


def compute_explicit_implicit_rates(db_path: str, judge_name: str, model_name: str) -> Dict[str, float]:
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """
        SELECT answers_json
        FROM rubric_evaluations
        WHERE judge_name = ? AND model_name = ?
        """,
        (judge_name, model_name),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return {"explicit_stereotype_rate": 0.0, "implicit_stereotype_rate": 0.0}

    explicit = 0
    implicit = 0
    for row in rows:
        answers = json.loads(row["answers_json"])
        explicit += 1 if int(answers.get("explicit_stereotype", 0)) == 1 else 0
        implicit += 1 if int(answers.get("implicit_assumption", 0)) == 1 else 0

    n = len(rows)
    return {
        "explicit_stereotype_rate": (explicit / n) * 100.0,
        "implicit_stereotype_rate": (implicit / n) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leaderboard snapshot from rubric evaluations")
    parser.add_argument("--db-path", default="stereowipe.db")
    parser.add_argument("--judge-name", default="gemini_rubric_v1")
    parser.add_argument("--snapshot-date", default=str(date.today()))
    parser.add_argument("--output-json", default="data/leaderboard_snapshot.json")
    args = parser.parse_args()

    db = Database(args.db_path)
    metrics = StereoWipeMetrics(db_path=args.db_path, judge_name=args.judge_name)

    # get models with rubric evaluations
    import sqlite3
    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT DISTINCT model_name FROM rubric_evaluations WHERE judge_name = ? ORDER BY model_name",
        (args.judge_name,),
    )
    models = [r["model_name"] for r in cur.fetchall()]
    conn.close()

    if not models:
        print("No rubric evaluations found. Run scripts/run_evaluations.py first.")
        return

    snapshot_rows: List[Dict] = []

    for model_name in models:
        summary = metrics.model_summary(model_name)
        category_scores = compute_category_scores(args.db_path, args.judge_name, model_name)
        rates = compute_explicit_implicit_rates(args.db_path, args.judge_name, model_name)

        scores = {
            "overall_score": summary["normalized_score"],
            "total_prompts_evaluated": summary["num_evaluations"],
            "cultural_sensitivity_score": summary["normalized_score"],  # placeholder until CSI lands
            **category_scores,
            **rates,
        }
        db.insert_leaderboard_snapshot(args.snapshot_date, model_name, scores)

        snapshot_rows.append(
            {
                "model_name": model_name,
                **summary,
                **scores,
            }
        )

    snapshot_rows.sort(key=lambda r: r.get("overall_score", 0.0), reverse=True)
    for idx, row in enumerate(snapshot_rows, start=1):
        row["rank"] = idx

    agreement = metrics.judge_human_agreement()
    inter_region = metrics.inter_region_agreement()
    regional_disagreement = metrics.regional_disagreement_rate()

    payload = {
        "snapshot_date": args.snapshot_date,
        "judge_name": args.judge_name,
        "models": snapshot_rows,
        "global_metrics": {
            "judge_human_agreement_spearman": agreement.correlation,
            "judge_human_agreement_p_value": agreement.p_value,
            "judge_human_agreement_n_models": agreement.n_models,
            "inter_region_agreement": inter_region,
            "regional_disagreement_rate": regional_disagreement,
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Generated leaderboard snapshot for {len(models)} models")
    print(f"Saved DB snapshot_date={args.snapshot_date} and JSON={out_path}")


if __name__ == "__main__":
    main()
