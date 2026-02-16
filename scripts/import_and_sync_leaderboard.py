#!/usr/bin/env python3
import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from biaswipe.database import Database
from biaswipe.judge import MockJudge
from biaswipe.scoring import score_model_responses

def main():
    db = Database()
    
    # 1. Import Expanded Prompts
    prompts_path = "sample_data/expanded_prompts.json"
    print(f"Loading prompts from {prompts_path}...")
    try:
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
        db.import_prompts_from_json(prompts)
        print(f"  Imported {len(prompts)} prompts.")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return

    # 2. Score and Import "Jules" model responses
    jules_path = "data/model_responses/Jules.json"
    print(f"Scoring responses from {jules_path}...")
    try:
        with open(jules_path, 'r') as f:
            jules_responses = json.load(f)
        
        judge = MockJudge("EnsembleMock")
        scores = score_model_responses(
            prompts=prompts,
            model_responses=jules_responses,
            category_weights={},
            judges=[judge],
            model_name="Jules",
            save_to_db=True
        )
        print(f"  Scored 'Jules' model. SR: {scores['SR']:.2f}")
    except Exception as e:
        print(f"Error processing Jules.json: {e}")

    # 3. Import responses_stereowipe.json (Nested format)
    # Format: { "Prompt Text": { "ModelName": "Response" } }
    text_to_id = {v['text']: k for k, v in prompts.items()}
    
    responses_path = "data/model_responses/responses_stereowipe.json"
    print(f"Processing {responses_path}...")
    try:
        with open(responses_path, 'r') as f:
            nested_data = json.load(f)
        
        model_data = {} # model_name -> {prompt_id: response}
        
        for text, models in nested_data.items():
            prompt_id = text_to_id.get(text)
            if not prompt_id: continue
            
            for model_name, response in models.items():
                if model_name not in model_data:
                    model_data[model_name] = {}
                model_data[model_name][prompt_id] = response

        for model_name, responses in model_data.items():
            print(f"Scoring model: {model_name}...")
            score_model_responses(
                prompts=prompts,
                model_responses=responses,
                category_weights={},
                judges=[judge],
                model_name=model_name,
                save_to_db=True
            )
    except Exception as e:
        print(f"Error processing responses_stereowipe.json: {e}")

    # 4. Generate Leaderboard Snapshot
    print("Generating leaderboard snapshot...")
    try:
        summary = db.get_evaluation_summary()
        today = datetime.now().strftime("%Y-%m-%d")
        
        for model_name, stats in summary['stereotype_stats'].items():
            avg_sev = stats['avg_severity'] or 0
            scores = {
                'overall_score': 100 - (avg_sev * 100),
                'explicit_stereotype_rate': stats['stereotype_rate'],
                'total_prompts_evaluated': summary['evaluations_by_model'].get(model_name, 0)
            }
            db.insert_leaderboard_snapshot(today, model_name, scores)
            print(f"  Snapshot created for {model_name}: {scores['overall_score']:.1f}")
    except Exception as e:
        print(f"Error generating snapshot: {e}")

    print("\nSync complete! The UI leaderboard should now reflect your test data.")

if __name__ == "__main__":
    main()