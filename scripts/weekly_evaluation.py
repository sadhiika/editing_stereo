#!/usr/bin/env python3
"""
Weekly Batch Evaluation Script for StereoWipe Leaderboard

This script runs the full evaluation pipeline:
1. Loads prompts from database
2. Queries each model for responses
3. Evaluates responses using Gemini Flash judge
4. Calculates scores and updates leaderboard

Usage:
    python scripts/weekly_evaluation.py [--models MODEL1,MODEL2] [--dry-run] [--judge gemini]
    
Schedule with cron:
    0 0 * * 0 cd /path/to/stereowipe && python scripts/weekly_evaluation.py >> logs/weekly.log 2>&1
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from biaswipe.database import Database
from biaswipe.judge import GeminiJudge, MissingApiKeyError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/weekly_evaluation.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# ============ MODEL API CLIENTS ============

class ModelAPIClient:
    """Base class for model API clients."""
    
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
    
    def generate(self, prompt: str) -> str:
        """Generate a response from the model."""
        raise NotImplementedError


class OpenAIClient(ModelAPIClient):
    """OpenAI API client."""
    
    def __init__(self, model_name: str, api_key: str = None):
        super().__init__(model_name, api_key or os.environ.get("OPENAI_API_KEY"))
        if not self.api_key:
            raise MissingApiKeyError("OpenAI API key not found")
        
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
    
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content


class AnthropicClient(ModelAPIClient):
    """Anthropic API client."""
    
    def __init__(self, model_name: str, api_key: str = None):
        super().__init__(model_name, api_key or os.environ.get("ANTHROPIC_API_KEY"))
        if not self.api_key:
            raise MissingApiKeyError("Anthropic API key not found")
        
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class GeminiClient(ModelAPIClient):
    """Google Gemini API client."""
    
    def __init__(self, model_name: str, api_key: str = None):
        super().__init__(model_name, api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        if not self.api_key:
            raise MissingApiKeyError("Google/Gemini API key not found")
        
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model_name)
    
    def generate(self, prompt: str) -> str:
        response = self.client.generate_content(prompt)
        return response.text


class MockClient(ModelAPIClient):
    """Mock client for testing."""
    
    def generate(self, prompt: str) -> str:
        # Return varied mock responses
        mock_responses = [
            "Based on my analysis, I would describe this person as hardworking and dedicated.",
            "I don't think it's appropriate to generalize about any group of people.",
            "There are many stereotypes associated with this topic, but individuals vary greatly.",
            "I try to avoid making assumptions based on group membership.",
            "This is a complex topic that requires nuanced understanding."
        ]
        import hashlib
        idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(mock_responses)
        return mock_responses[idx]


def get_model_client(model_name: str, provider: str) -> Optional[ModelAPIClient]:
    """Get the appropriate API client for a model."""
    try:
        if provider == "OpenAI":
            # Map model names to actual OpenAI model IDs
            model_map = {
                "gpt-5.2": "gpt-4o",
                "gpt-5.1": "gpt-4o",
                "gpt-4.5-preview": "gpt-4o",
                "chatgpt-4o-latest": "gpt-4o"
            }
            actual_model = model_map.get(model_name, "gpt-4o")
            return OpenAIClient(actual_model)
        
        elif provider == "Anthropic":
            model_map = {
                "claude-opus-4-5": "claude-3-5-sonnet-latest",
                "claude-sonnet-4-5": "claude-3-5-sonnet-latest",
                "claude-opus-4-1": "claude-3-opus-20240229"
            }
            actual_model = model_map.get(model_name, "claude-3-5-sonnet-latest")
            return AnthropicClient(actual_model)
        
        elif provider == "Google":
            model_map = {
                "gemini-3-pro": "gemini-2.0-flash-exp",
                "gemini-3-flash": "gemini-2.0-flash-exp",
                "gemini-2.5-pro": "gemini-1.5-pro"
            }
            actual_model = model_map.get(model_name, "gemini-2.0-flash-exp")
            return GeminiClient(actual_model)
        
        else:
            # Use mock client for unsupported providers
            logger.warning(f"No API client for provider {provider}, using mock")
            return MockClient(model_name)
    
    except MissingApiKeyError as e:
        logger.warning(f"Missing API key for {model_name}: {e}")
        return None
    except ImportError as e:
        logger.warning(f"Missing library for {model_name}: {e}")
        return None


# ============ EVALUATION PIPELINE ============

def evaluate_model(
    db: Database,
    judge: GeminiJudge,
    model_name: str,
    provider: str,
    prompts: List[Dict],
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a single model on all prompts.
    
    Returns:
        Dictionary with evaluation results and scores
    """
    logger.info(f"Evaluating {model_name} ({provider}) on {len(prompts)} prompts")
    
    # Get model client
    client = get_model_client(model_name, provider)
    if not client:
        logger.error(f"Could not create client for {model_name}")
        return None
    
    results = {
        'model_name': model_name,
        'provider': provider,
        'evaluations': [],
        'category_scores': {},
        'region_scores': {},
        'overall': {
            'total': 0,
            'explicit_count': 0,
            'implicit_count': 0,
            'total_severity': 0.0
        }
    }
    
    for i, prompt in enumerate(prompts):
        prompt_id = prompt['prompt_id']
        prompt_text = prompt['prompt_text']
        category = prompt['category']
        region = prompt.get('region', 'global')
        
        logger.debug(f"  [{i+1}/{len(prompts)}] Processing: {prompt_id}")
        
        try:
            # Generate model response
            if dry_run:
                response_text = f"[DRY RUN] Mock response for {prompt_id}"
            else:
                response_text = client.generate(prompt_text)
                time.sleep(0.5)  # Rate limiting
            
            # Judge the response
            judgment = judge.judge_response(
                response_text=response_text,
                prompt_text=prompt_text,
                category=category,
                region=region
            )
            
            # Store detailed evaluation
            evaluation = {
                'prompt_id': prompt_id,
                'prompt_text': prompt_text,
                'prompt_category': category,
                'prompt_region': region,
                'model_name': model_name,
                'response_text': response_text,
                'judge_name': judge.name,
                'has_explicit_stereotype': judgment.get('has_explicit_stereotype', False),
                'has_implicit_stereotype': judgment.get('has_implicit_stereotype', False),
                'explicit_severity': judgment.get('explicit_severity', 0.0),
                'implicit_severity': judgment.get('implicit_severity', 0.0),
                'combined_severity': judgment.get('combined_severity', 0.0),
                'stereotype_type': judgment.get('stereotype_type', 'none'),
                'affected_group': judgment.get('affected_group'),
                'reasoning': judgment.get('reasoning', '')
            }
            
            if not dry_run:
                db.insert_detailed_evaluation(evaluation)
            
            results['evaluations'].append(evaluation)
            
            # Update category scores
            if category not in results['category_scores']:
                results['category_scores'][category] = {
                    'total': 0, 'stereotype_count': 0, 'total_severity': 0.0
                }
            results['category_scores'][category]['total'] += 1
            if evaluation['has_explicit_stereotype'] or evaluation['has_implicit_stereotype']:
                results['category_scores'][category]['stereotype_count'] += 1
            results['category_scores'][category]['total_severity'] += evaluation['combined_severity']
            
            # Update region scores
            if region:
                if region not in results['region_scores']:
                    results['region_scores'][region] = {
                        'total': 0, 'stereotype_count': 0, 'total_severity': 0.0
                    }
                results['region_scores'][region]['total'] += 1
                if evaluation['has_explicit_stereotype'] or evaluation['has_implicit_stereotype']:
                    results['region_scores'][region]['stereotype_count'] += 1
                results['region_scores'][region]['total_severity'] += evaluation['combined_severity']
            
            # Update overall
            results['overall']['total'] += 1
            if evaluation['has_explicit_stereotype']:
                results['overall']['explicit_count'] += 1
            if evaluation['has_implicit_stereotype']:
                results['overall']['implicit_count'] += 1
            results['overall']['total_severity'] += evaluation['combined_severity']
            
        except Exception as e:
            logger.error(f"  Error evaluating prompt {prompt_id}: {e}")
            continue
    
    return results


def calculate_scores(results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate final scores from evaluation results."""
    total = results['overall']['total']
    if total == 0:
        return {}
    
    # Calculate base scores (higher = less biased = better)
    explicit_rate = results['overall']['explicit_count'] / total
    implicit_rate = results['overall']['implicit_count'] / total
    avg_severity = results['overall']['total_severity'] / total
    
    # Overall score: 100 - (severity * 100)
    overall_score = max(0, min(100, 100 - (avg_severity * 100)))
    
    scores = {
        'overall_score': overall_score,
        'explicit_stereotype_rate': explicit_rate * 100,
        'implicit_stereotype_rate': implicit_rate * 100,
        'total_prompts_evaluated': total
    }
    
    # Category scores
    category_map = {
        'gender': 'gender_score',
        'race': 'race_score',
        'religion': 'religion_score',
        'nationality': 'nationality_score',
        'profession': 'profession_score',
        'age': 'age_score',
        'disability': 'disability_score',
        'socioeconomic': 'socioeconomic_score',
        'lgbtq': 'lgbtq_score'
    }
    
    for category, score_key in category_map.items():
        if category in results['category_scores']:
            cat_data = results['category_scores'][category]
            if cat_data['total'] > 0:
                cat_severity = cat_data['total_severity'] / cat_data['total']
                scores[score_key] = max(0, min(100, 100 - (cat_severity * 100)))
    
    # Cultural sensitivity score (average of non-Western regions vs Western)
    western_regions = ['north_america', 'western_europe']
    global_south_regions = ['latin_america', 'sub_saharan_africa', 'south_asia', 
                           'east_asia', 'southeast_asia', 'mena']
    
    western_scores = []
    global_south_scores = []
    
    for region, data in results['region_scores'].items():
        if data['total'] > 0:
            region_severity = data['total_severity'] / data['total']
            region_score = 100 - (region_severity * 100)
            
            if region in western_regions:
                western_scores.append(region_score)
            elif region in global_south_regions:
                global_south_scores.append(region_score)
    
    if global_south_scores:
        scores['cultural_sensitivity_score'] = sum(global_south_scores) / len(global_south_scores)
    
    return scores


def run_weekly_evaluation(
    db: Database,
    models: List[str] = None,
    dry_run: bool = False,
    judge_type: str = 'gemini'
):
    """
    Run the complete weekly evaluation pipeline.
    """
    logger.info("=" * 60)
    logger.info("Starting Weekly Evaluation Pipeline")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 60)
    
    # Initialize judge
    try:
        if judge_type == 'gemini':
            judge = GeminiJudge(model="gemini-2.0-flash-exp")
        else:
            from biaswipe.judge import MockJudge
            judge = MockJudge("WeeklyEvalMock")
        logger.info(f"Initialized judge: {judge.name}")
    except MissingApiKeyError as e:
        logger.error(f"Cannot initialize judge: {e}")
        return
    
    # Load prompts
    prompts = db.get_prompts(active_only=True)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Get models to evaluate
    all_models = db.get_active_models()
    if models:
        all_models = [m for m in all_models if m['model_name'] in models]
    
    logger.info(f"Evaluating {len(all_models)} models")
    
    # Evaluation results
    snapshot_date = datetime.now().strftime('%Y-%m-%d')
    all_results = []
    
    for model in all_models:
        model_name = model['model_name']
        provider = model['provider']
        
        try:
            results = evaluate_model(
                db=db,
                judge=judge,
                model_name=model_name,
                provider=provider,
                prompts=prompts,
                dry_run=dry_run
            )
            
            if results:
                scores = calculate_scores(results)
                all_results.append({
                    'model_name': model_name,
                    'provider': provider,
                    'scores': scores
                })
                
                # Save to leaderboard
                if not dry_run and scores:
                    db.insert_leaderboard_snapshot(snapshot_date, model_name, scores)
                    
                    # Save regional scores
                    for region, data in results['region_scores'].items():
                        if data['total'] > 0:
                            regional_scores = {
                                'overall_score': 100 - (data['total_severity'] / data['total'] * 100),
                                'stereotype_rate': data['stereotype_count'] / data['total'] * 100,
                                'severity_score': data['total_severity'] / data['total'],
                                'sample_count': data['total']
                            }
                            db.insert_regional_score(snapshot_date, model_name, region, regional_scores)
                    
                    db.update_model_last_evaluated(model_name)
                
                logger.info(f"  ✓ {model_name}: Overall score = {scores.get('overall_score', 0):.1f}")
        
        except Exception as e:
            logger.error(f"  ✗ {model_name}: Error - {e}")
            continue
    
    # Summary
    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info(f"Models evaluated: {len(all_results)}")
    
    if all_results:
        # Sort by overall score
        all_results.sort(key=lambda x: x['scores'].get('overall_score', 0), reverse=True)
        
        logger.info("\nTop 5 Models:")
        for i, result in enumerate(all_results[:5], 1):
            score = result['scores'].get('overall_score', 0)
            logger.info(f"  {i}. {result['model_name']}: {score:.1f}")
    
    logger.info("=" * 60)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Weekly StereoWipe Evaluation')
    parser.add_argument('--models', type=str, default=None,
                       help='Comma-separated list of model names to evaluate')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without making API calls or saving to database')
    parser.add_argument('--judge', type=str, default='mock',
                       choices=['gemini', 'mock'],
                       help='Judge to use for evaluation')
    parser.add_argument('--db-path', type=str, default=None,
                       help='Path to database file')
    
    args = parser.parse_args()
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Initialize database
    db = Database(args.db_path)
    
    # Parse models
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(',')]
    
    # Run evaluation
    run_weekly_evaluation(
        db=db,
        models=models,
        dry_run=args.dry_run,
        judge_type=args.judge
    )


if __name__ == '__main__':
    main()

