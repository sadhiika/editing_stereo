#!/usr/bin/env python3
"""
Daily Elo Rating Calculator for StereoWipe Arena

This script calculates Elo ratings for models based on arena battle results.
It should be run daily as a batch job to update the leaderboard.

Usage:
    python scripts/calculate_elo.py [--db-path PATH] [--k-factor K]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging

# Add parent directory to path to import biaswipe modules
sys.path.append(str(Path(__file__).parent.parent))
from biaswipe.database import Database


class EloCalculator:
    """Elo rating calculator for model arena battles."""
    
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        """Initialize Elo calculator.
        
        Args:
            k_factor: Elo K-factor (higher = more volatile ratings)
            initial_rating: Starting Elo rating for new models
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.logger = logging.getLogger(__name__)
    
    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for model A against model B.
        
        Args:
            rating_a: Current Elo rating of model A
            rating_b: Current Elo rating of model B
            
        Returns:
            Expected score for model A (0.0 to 1.0)
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, rating_a: float, rating_b: float, 
                      actual_score_a: float) -> Tuple[float, float]:
        """Update Elo ratings based on battle result.
        
        Args:
            rating_a: Current Elo rating of model A
            rating_b: Current Elo rating of model B
            actual_score_a: Actual score for model A (1.0 = win, 0.5 = tie, 0.0 = loss)
            
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        expected_a = self.calculate_expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        actual_b = 1.0 - actual_score_a
        
        new_rating_a = rating_a + self.k_factor * (actual_score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (actual_b - expected_b)
        
        return new_rating_a, new_rating_b
    
    def calculate_all_ratings(self, battles: List[Dict]) -> Dict[str, float]:
        """Calculate Elo ratings for all models based on battle history.
        
        Args:
            battles: List of battle dictionaries with keys:
                    - model_a, model_b: Model names
                    - winner: 'a', 'b', or 'tie'
                    - created_at: Battle timestamp
                    
        Returns:
            Dictionary mapping model names to their Elo ratings
        """
        # Initialize ratings for all models
        ratings = {}
        all_models = set()
        
        # Collect all model names
        for battle in battles:
            all_models.add(battle['model_a'])
            all_models.add(battle['model_b'])
        
        # Initialize all models with starting rating
        for model in all_models:
            ratings[model] = self.initial_rating
        
        # Sort battles by timestamp to process chronologically
        battles_sorted = sorted(battles, key=lambda x: x['created_at'])
        
        # Process each battle
        for battle in battles_sorted:
            model_a = battle['model_a']
            model_b = battle['model_b']
            winner = battle['winner']
            
            # Convert winner to actual score for model A
            if winner == 'a':
                actual_score_a = 1.0
            elif winner == 'b':
                actual_score_a = 0.0
            else:  # tie
                actual_score_a = 0.5
            
            # Update ratings
            new_rating_a, new_rating_b = self.update_ratings(
                ratings[model_a], ratings[model_b], actual_score_a
            )
            
            ratings[model_a] = new_rating_a
            ratings[model_b] = new_rating_b
            
            self.logger.debug(f"Battle: {model_a} vs {model_b}, winner: {winner}")
            self.logger.debug(f"Ratings: {model_a}: {new_rating_a:.1f}, {model_b}: {new_rating_b:.1f}")
        
        return ratings


def main():
    """Main function to calculate and update Elo ratings."""
    parser = argparse.ArgumentParser(description='Calculate Elo ratings for StereoWipe Arena')
    parser.add_argument('--db-path', type=str, default=None,
                      help='Path to SQLite database file')
    parser.add_argument('--k-factor', type=float, default=32.0,
                      help='Elo K-factor (default: 32.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true',
                      help='Calculate ratings but do not update database')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize database
    db = Database(args.db_path)
    
    try:
        # Get all arena battles
        logger.info("Fetching arena battles...")
        battles = db.get_arena_battles()
        
        if not battles:
            logger.info("No arena battles found. Nothing to calculate.")
            return
        
        logger.info(f"Found {len(battles)} arena battles")
        
        # Calculate Elo ratings
        calculator = EloCalculator(k_factor=args.k_factor)
        logger.info("Calculating Elo ratings...")
        ratings = calculator.calculate_all_ratings(battles)
        
        # Count battles per model
        battle_counts = {}
        for battle in battles:
            model_a = battle['model_a']
            model_b = battle['model_b']
            battle_counts[model_a] = battle_counts.get(model_a, 0) + 1
            battle_counts[model_b] = battle_counts.get(model_b, 0) + 1
        
        # Display results
        logger.info("Final Elo ratings:")
        sorted_models = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (model, rating) in enumerate(sorted_models, 1):
            battles_count = battle_counts.get(model, 0)
            logger.info(f"{rank}. {model}: {rating:.1f} ({battles_count} battles)")
        
        # Update database
        if not args.dry_run:
            logger.info("Updating database with new ratings...")
            for model, rating in ratings.items():
                battles_count = battle_counts.get(model, 0)
                db.update_elo_rating(model, rating, battles_count)
            
            logger.info("Database updated successfully")
        else:
            logger.info("Dry run - database not updated")
        
        # Summary statistics
        total_battles = len(battles)
        total_models = len(ratings)
        avg_rating = sum(ratings.values()) / len(ratings) if ratings else 0
        max_rating = max(ratings.values()) if ratings else 0
        min_rating = min(ratings.values()) if ratings else 0
        
        logger.info(f"Summary:")
        logger.info(f"  Total battles: {total_battles}")
        logger.info(f"  Total models: {total_models}")
        logger.info(f"  Average rating: {avg_rating:.1f}")
        logger.info(f"  Highest rating: {max_rating:.1f}")
        logger.info(f"  Lowest rating: {min_rating:.1f}")
        
    except Exception as e:
        logger.error(f"Error calculating Elo ratings: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()