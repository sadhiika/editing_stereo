import sqlite3
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import os


class Database:
    """SQLite database handler for StereoWipe evaluation data."""
    
    def __init__(self, db_path: str = None):
        """Initialize database connection and create tables if needed.
        
        Args:
            db_path: Path to database file. Defaults to 'stereowipe.db' in project root.
        """
        if db_path is None:
            # Get project root (parent of biaswipe directory)
            project_root = Path(__file__).parent.parent
            self.db_path = str(project_root / "stereowipe.db")
        else:
            self.db_path = db_path
        
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_db() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_id TEXT NOT NULL,
                    prompt_text TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    judge_name TEXT NOT NULL,
                    is_stereotype BOOLEAN NOT NULL,
                    severity_score REAL,
                    reasoning TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS human_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    prompt_id TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    is_stereotype BOOLEAN NOT NULL,
                    severity_score REAL,
                    annotator_comments TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS arena_battles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    prompt_id TEXT NOT NULL,
                    model_a TEXT NOT NULL,
                    response_a TEXT NOT NULL,
                    model_b TEXT NOT NULL,
                    response_b TEXT NOT NULL,
                    winner TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Leaderboard snapshots for weekly rankings
                CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_date DATE NOT NULL,
                    model_name TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    gender_score REAL,
                    race_score REAL,
                    religion_score REAL,
                    nationality_score REAL,
                    profession_score REAL,
                    age_score REAL,
                    disability_score REAL,
                    socioeconomic_score REAL,
                    lgbtq_score REAL,
                    cultural_sensitivity_score REAL,
                    explicit_stereotype_rate REAL,
                    implicit_stereotype_rate REAL,
                    total_prompts_evaluated INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(snapshot_date, model_name)
                );
                
                -- Regional performance tracking
                CREATE TABLE IF NOT EXISTS regional_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_date DATE NOT NULL,
                    model_name TEXT NOT NULL,
                    region TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    stereotype_rate REAL NOT NULL,
                    severity_score REAL,
                    sample_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(snapshot_date, model_name, region)
                );
                
                -- Extended evaluations with explicit/implicit tracking
                CREATE TABLE IF NOT EXISTS detailed_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_id TEXT NOT NULL,
                    prompt_text TEXT NOT NULL,
                    prompt_category TEXT NOT NULL,
                    prompt_region TEXT,
                    model_name TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    judge_name TEXT NOT NULL,
                    has_explicit_stereotype BOOLEAN DEFAULT FALSE,
                    has_implicit_stereotype BOOLEAN DEFAULT FALSE,
                    explicit_severity REAL DEFAULT 0.0,
                    implicit_severity REAL DEFAULT 0.0,
                    combined_severity REAL DEFAULT 0.0,
                    stereotype_type TEXT,
                    affected_group TEXT,
                    reasoning TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Prompts dataset
                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_id TEXT UNIQUE NOT NULL,
                    prompt_text TEXT NOT NULL,
                    category TEXT NOT NULL,
                    region TEXT,
                    stereotype_type TEXT,
                    difficulty TEXT DEFAULT 'standard',
                    source TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Models registry
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT UNIQUE NOT NULL,
                    provider TEXT NOT NULL,
                    model_type TEXT DEFAULT 'proprietary',
                    api_endpoint TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_evaluated TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_evaluations_model_prompt 
                    ON evaluations(model_name, prompt_id);
                CREATE INDEX IF NOT EXISTS idx_evaluations_created 
                    ON evaluations(created_at);
                CREATE INDEX IF NOT EXISTS idx_human_annotations_session 
                    ON human_annotations(session_id);
                CREATE INDEX IF NOT EXISTS idx_arena_battles_session 
                    ON arena_battles(session_id);
                CREATE INDEX IF NOT EXISTS idx_leaderboard_date 
                    ON leaderboard_snapshots(snapshot_date);
                CREATE INDEX IF NOT EXISTS idx_regional_scores_date_model 
                    ON regional_scores(snapshot_date, model_name);
                CREATE INDEX IF NOT EXISTS idx_detailed_evals_model_category 
                    ON detailed_evaluations(model_name, prompt_category);
                CREATE INDEX IF NOT EXISTS idx_prompts_category_region 
                    ON prompts(category, region);
            """)
    
    @contextmanager
    def _get_db(self):
        """Context manager for database connections.
        
        Ensures connections are properly closed even if errors occur.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
    
    def insert_evaluation(self, prompt_id: str, prompt_text: str, model_name: str,
                         response_text: str, judge_name: str, is_stereotype: bool,
                         severity_score: Optional[float] = None, 
                         reasoning: Optional[str] = None) -> int:
        """Insert a single evaluation result.
        
        Returns:
            The ID of the inserted evaluation.
        """
        with self._get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO evaluations 
                (prompt_id, prompt_text, model_name, response_text, judge_name,
                 is_stereotype, severity_score, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (prompt_id, prompt_text, model_name, response_text, judge_name,
                  is_stereotype, severity_score, reasoning))
            return cursor.lastrowid
    
    def insert_evaluation_batch(self, evaluations: List[Dict[str, Any]]) -> int:
        """Insert multiple evaluation results in a single transaction.
        
        Args:
            evaluations: List of dictionaries containing evaluation data
            
        Returns:
            Number of evaluations inserted.
        """
        with self._get_db() as conn:
            conn.executemany("""
                INSERT INTO evaluations 
                (prompt_id, prompt_text, model_name, response_text, judge_name,
                 is_stereotype, severity_score, reasoning)
                VALUES (:prompt_id, :prompt_text, :model_name, :response_text, 
                        :judge_name, :is_stereotype, :severity_score, :reasoning)
            """, evaluations)
            return len(evaluations)
    
    def get_evaluations_by_model(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all evaluations for a specific model."""
        with self._get_db() as conn:
            cursor = conn.execute("""
                SELECT * FROM evaluations 
                WHERE model_name = ? 
                ORDER BY created_at DESC
            """, (model_name,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_evaluations(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all evaluations, optionally limited."""
        with self._get_db() as conn:
            if limit:
                cursor = conn.execute("""
                    SELECT * FROM evaluations 
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            else:
                cursor = conn.execute("""
                    SELECT * FROM evaluations 
                    ORDER BY created_at DESC
                """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_unannotated_responses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get responses that haven't been human-annotated yet.
        
        Returns responses from evaluations that don't have corresponding
        human annotations.
        """
        with self._get_db() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT e.prompt_id, e.prompt_text, e.response_text
                FROM evaluations e
                LEFT JOIN human_annotations h 
                    ON e.prompt_id = h.prompt_id 
                    AND e.response_text = h.response_text
                WHERE h.id IS NULL
                ORDER BY RANDOM()
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_next_unannotated_response(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the next unannotated response for a session using round-robin assignment.
        
        This method implements round-robin assignment to ensure all responses get
        annotated by multiple people while distributing work fairly.
        """
        with self._get_db() as conn:
            # Get responses that haven't been annotated by this session
            cursor = conn.execute("""
                SELECT DISTINCT e.prompt_id, e.prompt_text, e.response_text
                FROM evaluations e
                LEFT JOIN human_annotations h 
                    ON e.prompt_id = h.prompt_id 
                    AND e.response_text = h.response_text
                    AND h.session_id = ?
                WHERE h.id IS NULL
                ORDER BY RANDOM()
                LIMIT 1
            """, (session_id,))
            
            result = cursor.fetchone()
            return dict(result) if result else None
    
    def get_session_annotation_count(self, session_id: str) -> int:
        """Get the number of annotations completed by a session."""
        with self._get_db() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as count
                FROM human_annotations
                WHERE session_id = ?
            """, (session_id,))
            return cursor.fetchone()['count']
    
    def get_total_annotation_count(self) -> int:
        """Get the total number of human annotations in the database."""
        with self._get_db() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM human_annotations")
            return cursor.fetchone()['count']
    
    def insert_human_annotation(self, session_id: str, prompt_id: str,
                               response_text: str, is_stereotype: bool,
                               severity_score: Optional[float] = None,
                               annotator_comments: Optional[str] = None) -> int:
        """Insert a human annotation."""
        with self._get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO human_annotations
                (session_id, prompt_id, response_text, is_stereotype,
                 severity_score, annotator_comments)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, prompt_id, response_text, is_stereotype,
                  severity_score, annotator_comments))
            return cursor.lastrowid
    
    def get_human_annotations(self, prompt_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get human annotations, optionally filtered by prompt_id."""
        with self._get_db() as conn:
            if prompt_id:
                cursor = conn.execute("""
                    SELECT * FROM human_annotations
                    WHERE prompt_id = ?
                    ORDER BY created_at DESC
                """, (prompt_id,))
            else:
                cursor = conn.execute("""
                    SELECT * FROM human_annotations
                    ORDER BY created_at DESC
                """)
            return [dict(row) for row in cursor.fetchall()]
    
    def insert_arena_battle(self, session_id: str, prompt_id: str,
                           model_a: str, response_a: str,
                           model_b: str, response_b: str,
                           winner: str) -> int:
        """Insert an arena battle result.
        
        Args:
            winner: Must be 'a', 'b', or 'tie'
        """
        if winner not in ['a', 'b', 'tie']:
            raise ValueError("Winner must be 'a', 'b', or 'tie'")
            
        with self._get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO arena_battles
                (session_id, prompt_id, model_a, response_a,
                 model_b, response_b, winner)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_id, prompt_id, model_a, response_a,
                  model_b, response_b, winner))
            return cursor.lastrowid
    
    def get_arena_battles(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get arena battles, optionally filtered by model."""
        with self._get_db() as conn:
            if model_name:
                cursor = conn.execute("""
                    SELECT * FROM arena_battles
                    WHERE model_a = ? OR model_b = ?
                    ORDER BY created_at DESC
                """, (model_name, model_name))
            else:
                cursor = conn.execute("""
                    SELECT * FROM arena_battles
                    ORDER BY created_at DESC
                """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_model_win_stats(self) -> Dict[str, Dict[str, int]]:
        """Calculate win/loss/tie statistics for all models in arena battles."""
        stats = {}
        
        with self._get_db() as conn:
            # Get all unique models
            cursor = conn.execute("""
                SELECT DISTINCT model_a as model FROM arena_battles
                UNION
                SELECT DISTINCT model_b as model FROM arena_battles
            """)
            models = [row['model'] for row in cursor.fetchall()]
            
            for model in models:
                # Count wins when model is A
                cursor = conn.execute("""
                    SELECT COUNT(*) as wins FROM arena_battles
                    WHERE model_a = ? AND winner = 'a'
                """, (model,))
                wins_as_a = cursor.fetchone()['wins']
                
                # Count wins when model is B
                cursor = conn.execute("""
                    SELECT COUNT(*) as wins FROM arena_battles
                    WHERE model_b = ? AND winner = 'b'
                """, (model,))
                wins_as_b = cursor.fetchone()['wins']
                
                # Count losses
                cursor = conn.execute("""
                    SELECT COUNT(*) as losses FROM arena_battles
                    WHERE (model_a = ? AND winner = 'b')
                       OR (model_b = ? AND winner = 'a')
                """, (model, model))
                losses = cursor.fetchone()['losses']
                
                # Count ties
                cursor = conn.execute("""
                    SELECT COUNT(*) as ties FROM arena_battles
                    WHERE (model_a = ? OR model_b = ?)
                      AND winner = 'tie'
                """, (model, model))
                ties = cursor.fetchone()['ties']
                
                stats[model] = {
                    'wins': wins_as_a + wins_as_b,
                    'losses': losses,
                    'ties': ties,
                    'total': wins_as_a + wins_as_b + losses + ties
                }
        
        return stats
    
    def export_to_csv(self, table_name: str, output_path: str):
        """Export a table to CSV format."""
        import csv
        
        with self._get_db() as conn:
            cursor = conn.execute(f"SELECT * FROM {table_name}")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
    
    def export_to_json(self, table_name: str, output_path: str):
        """Export a table to JSON format."""
        with self._get_db() as conn:
            cursor = conn.execute(f"SELECT * FROM {table_name}")
            rows = [dict(row) for row in cursor.fetchall()]
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2, default=str)
    
    def get_random_battle_data(self) -> Optional[Dict[str, Any]]:
        """Get random prompt and two different model responses for arena battle.
        
        Returns:
            Dictionary with prompt and two model responses, or None if insufficient data.
        """
        with self._get_db() as conn:
            # Get a random prompt that has at least 2 different model responses
            cursor = conn.execute("""
                SELECT prompt_id, prompt_text, COUNT(DISTINCT model_name) as model_count
                FROM evaluations
                GROUP BY prompt_id, prompt_text
                HAVING model_count >= 2
                ORDER BY RANDOM()
                LIMIT 1
            """)
            
            prompt_data = cursor.fetchone()
            if not prompt_data:
                return None
            
            prompt_id = prompt_data['prompt_id']
            prompt_text = prompt_data['prompt_text']
            
            # Get two different model responses for this prompt
            cursor = conn.execute("""
                SELECT model_name, response_text
                FROM evaluations
                WHERE prompt_id = ?
                ORDER BY RANDOM()
                LIMIT 2
            """, (prompt_id,))
            
            responses = cursor.fetchall()
            if len(responses) < 2:
                return None
            
            return {
                'prompt_id': prompt_id,
                'prompt_text': prompt_text,
                'model_a': responses[0]['model_name'],
                'response_a': responses[0]['response_text'],
                'model_b': responses[1]['model_name'],
                'response_b': responses[1]['response_text']
            }
    
    def get_model_elo_ratings(self) -> Dict[str, float]:
        """Get current Elo ratings for all models.
        
        Returns:
            Dictionary mapping model names to their Elo ratings.
        """
        with self._get_db() as conn:
            # Check if elo_ratings table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='elo_ratings'
            """)
            
            if not cursor.fetchone():
                # Create elo_ratings table if it doesn't exist
                conn.execute("""
                    CREATE TABLE elo_ratings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT UNIQUE NOT NULL,
                        elo_rating REAL NOT NULL DEFAULT 1500.0,
                        battles_count INTEGER NOT NULL DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                return {}
            
            # Get all current Elo ratings
            cursor = conn.execute("""
                SELECT model_name, elo_rating
                FROM elo_ratings
                ORDER BY elo_rating DESC
            """)
            
            return {row['model_name']: row['elo_rating'] for row in cursor.fetchall()}
    
    def update_elo_rating(self, model_name: str, new_rating: float, battles_count: int = None):
        """Update Elo rating for a model.
        
        Args:
            model_name: Name of the model
            new_rating: New Elo rating
            battles_count: Optional battle count to update
        """
        with self._get_db() as conn:
            # Check if elo_ratings table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='elo_ratings'
            """)
            
            if not cursor.fetchone():
                # Create elo_ratings table if it doesn't exist
                conn.execute("""
                    CREATE TABLE elo_ratings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT UNIQUE NOT NULL,
                        elo_rating REAL NOT NULL DEFAULT 1500.0,
                        battles_count INTEGER NOT NULL DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
            
            if battles_count is not None:
                conn.execute("""
                    INSERT OR REPLACE INTO elo_ratings 
                    (model_name, elo_rating, battles_count, last_updated)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (model_name, new_rating, battles_count))
            else:
                conn.execute("""
                    INSERT OR REPLACE INTO elo_ratings 
                    (model_name, elo_rating, last_updated)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (model_name, new_rating))
    
    def get_leaderboard_data(self) -> List[Dict[str, Any]]:
        """Get leaderboard data with Elo ratings and battle statistics.
        
        Returns:
            List of dictionaries containing model rankings and stats.
        """
        with self._get_db() as conn:
            # Check if elo_ratings table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='elo_ratings'
            """)
            
            if not cursor.fetchone():
                return []
            
            # Get leaderboard data with battle statistics
            cursor = conn.execute("""
                SELECT 
                    e.model_name,
                    e.elo_rating,
                    e.battles_count,
                    e.last_updated,
                    COALESCE(w.wins, 0) as wins,
                    COALESCE(l.losses, 0) as losses,
                    COALESCE(t.ties, 0) as ties
                FROM elo_ratings e
                LEFT JOIN (
                    SELECT model_name, COUNT(*) as wins
                    FROM (
                        SELECT model_a as model_name FROM arena_battles WHERE winner = 'a'
                        UNION ALL
                        SELECT model_b as model_name FROM arena_battles WHERE winner = 'b'
                    ) 
                    GROUP BY model_name
                ) w ON e.model_name = w.model_name
                LEFT JOIN (
                    SELECT model_name, COUNT(*) as losses
                    FROM (
                        SELECT model_a as model_name FROM arena_battles WHERE winner = 'b'
                        UNION ALL
                        SELECT model_b as model_name FROM arena_battles WHERE winner = 'a'
                    ) 
                    GROUP BY model_name
                ) l ON e.model_name = l.model_name
                LEFT JOIN (
                    SELECT model_name, COUNT(*) as ties
                    FROM (
                        SELECT model_a as model_name FROM arena_battles WHERE winner = 'tie'
                        UNION ALL
                        SELECT model_b as model_name FROM arena_battles WHERE winner = 'tie'
                    ) 
                    GROUP BY model_name
                ) t ON e.model_name = t.model_name
                ORDER BY e.elo_rating DESC
            """)
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all evaluations."""
        with self._get_db() as conn:
            # Total evaluations
            cursor = conn.execute("SELECT COUNT(*) as total FROM evaluations")
            total = cursor.fetchone()['total']
            
            # Evaluations by model
            cursor = conn.execute("""
                SELECT model_name, COUNT(*) as count
                FROM evaluations
                GROUP BY model_name
            """)
            by_model = {row['model_name']: row['count'] for row in cursor.fetchall()}
            
            # Stereotype rate by model
            cursor = conn.execute("""
                SELECT model_name, 
                       SUM(CASE WHEN is_stereotype THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as stereotype_rate,
                       AVG(CASE WHEN is_stereotype THEN severity_score ELSE NULL END) as avg_severity
                FROM evaluations
                GROUP BY model_name
            """)
            stereotype_stats = {
                row['model_name']: {
                    'stereotype_rate': row['stereotype_rate'],
                    'avg_severity': row['avg_severity']
                }
                for row in cursor.fetchall()
            }
            
            return {
                'total_evaluations': total,
                'evaluations_by_model': by_model,
                'stereotype_stats': stereotype_stats
            }

    # ========== Leaderboard Methods ==========
    
    def insert_leaderboard_snapshot(self, snapshot_date: str, model_name: str, 
                                    scores: Dict[str, float]) -> int:
        """Insert or update a leaderboard snapshot for a model.
        
        Args:
            snapshot_date: Date string in YYYY-MM-DD format
            model_name: Name of the model
            scores: Dictionary with score fields (overall_score, gender_score, etc.)
        """
        with self._get_db() as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO leaderboard_snapshots 
                (snapshot_date, model_name, overall_score, gender_score, race_score,
                 religion_score, nationality_score, profession_score, age_score,
                 disability_score, socioeconomic_score, lgbtq_score,
                 cultural_sensitivity_score, explicit_stereotype_rate, 
                 implicit_stereotype_rate, total_prompts_evaluated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_date, model_name,
                scores.get('overall_score', 0),
                scores.get('gender_score'),
                scores.get('race_score'),
                scores.get('religion_score'),
                scores.get('nationality_score'),
                scores.get('profession_score'),
                scores.get('age_score'),
                scores.get('disability_score'),
                scores.get('socioeconomic_score'),
                scores.get('lgbtq_score'),
                scores.get('cultural_sensitivity_score'),
                scores.get('explicit_stereotype_rate'),
                scores.get('implicit_stereotype_rate'),
                scores.get('total_prompts_evaluated', 0)
            ))
            return cursor.lastrowid
    
    def get_latest_leaderboard(self) -> List[Dict[str, Any]]:
        """Get the most recent leaderboard snapshot with rankings."""
        with self._get_db() as conn:
            # Get the latest snapshot date
            cursor = conn.execute("""
                SELECT MAX(snapshot_date) as latest_date 
                FROM leaderboard_snapshots
            """)
            result = cursor.fetchone()
            if not result or not result['latest_date']:
                return []
            
            latest_date = result['latest_date']
            
            # Get all models from the latest snapshot, ranked by overall score
            cursor = conn.execute("""
                SELECT 
                    ROW_NUMBER() OVER (ORDER BY overall_score DESC) as rank,
                    model_name,
                    overall_score,
                    gender_score,
                    race_score,
                    religion_score,
                    nationality_score,
                    profession_score,
                    age_score,
                    disability_score,
                    socioeconomic_score,
                    lgbtq_score,
                    cultural_sensitivity_score,
                    explicit_stereotype_rate,
                    implicit_stereotype_rate,
                    total_prompts_evaluated,
                    snapshot_date
                FROM leaderboard_snapshots
                WHERE snapshot_date = ?
                ORDER BY overall_score DESC
            """, (latest_date,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_leaderboard_by_date(self, snapshot_date: str) -> List[Dict[str, Any]]:
        """Get leaderboard snapshot for a specific date."""
        with self._get_db() as conn:
            cursor = conn.execute("""
                SELECT 
                    ROW_NUMBER() OVER (ORDER BY overall_score DESC) as rank,
                    model_name,
                    overall_score,
                    gender_score,
                    race_score,
                    religion_score,
                    nationality_score,
                    profession_score,
                    age_score,
                    disability_score,
                    socioeconomic_score,
                    lgbtq_score,
                    cultural_sensitivity_score,
                    explicit_stereotype_rate,
                    implicit_stereotype_rate,
                    total_prompts_evaluated,
                    snapshot_date
                FROM leaderboard_snapshots
                WHERE snapshot_date = ?
                ORDER BY overall_score DESC
            """, (snapshot_date,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_category_rankings(self, category: str) -> List[Dict[str, Any]]:
        """Get rankings for a specific category (e.g., 'gender', 'race')."""
        score_column = f"{category}_score"
        
        with self._get_db() as conn:
            # Get the latest snapshot date
            cursor = conn.execute("""
                SELECT MAX(snapshot_date) as latest_date 
                FROM leaderboard_snapshots
            """)
            result = cursor.fetchone()
            if not result or not result['latest_date']:
                return []
            
            latest_date = result['latest_date']
            
            cursor = conn.execute(f"""
                SELECT 
                    ROW_NUMBER() OVER (ORDER BY {score_column} DESC NULLS LAST) as rank,
                    model_name,
                    {score_column} as score,
                    overall_score,
                    snapshot_date
                FROM leaderboard_snapshots
                WHERE snapshot_date = ? AND {score_column} IS NOT NULL
                ORDER BY {score_column} DESC
            """, (latest_date,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def insert_regional_score(self, snapshot_date: str, model_name: str,
                              region: str, scores: Dict[str, float]) -> int:
        """Insert regional performance score."""
        with self._get_db() as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO regional_scores 
                (snapshot_date, model_name, region, overall_score, 
                 stereotype_rate, severity_score, sample_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_date, model_name, region,
                scores.get('overall_score', 0),
                scores.get('stereotype_rate', 0),
                scores.get('severity_score', 0),
                scores.get('sample_count', 0)
            ))
            return cursor.lastrowid
    
    def get_regional_comparison(self, model_name: str = None) -> List[Dict[str, Any]]:
        """Get regional scores for models, optionally filtered by model."""
        with self._get_db() as conn:
            # Get the latest snapshot date
            cursor = conn.execute("""
                SELECT MAX(snapshot_date) as latest_date 
                FROM regional_scores
            """)
            result = cursor.fetchone()
            if not result or not result['latest_date']:
                return []
            
            latest_date = result['latest_date']
            
            if model_name:
                cursor = conn.execute("""
                    SELECT * FROM regional_scores
                    WHERE snapshot_date = ? AND model_name = ?
                    ORDER BY region
                """, (latest_date, model_name))
            else:
                cursor = conn.execute("""
                    SELECT * FROM regional_scores
                    WHERE snapshot_date = ?
                    ORDER BY model_name, region
                """, (latest_date,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def insert_detailed_evaluation(self, evaluation: Dict[str, Any]) -> int:
        """Insert a detailed evaluation with explicit/implicit tracking."""
        with self._get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO detailed_evaluations 
                (prompt_id, prompt_text, prompt_category, prompt_region, model_name,
                 response_text, judge_name, has_explicit_stereotype, has_implicit_stereotype,
                 explicit_severity, implicit_severity, combined_severity, stereotype_type,
                 affected_group, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation['prompt_id'],
                evaluation['prompt_text'],
                evaluation['prompt_category'],
                evaluation.get('prompt_region'),
                evaluation['model_name'],
                evaluation['response_text'],
                evaluation['judge_name'],
                evaluation.get('has_explicit_stereotype', False),
                evaluation.get('has_implicit_stereotype', False),
                evaluation.get('explicit_severity', 0.0),
                evaluation.get('implicit_severity', 0.0),
                evaluation.get('combined_severity', 0.0),
                evaluation.get('stereotype_type'),
                evaluation.get('affected_group'),
                evaluation.get('reasoning')
            ))
            return cursor.lastrowid
    
    def get_model_detailed_stats(self, model_name: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific model."""
        with self._get_db() as conn:
            # Overall stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_evaluations,
                    SUM(CASE WHEN has_explicit_stereotype THEN 1 ELSE 0 END) as explicit_count,
                    SUM(CASE WHEN has_implicit_stereotype THEN 1 ELSE 0 END) as implicit_count,
                    AVG(combined_severity) as avg_severity,
                    AVG(explicit_severity) as avg_explicit_severity,
                    AVG(implicit_severity) as avg_implicit_severity
                FROM detailed_evaluations
                WHERE model_name = ?
            """, (model_name,))
            overall = dict(cursor.fetchone())
            
            # By category
            cursor = conn.execute("""
                SELECT 
                    prompt_category,
                    COUNT(*) as count,
                    SUM(CASE WHEN has_explicit_stereotype OR has_implicit_stereotype THEN 1 ELSE 0 END) as stereotype_count,
                    AVG(combined_severity) as avg_severity
                FROM detailed_evaluations
                WHERE model_name = ?
                GROUP BY prompt_category
            """, (model_name,))
            by_category = {row['prompt_category']: dict(row) for row in cursor.fetchall()}
            
            # By region
            cursor = conn.execute("""
                SELECT 
                    prompt_region,
                    COUNT(*) as count,
                    SUM(CASE WHEN has_explicit_stereotype OR has_implicit_stereotype THEN 1 ELSE 0 END) as stereotype_count,
                    AVG(combined_severity) as avg_severity
                FROM detailed_evaluations
                WHERE model_name = ? AND prompt_region IS NOT NULL
                GROUP BY prompt_region
            """, (model_name,))
            by_region = {row['prompt_region']: dict(row) for row in cursor.fetchall()}
            
            return {
                'overall': overall,
                'by_category': by_category,
                'by_region': by_region
            }
    
    # ========== Prompts Management ==========
    
    def insert_prompt(self, prompt_id: str, prompt_text: str, category: str,
                      region: str = None, stereotype_type: str = None,
                      difficulty: str = 'standard', source: str = None) -> int:
        """Insert a new prompt into the dataset."""
        with self._get_db() as conn:
            cursor = conn.execute("""
                INSERT OR IGNORE INTO prompts 
                (prompt_id, prompt_text, category, region, stereotype_type, difficulty, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (prompt_id, prompt_text, category, region, stereotype_type, difficulty, source))
            return cursor.lastrowid
    
    def get_prompts(self, category: str = None, region: str = None, 
                    active_only: bool = True) -> List[Dict[str, Any]]:
        """Get prompts, optionally filtered by category and region."""
        with self._get_db() as conn:
            query = "SELECT * FROM prompts WHERE 1=1"
            params = []
            
            if active_only:
                query += " AND is_active = 1"
            if category:
                query += " AND category = ?"
                params.append(category)
            if region:
                query += " AND region = ?"
                params.append(region)
            
            query += " ORDER BY category, region, prompt_id"
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_prompt_stats(self) -> Dict[str, Any]:
        """Get statistics about the prompt dataset."""
        with self._get_db() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT category) as categories,
                    COUNT(DISTINCT region) as regions
                FROM prompts WHERE is_active = 1
            """)
            totals = dict(cursor.fetchone())
            
            cursor = conn.execute("""
                SELECT category, COUNT(*) as count
                FROM prompts WHERE is_active = 1
                GROUP BY category
            """)
            by_category = {row['category']: row['count'] for row in cursor.fetchall()}
            
            cursor = conn.execute("""
                SELECT region, COUNT(*) as count
                FROM prompts WHERE is_active = 1 AND region IS NOT NULL
                GROUP BY region
            """)
            by_region = {row['region']: row['count'] for row in cursor.fetchall()}
            
            return {
                'total': totals['total'],
                'categories': totals['categories'],
                'regions': totals['regions'],
                'by_category': by_category,
                'by_region': by_region
            }
    
    # ========== Models Registry ==========
    
    def insert_model(self, model_name: str, provider: str, 
                     model_type: str = 'proprietary', api_endpoint: str = None) -> int:
        """Register a new model."""
        with self._get_db() as conn:
            cursor = conn.execute("""
                INSERT OR IGNORE INTO models 
                (model_name, provider, model_type, api_endpoint)
                VALUES (?, ?, ?, ?)
            """, (model_name, provider, model_type, api_endpoint))
            return cursor.lastrowid
    
    def get_active_models(self) -> List[Dict[str, Any]]:
        """Get all active models."""
        with self._get_db() as conn:
            cursor = conn.execute("""
                SELECT * FROM models WHERE is_active = 1
                ORDER BY provider, model_name
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def update_model_last_evaluated(self, model_name: str):
        """Update the last evaluation timestamp for a model."""
        with self._get_db() as conn:
            conn.execute("""
                UPDATE models SET last_evaluated = CURRENT_TIMESTAMP
                WHERE model_name = ?
            """, (model_name,))

    def import_prompts_from_json(self, prompts_dict: Dict[str, Any]):
        """Import prompts from a dictionary (id: {text, category, ...}) into the prompts table."""
        with self._get_db() as conn:
            for prompt_id, data in prompts_dict.items():
                conn.execute("""
                    INSERT OR REPLACE INTO prompts 
                    (prompt_id, prompt_text, category, region, stereotype_type)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    prompt_id, 
                    data.get('text', ''), 
                    data.get('category', 'general'),
                    data.get('region'),
                    data.get('stereotype_type')
                ))

    def import_evaluations_from_json(self, evaluations: List[Dict[str, Any]]):
        """Import evaluations into the evaluations table."""
        self.insert_evaluation_batch(evaluations)