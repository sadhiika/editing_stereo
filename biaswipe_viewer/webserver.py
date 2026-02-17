from flask import Flask, render_template, jsonify, request, session
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from biaswipe.database import Database

app = Flask(__name__)

# Secret key for session management (use env var in production)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'stereowipe-dev-secret-key-change-in-prod')

# Initialize database - use path relative to project root
project_root = Path(__file__).parent.parent
db_path = str(project_root / "stereowipe.db")
db = Database(db_path)

@app.route('/')
def home():
    """
    Displays the home page with project overview.
    """
    return render_template('index.html', active_page='home')

@app.route('/report')
def view_report():
    # Construct the path to report.json, assuming it's in the parent directory of biaswipe_viewer
    # os.path.dirname(__file__) gives the directory of webserver.py (i.e., biaswipe_viewer)
    # os.path.join( ... , '..', 'report.json') goes one level up to the project root
    report_path = os.path.join(os.path.dirname(__file__), '..', 'report.json')

    report_data = None
    error_message = None

    try:
        with open(report_path, 'r') as f:
            report_data = json.load(f)
    except FileNotFoundError:
        error_message = f"Error: Report file not found at '{os.path.abspath(report_path)}'. Please generate it first using the BiasWipe CLI."
    except json.JSONDecodeError:
        error_message = f"Error: Could not decode JSON from '{os.path.abspath(report_path)}'. The file might be corrupted or not valid JSON."
    except Exception as e:
        error_message = f"An unexpected error occurred while trying to load the report: {str(e)}"

    return render_template('report_display.html', report_data=report_data, error_message=error_message, active_page='report')

@app.route('/documentation')
def documentation():
    """
    Displays the documentation page explaining the BiasWipe system.
    """
    return render_template('documentation.html', active_page='documentation')

@app.route('/resources')
def resources():
    """
    Displays the resources page with research papers and datasets.
    """
    return render_template('resources.html', active_page='resources')

@app.route('/faq')
def faq():
    """
    Displays the FAQ page with frequently asked questions.
    """
    return render_template('faq.html', active_page='faq')

@app.route('/about')
def about():
    """
    Displays the About Us page with team and mission information.
    """
    return render_template('about.html', active_page='about')

@app.route('/volunteer')
def volunteer():
    """
    Displays the Volunteer Researchers page for recruitment.
    """
    return render_template('volunteer.html', active_page='volunteer')

@app.route('/contact')
def contact():
    """
    Displays the Contact Us page.
    """
    return render_template('contact.html', active_page='contact')

@app.route('/terms')
def terms():
    """
    Displays the Terms of Service page.
    """
    return render_template('terms.html', active_page='terms')


# ========== LEADERBOARD ROUTES ==========

@app.route('/leaderboard')
def leaderboard():
    """
    Displays the stereotyping leaderboard with model rankings.
    """
    try:
        leaderboard_data = db.get_latest_leaderboard()
        
        # Get snapshot date from the first entry if available
        snapshot_date = None
        if leaderboard_data and len(leaderboard_data) > 0:
            snapshot_date = leaderboard_data[0].get('snapshot_date')
        
        # Add provider info from models table
        models = {m['model_name']: m for m in db.get_active_models()}
        for entry in leaderboard_data:
            model_info = models.get(entry['model_name'], {})
            entry['provider'] = model_info.get('provider', 'Unknown')
            entry['model_type'] = model_info.get('model_type', 'unknown')
        
        return render_template('leaderboard_v2.html', 
                             leaderboard_data=leaderboard_data,
                             snapshot_date=snapshot_date,
                             active_page='leaderboard')
    except Exception as e:
        print(f"Error loading leaderboard: {e}")
        return render_template('leaderboard_v2.html', 
                             leaderboard_data=[],
                             snapshot_date=None,
                             active_page='leaderboard')


@app.route('/leaderboard/category/<category>')
def leaderboard_category(category):
    """
    Displays rankings for a specific category.
    """
    valid_categories = ['gender', 'race', 'religion', 'nationality', 
                       'profession', 'age', 'disability', 'socioeconomic', 'lgbtq']
    
    if category not in valid_categories:
        return jsonify({'error': f'Invalid category. Valid options: {valid_categories}'}), 400
    
    try:
        rankings = db.get_category_rankings(category)
        return render_template('leaderboard_v2.html',
                             leaderboard_data=rankings,
                             category=category,
                             snapshot_date=rankings[0]['snapshot_date'] if rankings else None,
                             active_page='leaderboard')
    except Exception as e:
        print(f"Error loading category leaderboard: {e}")
        return render_template('leaderboard_v2.html', leaderboard_data=[], category=category, active_page='leaderboard')


@app.route('/leaderboard/model/<model_name>')
def model_profile(model_name):
    """
    Displays detailed stats for a specific model.
    """
    try:
        stats = db.get_model_detailed_stats(model_name)
        regional_scores = db.get_regional_comparison(model_name)
        
        return render_template('model_profile.html',
                             model_name=model_name,
                             stats=stats,
                             regional_scores=regional_scores)
    except Exception as e:
        print(f"Error loading model profile: {e}")
        return render_template('model_profile.html',
                             model_name=model_name,
                             stats={},
                             regional_scores=[])


# ========== API ENDPOINTS ==========

@app.route('/api/leaderboard.json')
def api_leaderboard():
    """
    JSON API for leaderboard data - for media embedding.
    """
    try:
        leaderboard_data = db.get_latest_leaderboard()
        
        # Add provider info
        models = {m['model_name']: m for m in db.get_active_models()}
        for entry in leaderboard_data:
            model_info = models.get(entry['model_name'], {})
            entry['provider'] = model_info.get('provider', 'Unknown')
            entry['model_type'] = model_info.get('model_type', 'unknown')
        
        return jsonify({
            'success': True,
            'data': leaderboard_data,
            'metadata': {
                'total_models': len(leaderboard_data),
                'snapshot_date': leaderboard_data[0]['snapshot_date'] if leaderboard_data else None,
                'categories': ['overall', 'gender', 'race', 'religion', 'nationality', 
                              'profession', 'age', 'disability', 'socioeconomic', 'lgbtq'],
                'methodology': 'https://stereowipe.org/documentation'
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/leaderboard/category/<category>.json')
def api_category_leaderboard(category):
    """
    JSON API for category-specific rankings.
    """
    valid_categories = ['gender', 'race', 'religion', 'nationality', 
                       'profession', 'age', 'disability', 'socioeconomic', 'lgbtq']
    
    if category not in valid_categories:
        return jsonify({
            'success': False,
            'error': f'Invalid category. Valid options: {valid_categories}'
        }), 400
    
    try:
        rankings = db.get_category_rankings(category)
        return jsonify({
            'success': True,
            'category': category,
            'data': rankings
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/regional/<model_name>.json')
def api_regional_scores(model_name):
    """
    JSON API for regional scores of a model.
    """
    try:
        regional_scores = db.get_regional_comparison(model_name)
        return jsonify({
            'success': True,
            'model_name': model_name,
            'data': regional_scores
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/models.json')
def api_models():
    """
    JSON API for list of all models.
    """
    try:
        models = db.get_active_models()
        return jsonify({
            'success': True,
            'data': models,
            'total': len(models)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/prompts/stats.json')
def api_prompt_stats():
    """
    JSON API for prompt dataset statistics.
    """
    try:
        stats = db.get_prompt_stats()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ========== ARENA ROUTES (existing but enhanced) ==========

@app.route('/arena')
def arena():
    """
    Displays the Arena page for model comparison voting.
    Battle data is stored in session to ensure vote matches displayed battle.
    """
    try:
        battle_data = db.get_random_battle_data()
        battle_count = len(db.get_arena_battles())
        
        if battle_data:
            # Store current battle in session for vote validation
            session['current_battle'] = battle_data
            
            return render_template('arena.html', 
                                 battle_data=battle_data,
                                 battle_count=battle_count,
                                 active_page='arena')
        else:
            return render_template('arena.html',
                                 error_message="Not enough model responses available for arena battles. Run evaluations first.",
                                 battle_count=battle_count,
                                 active_page='arena')
    except Exception as e:
        print(f"Error loading arena: {e}")
        return render_template('arena.html',
                             error_message=f"Error loading arena: {str(e)}",
                             battle_count=0,
                             active_page='arena')


@app.route('/arena/vote', methods=['POST'])
def arena_vote():
    """
    Records an arena vote using battle data stored in session.
    """
    try:
        data = request.get_json()
        winner = data.get('winner')
        demographics = data.get('demographics', {})
        
        if winner not in ['a', 'b', 'tie']:
            return jsonify({'success': False, 'error': 'Invalid winner'}), 400
        
        # Get battle data from session (the actual battle shown to user)
        battle_data = session.get('current_battle')
        
        if not battle_data:
            return jsonify({
                'success': False, 
                'error': 'No active battle found. Please refresh the page.'
            }), 400
        
        # Generate or get session ID for tracking
        import uuid
        if 'user_session_id' not in session:
            session['user_session_id'] = str(uuid.uuid4())
        session_id = session['user_session_id']
        
        # Record the battle result
        db.insert_arena_battle(
            session_id=session_id,
            prompt_id=battle_data['prompt_id'],
            model_a=battle_data['model_a'],
            response_a=battle_data['response_a'],
            model_b=battle_data['model_b'],
            response_b=battle_data['response_b'],
            winner=winner,
            voter_region=demographics.get('region'),
            voter_country=demographics.get('country'),
            voter_age_range=demographics.get('age_range'),
            voter_gender=demographics.get('gender'),
            prompt_category=battle_data.get('prompt_category')
        )
        
        # Clear current battle from session (prevents double voting)
        session.pop('current_battle', None)
        
        return jsonify({
            'success': True,
            'winner': winner,
            'model_a': battle_data['model_a'],
            'model_b': battle_data['model_b'],
            'battle_count': len(db.get_arena_battles())
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/arena/demographics.json')
def api_arena_demographics():
    """JSON API for aggregate arena demographic distributions."""
    try:
        return jsonify({
            'success': True,
            'data': db.get_arena_demographic_breakdown()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Runs the Flask development server
    # Debug=True enables auto-reloading on code changes and provides a debugger
    # host='0.0.0.0' would make it accessible from network, default is 127.0.0.1 (localhost)
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
