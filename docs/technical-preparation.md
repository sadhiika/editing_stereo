# Technical Preparation for StereoWipe MVP

This document outlines the specific technical work required to transform the current prototype into a minimal viable research tool. Each task includes concrete implementation details and acceptance criteria.

## Current Technical Debt

### 1. Mock Implementations
- **Issue**: OpenAI and Anthropic judges return mock data
- **Impact**: Cannot conduct real research with fake data
- **Priority**: CRITICAL - Blocks all research

### 2. No Data Persistence
- **Issue**: No database, results only in JSON files
- **Impact**: Cannot collect human annotations or preferences
- **Priority**: HIGH - Blocks validation studies

### 3. Broken Web Features
- **Issue**: Upload endpoint referenced but not implemented
- **Impact**: Confusing user experience
- **Priority**: MEDIUM - Affects usability

### 4. Missing Progress Feedback
- **Issue**: CLI runs silently, no progress indication
- **Impact**: Users don't know if large batches are running
- **Priority**: LOW - Quality of life

---

## Phase 0: Immediate Technical Tasks (Week 1-2)

### Task 0.1: Implement Real OpenAI Judge
**Current State**: Returns mock stereotyping assessments
**Required Changes**:

```python
# biaswipe/judge.py - Replace mock with real implementation
class OpenAIJudge(Judge):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def judge(self, prompt: str, response: str) -> dict:
        # Load judge prompt template
        system_prompt = self._load_judge_prompt()
        
        # Make actual API call
        completion = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Prompt: {prompt}\nResponse: {response}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Parse and validate response
        result = json.loads(completion.choices[0].message.content)
        self._validate_response(result)
        return result
```

**Testing Required**:
- Unit tests with mocked API responses
- Integration test with real API (marked as slow)
- Cost tracking test
- Error handling for API failures

**Acceptance Criteria**:
- ✓ Real API calls to GPT-4
- ✓ Structured JSON responses
- ✓ Proper error handling
- ✓ Cost estimation before running

---

### Task 0.2: Add SQLite Database
**Current State**: No database
**Required Changes**:

```python
# biaswipe/database.py - New file
import sqlite3
from contextlib import contextmanager
from typing import Dict, List, Optional

class Database:
    def __init__(self, db_path: str = "stereowipe.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist"""
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
                    winner TEXT NOT NULL,  -- 'a', 'b', or 'tie'
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
```

**Integration Points**:
- Modify `scoring.py` to save results to database
- Add database queries for web viewer
- Create data export utilities

**Acceptance Criteria**:
- ✓ SQLite database initialized on first run
- ✓ All evaluations saved to database
- ✓ Query functions for common operations
- ✓ Data export to CSV/JSON

---

### Task 0.3: Fix Web Upload Endpoint
**Current State**: Form exists but endpoint missing
**Required Changes**:

```python
# biaswipe_viewer/webserver.py - Add missing endpoint
@app.route('/upload_report', methods=['POST'])
def upload_report():
    if 'report' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['report']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.json'):
        # Save to temporary location
        temp_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
        file.save(temp_path)
        
        # Validate JSON structure
        try:
            with open(temp_path, 'r') as f:
                report_data = json.load(f)
                # Basic validation
                required_keys = ['models', 'overall_metrics']
                if not all(key in report_data for key in required_keys):
                    raise ValueError("Invalid report format")
        except Exception as e:
            os.remove(temp_path)
            return jsonify({'error': f'Invalid JSON: {str(e)}'}), 400
        
        # Store in session for viewing
        session['uploaded_report'] = temp_path
        return jsonify({'redirect': '/report_display'}), 200
    
    return jsonify({'error': 'Only JSON files accepted'}), 400
```

**Acceptance Criteria**:
- ✓ Upload endpoint functional
- ✓ JSON validation
- ✓ Proper error messages
- ✓ Successful redirect to display

---

### Task 0.4: Add CLI Progress Indicators
**Current State**: Silent execution
**Required Changes**:

```python
# biaswipe/cli.py - Add progress bars
from tqdm import tqdm

def evaluate_model(model_name, responses, prompts, judges, cache_dir):
    results = {}
    
    # Add progress bar
    total_evaluations = len(responses) * len(judges)
    with tqdm(total=total_evaluations, desc=f"Evaluating {model_name}") as pbar:
        for prompt_id, response in responses.items():
            # ... existing code ...
            pbar.update(1)
            
    return results
```

**Acceptance Criteria**:
- ✓ Progress bars for long operations
- ✓ ETA estimates
- ✓ Can be disabled with --quiet flag

---

## Phase 1: MVP Infrastructure (Week 3-4)

### Task 1.1: Human Annotation Interface
**Create simple web forms for collecting human ratings**

```python
# biaswipe_viewer/templates/annotate.html
# Simple form with:
# - Display prompt and response
# - Binary choice: stereotype or not
# - Severity slider (1-5) if stereotype
# - Optional comments
# - Next button to load next item

# biaswipe_viewer/webserver.py
@app.route('/annotate')
def annotate():
    # Get next unannotated item from database
    # Track session_id for anonymous users
    # Simple round-robin assignment
```

### Task 1.2: Batch Processing Scripts
**Scripts for research workflows**

```bash
# scripts/run_validation_study.py
# - Load 100 prompts
# - Get responses from 5 models  
# - Run through LLM judge
# - Export for human annotation

# scripts/calculate_agreement.py
# - Load human annotations
# - Calculate inter-rater reliability
# - Compare with LLM judge
# - Generate statistics

# scripts/export_research_data.py
# - Export all data to CSV
# - Create HuggingFace dataset
# - Generate paper tables
```

### Task 1.3: Simple Arena Implementation
**Minimal preference collection system**

```python
# biaswipe_viewer/templates/arena.html
# Dead simple:
# - Show prompt
# - Show two responses (anonymous)
# - Three buttons: A wins, B wins, Tie
# - Reveal models after vote
# - Show next battle

# scripts/calculate_elo.py
# - Run daily/hourly
# - Read arena_battles table
# - Calculate Elo scores
# - Update static leaderboard.json
```

---

## Phase 2: Research Tooling (Week 5-6)

### Task 2.1: Analysis Notebooks
**Jupyter notebooks for reproducible research**

```
notebooks/
├── 01_data_exploration.ipynb
├── 02_human_llm_agreement.ipynb
├── 03_arena_analysis.ipynb
├── 04_bias_categories.ipynb
└── 05_paper_figures.ipynb
```

### Task 2.2: Dataset Packaging
**Prepare for public release**

```python
# scripts/prepare_dataset.py
# - Anonymize any PII
# - Add metadata
# - Create train/test splits
# - Generate dataset card
# - Upload to HuggingFace
```

### Task 2.3: Documentation Update
**Research-focused documentation**

```
docs/
├── SETUP.md          # Simple setup guide
├── RESEARCH.md      # How to reproduce our results
├── CONTRIBUTING.md  # How to add prompts/annotations
└── DATA_FORMAT.md   # Schema documentation
```

---

## Technical Architecture (Simplified)

### Before (Current):
```
┌─────────────┐     ┌──────────────┐
│   CLI Tool  │────▶│ Mock Judges  │
└─────────────┘     └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ JSON Reports │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Web Viewer  │
                    └──────────────┘
```

### After (MVP):
```
┌─────────────┐     ┌──────────────┐     ┌──────────┐
│   CLI Tool  │────▶│ Real LLM API │────▶│  SQLite  │
└─────────────┘     └──────────────┘     └──────────┘
                                                │
                    ┌──────────────┐            │
                    │   Web Forms  │────────────┘
                    │  (Annotate)  │
                    └──────────────┘
                           │
                    ┌──────────────┐
                    │Arena (Simple)│
                    └──────────────┘
```

---

## Dependencies to Add

### Python Dependencies:
```txt
# requirements.txt additions
openai>=1.0.0        # For real API calls
tqdm>=4.65.0         # For progress bars
pandas>=2.0.0        # For data analysis
scipy>=1.10.0        # For statistical tests
jupyter>=1.0.0       # For notebooks
datasets>=2.14.0     # For HuggingFace
```

### Development Dependencies:
```txt
# requirements-dev.txt
pytest-mock>=3.11.0  # For mocking API calls
black>=23.0.0        # For code formatting
mypy>=1.5.0          # For type checking
```

---

## Testing Strategy

### Unit Tests:
- Mock API calls to avoid costs
- Test database operations
- Test metrics calculations
- Test data validation

### Integration Tests:
- Small test with real API (5 calls)
- End-to-end annotation flow
- Arena battle recording
- Data export verification

### Research Validation:
- Synthetic data for known results
- Statistical power analysis
- Reproducibility checks

---

## Deployment Simplification

### Current: Complex
- Docker, Cloud Run, multiple configs

### MVP: Dead Simple
- Single `pip install -e .`
- SQLite file in project directory
- `python -m biaswipe.cli` to run
- `python -m biaswipe_viewer.webserver` for web
- Static files on GitHub Pages

---

## Risk Mitigation

### API Costs:
- Cache aggressively (already implemented)
- Batch API calls
- Set spending limits
- Use GPT-3.5 for testing

### Data Loss:
- SQLite backups before major operations
- Git LFS for dataset storage
- Export to CSV regularly

### Scalability:
- Don't worry about it for MVP
- SQLite handles 100k records fine
- Static leaderboard is enough

---

## Success Criteria

### Week 2 Checkpoint:
- ✓ Real LLM judge working
- ✓ SQLite integrated
- ✓ Web upload fixed
- ✓ Progress indicators added

### Week 4 Checkpoint:
- ✓ Human annotation flow complete
- ✓ 100+ annotations collected
- ✓ Arena prototype functional
- ✓ Basic analysis notebook

### Week 6 Checkpoint:
- ✓ Dataset packaged
- ✓ Research findings documented
- ✓ Paper draft started
- ✓ Code release-ready

This technical preparation focuses on the absolute minimum needed to conduct meaningful research while avoiding over-engineering.