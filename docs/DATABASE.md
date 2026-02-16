# Database Documentation

StereoWipe uses SQLite for data persistence, storing evaluation results, human annotations, and arena battle data.

## Database Location

The database file `stereowipe.db` is created in the project root directory on first use.

## Schema

### evaluations table
Stores LLM judge evaluation results:
- `id`: Primary key
- `prompt_id`: Identifier for the prompt
- `prompt_text`: Full text of the prompt
- `model_name`: Name of the evaluated model
- `response_text`: Model's response
- `judge_name`: Name of the judge(s) used
- `is_stereotype`: Boolean indicating if response contains stereotypes
- `severity_score`: Severity score (0-1) if stereotyping detected
- `reasoning`: Judge's explanation
- `created_at`: Timestamp

### human_annotations table
Stores human annotator assessments:
- `id`: Primary key
- `session_id`: Annotator session identifier
- `prompt_id`: Identifier for the prompt
- `response_text`: Model's response being annotated
- `is_stereotype`: Human assessment
- `severity_score`: Human-assigned severity
- `annotator_comments`: Optional comments
- `created_at`: Timestamp

### arena_battles table
Stores preference data from head-to-head comparisons:
- `id`: Primary key
- `session_id`: Battle session identifier
- `prompt_id`: Identifier for the prompt
- `model_a`, `model_b`: Names of competing models
- `response_a`, `response_b`: Model responses
- `winner`: 'a', 'b', or 'tie'
- `created_at`: Timestamp

## CLI Integration

### Running evaluations with database storage (default)
```bash
python biaswipe/cli.py --prompts prompts.json --annotations annotations.json \
    --model-responses-dir responses/ --judge openai
```

### Running without database storage
```bash
python biaswipe/cli.py --prompts prompts.json --annotations annotations.json \
    --model-responses-dir responses/ --judge openai --no-db
```

## Database Utilities

The `biaswipe/db_utils.py` script provides utilities for database management:

### View summary statistics
```bash
python -m biaswipe.db_utils summary
```

### List recent evaluations
```bash
# All models
python -m biaswipe.db_utils list-evaluations

# Specific model
python -m biaswipe.db_utils list-evaluations --model gpt-4
```

### Export data
```bash
# Export to JSON
python -m biaswipe.db_utils export evaluations evaluations.json

# Export to CSV
python -m biaswipe.db_utils export evaluations evaluations.csv --format csv
```

### View unannotated responses
```bash
python -m biaswipe.db_utils list-unannotated
```

### Arena battle statistics
```bash
python -m biaswipe.db_utils arena-stats
```

## Python API

```python
from biaswipe.database import Database

# Initialize database
db = Database()

# Insert evaluation
eval_id = db.insert_evaluation(
    prompt_id="001",
    prompt_text="Tell me about elderly drivers",
    model_name="gpt-4",
    response_text="Response text here...",
    judge_name="OpenAIJudge",
    is_stereotype=True,
    severity_score=0.7,
    reasoning="Contains age-based generalization"
)

# Query evaluations
evaluations = db.get_evaluations_by_model("gpt-4")

# Get summary statistics
stats = db.get_evaluation_summary()
```

## Backup and Migration

To backup the database:
```bash
cp stereowipe.db stereowipe_backup_$(date +%Y%m%d).db
```

The database schema is automatically created on first use, ensuring compatibility across versions.