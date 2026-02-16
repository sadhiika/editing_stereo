# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StereoWipe is a stereotyping evaluation benchmark for Large Language Models that uses an LLM-as-a-Judge paradigm to assess stereotypical content in model responses. The project consists of a core Python package for evaluation and a Flask-based web viewer for report visualization.

## Common Commands

### Running the Main Evaluation
```bash
# Basic evaluation run
python biaswipe/cli.py --prompts sample_data/prompts.json --annotations sample_data/annotations.json --model-responses-dir sample_data/ --category-weights sample_data/category_weights.json --judge openai --output-dir output

# With custom judge prompt
python biaswipe/cli.py --prompts prompts.json --annotations annotations.json --model-responses-dir responses/ --judge anthropic --judge-prompt custom_prompt.txt
```

### Running Tests
```bash
# Run all tests (assuming pytest is installed)
pytest tests/

# Run specific test file
pytest tests/test_metrics.py

# Run with verbose output
pytest -v tests/
```

### Web Viewer
```bash
# Start the web viewer locally
python biaswipe_viewer/webserver.py

# Docker deployment
cd biaswipe_viewer
docker build -t biaswipe-viewer .
docker run -p 5000:5000 biaswipe-viewer
```

## Architecture

### Core Components

1. **CLI Entry Point** (`biaswipe/cli.py`): Orchestrates the evaluation pipeline through Click commands

2. **Data Flow**:
   - `data_loader.py`: Loads prompts, annotations, and model responses from JSON files
   - `judge.py`: Implements LLM-as-a-Judge with mock, OpenAI, and Anthropic judge classes
   - `scoring.py`: Main evaluation logic that coordinates judging and metric calculation
   - `metrics.py`: Calculates Stereotype Rate (SR), Stereotype Severity Score (SSS), Conditional Stereotype Severity Score (CSSS), and Weighted Overall Stereotyping Index (WOSI) metrics
   - `report.py`: Generates comprehensive JSON reports

3. **Judge System**: 
   - Abstract `Judge` base class with three implementations
   - Caching mechanism stores judge responses in `.biaswipe_cache/`
   - JSON schema validation for judge responses (`json_output_schema.json`)
   - Customizable judge prompts via `judge_prompt.txt`

4. **Web Viewer** (`biaswipe_viewer/`):
   - Flask application for visualizing evaluation reports
   - Templates use Bootstrap for responsive design
   - Supports report upload and interactive exploration

### Key Design Patterns

- **Judge Factory Pattern**: `get_judge()` function creates appropriate judge instance based on config
- **Caching Strategy**: Judge responses cached by hash of (prompt, response, judge_prompt) to avoid redundant API calls
- **Metric Aggregation**: Hierarchical aggregation from individual responses → categories → overall scores

## Environment Configuration

### Required API Keys
```bash
# For OpenAI judge
export OPENAI_API_KEY="your-key"

# For Anthropic judge  
export ANTHROPIC_API_KEY="your-key"
```

### Research Batch Processing Scripts

The `scripts/` directory contains batch processing tools for research workflows:

```bash
# Setup sample data for development and testing
python scripts/setup_sample_data.py --num-samples 50 --models gpt-4,claude-3

# Run validation study comparing human and LLM judges
python scripts/run_validation_study.py --config validation_config.json

# Calculate inter-rater reliability and agreement statistics
python scripts/calculate_agreement.py --database --output agreement_analysis.json

# Export research data in multiple formats
python scripts/export_research_data.py --format all --output research_data/

# Run complete example workflow
python scripts/example_workflow.py --quick
```

See `scripts/README.md` for detailed usage instructions and workflow examples.

### Development Notes

- Judge responses are cached in `.biaswipe_cache/` - delete this directory to force re-evaluation
- Mock judge always returns random assessments for testing
- Web viewer runs on port 5000 by default
- Sample data files in `sample_data/` demonstrate expected formats
- Research scripts require additional dependencies: `pip install pandas numpy scipy scikit-learn matplotlib seaborn datasets`