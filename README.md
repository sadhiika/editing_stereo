# StereoWipe Batch Processing Scripts

This directory contains batch processing scripts for research workflows in StereoWipe. These scripts implement Phase 1.3 of the development plan, focusing on tools that researchers can use to process data and generate results.

## Scripts Overview

### 1. `setup_sample_data.py`
**Purpose**: Populate database with sample evaluations and create test data for development.

**Usage**:
```bash
# Generate 50 sample prompts with default models
python scripts/setup_sample_data.py

# Generate 100 samples with custom models
python scripts/setup_sample_data.py --num-samples 100 --models gpt-4,claude-3,gemini-pro

# Clear existing data and generate new samples
python scripts/setup_sample_data.py --clear-existing --num-samples 25

# Save to custom directory
python scripts/setup_sample_data.py --output-dir custom_samples --quiet
```

**Features**:
- Generates diverse prompts across multiple bias categories
- Creates mock model responses with varying stereotype levels
- Simulates human annotations for testing
- Populates database with structured evaluation data
- Exports sample data files for development use

### 2. `run_validation_study.py`
**Purpose**: Run validation studies comparing human annotations with LLM judge results.

**Usage**:
```bash
# Run with configuration file
python scripts/run_validation_study.py --config validation_config.json

# Run with command line arguments
python scripts/run_validation_study.py --prompts sample_data/prompts.json --models gpt-4,claude-3 --judges openai,anthropic

# Run with sampling for annotation tasks
python scripts/run_validation_study.py --prompts prompts.json --models gpt-4,claude-3 --judges openai --sample-size 50
```

**Features**:
- Loads curated prompts for validation
- Generates or loads model responses
- Runs LLM judge evaluations
- Creates structured human annotation tasks
- Exports data for statistical analysis
- Tracks study metadata and progress

**Configuration File** (`validation_config.json`):
```json
{
  "study_name": "StereoWipe Validation Study",
  "prompts_path": "sample_data/prompts.json",
  "models": ["gpt-4", "claude-3-opus", "gemini-pro"],
  "judges": ["openai", "anthropic"],
  "sample_size": 100,
  "annotations_per_response": 3
}
```

### 3. `calculate_agreement.py`
**Purpose**: Calculate inter-rater reliability and agreement statistics.

**Usage**:
```bash
# Load data from database
python scripts/calculate_agreement.py --database --output agreement_report.json

# Load from files
python scripts/calculate_agreement.py --human-annotations annotations.csv --llm-evaluations evaluations.json

# Skip visualizations
python scripts/calculate_agreement.py --database --no-visualizations --output-dir agreement_analysis
```

**Features**:
- Calculates human-human inter-rater reliability
- Compares human annotations with LLM judges
- Computes Cohen's Kappa and correlation statistics
- Creates confusion matrices
- Generates agreement visualizations
- Produces research-ready statistical reports

**Output Files**:
- `agreement_analysis.json`: Detailed statistical results
- `agreement_analysis_summary.txt`: Human-readable summary
- `agreement_visualizations.png`: Agreement plots and distributions

### 4. `export_research_data.py`
**Purpose**: Export evaluation data for research analysis and publication.

**Usage**:
```bash
# Export all formats
python scripts/export_research_data.py --format all --output research_export/

# Export specific format
python scripts/export_research_data.py --format csv --output stereowipe_data.csv
python scripts/export_research_data.py --format huggingface --output stereowipe_dataset

# Create archive
python scripts/export_research_data.py --format all --create-archive
```

**Export Formats**:
- **CSV**: Structured data files for analysis
- **HuggingFace Dataset**: Ready-to-use datasets for ML research
- **Research Tables**: Pre-formatted tables for papers
- **Analysis Notebooks**: Jupyter notebooks for data exploration

**Features**:
- Exports all evaluation data from database
- Creates comprehensive documentation
- Generates research-ready tables
- Includes data dictionary and metadata
- Supports multiple output formats
- Creates analysis notebooks

## Installation and Dependencies

### Required Dependencies
```bash
pip install pandas numpy click tqdm
```

### Optional Dependencies (for full functionality)
```bash
# For statistical analysis
pip install scipy scikit-learn

# For visualizations
pip install matplotlib seaborn

# For HuggingFace datasets
pip install datasets

# For Jupyter notebooks
pip install jupyter
```

## Workflow Examples

### Complete Validation Study Workflow

1. **Setup Sample Data**:
```bash
python scripts/setup_sample_data.py --num-samples 100 --models gpt-4,claude-3,gemini-pro
```

2. **Run Validation Study**:
```bash
python scripts/run_validation_study.py --config validation_config.json --output validation_results.json
```

3. **Calculate Agreement**:
```bash
python scripts/calculate_agreement.py --database --output agreement_analysis.json
```

4. **Export Research Data**:
```bash
python scripts/export_research_data.py --format all --output research_data/ --create-archive
```

### Research Publication Workflow

1. **Export Research Tables**:
```bash
python scripts/export_research_data.py --format research_tables --output paper_tables/
```

2. **Create Analysis Notebooks**:
```bash
python scripts/export_research_data.py --format notebooks --output analysis/
```

3. **Export HuggingFace Dataset**:
```bash
python scripts/export_research_data.py --format huggingface --output stereowipe_dataset
```

## Configuration Options

### Database Configuration
All scripts use the default SQLite database (`stereowipe.db`) in the project root. To use a different database:
```bash
export STEREOWIPE_DB_PATH="/path/to/your/database.db"
```

### API Keys
For real LLM judge evaluations, set your API keys:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Output Directory Structure
```
research_export/
├── README.md                          # Documentation
├── data_dictionary.json               # Field descriptions
├── export_metadata.json              # Export information
├── evaluations.csv                    # Main evaluation data
├── human_annotations.csv             # Human annotation data
├── arena_battles.csv                 # Preference comparison data
├── research_tables/                   # Pre-formatted tables
│   ├── research_table_1_model_performance.csv
│   ├── research_table_2_category_breakdown.csv
│   └── research_table_3_judge_comparison.csv
├── analysis_notebooks/               # Jupyter notebooks
│   └── basic_analysis.ipynb
├── stereowipe_dataset/               # HuggingFace dataset
│   └── README.md
└── stereowipe_data_export.zip        # Complete archive
```

## Error Handling and Troubleshooting

### Common Issues

1. **Missing API Keys**:
   - Scripts will fall back to mock judges if API keys are not provided
   - Set environment variables for OpenAI and Anthropic APIs

2. **Database Connection Issues**:
   - Ensure the database file exists and is accessible
   - Check file permissions in the project directory

3. **Missing Dependencies**:
   - Install optional dependencies for full functionality
   - Scripts will warn about missing libraries and continue with reduced functionality

4. **Memory Issues with Large Datasets**:
   - Use `--sample-size` to limit the number of processed items
   - Process data in smaller batches

### Debugging

Enable verbose output in most scripts:
```bash
python scripts/script_name.py --verbose
```

Use quiet mode for automated workflows:
```bash
python scripts/script_name.py --quiet
```

## Contributing

When adding new scripts:
1. Follow the existing naming convention
2. Include comprehensive command-line help
3. Add error handling for common issues
4. Update this README with usage examples
5. Include unit tests in the `tests/` directory

## Research Use Cases

### Academic Research
- **Benchmark Evaluation**: Compare multiple models on stereotyping tasks
- **Human-AI Agreement**: Study consistency between human and AI evaluators
- **Bias Analysis**: Analyze stereotyping patterns across different categories

### Industry Applications
- **Model Evaluation**: Assess model safety before deployment
- **Bias Detection**: Identify problematic patterns in model outputs
- **Quality Assurance**: Validate model responses for fairness

### Dataset Creation
- **Annotation Studies**: Create human-labeled datasets for training
- **Benchmark Development**: Build comprehensive evaluation suites
- **Research Publication**: Generate reproducible datasets for sharing

## Support

For questions or issues with these scripts:
1. Check the troubleshooting section above
2. Review the error messages for specific guidance
3. Consult the main project documentation
4. Open an issue in the project repository

## License

These scripts are part of the StereoWipe project and are released under the MIT License.

## Phase 2: Rubric-based evaluations

Run rubric evaluations (dry run):

```bash
python scripts/run_evaluations.py --db-path stereowipe.db --judge-name gemini_rubric_v1 --limit 5 --dry-run
```

Run real Gemini rubric evaluations:

```bash
GOOGLE_API_KEY=... python scripts/run_evaluations.py --db-path stereowipe.db --judge-name gemini_rubric_v1
```


## Phase 3: Metrics and leaderboard snapshot

Generate leaderboard snapshot from rubric evaluations:

```bash
python scripts/generate_leaderboard.py --db-path stereowipe.db --judge-name gemini_rubric_v1
```

Programmatic metrics usage:

```bash
python -c "from metrics import StereoWipeMetrics; m=StereoWipeMetrics(); print(m.regional_disagreement_rate())"
```

## Phase 4: Import + validation hardening

Validate rubric file before running evaluations:

```bash
python scripts/validate_rubric.py --rubric-path data/rubric.json
```

Import responses from a JSON file (if responses were collected elsewhere):

```bash
python scripts/import_responses.py --db-path stereowipe.db --file data/model_responses/Jules.json --model jules --provider import
```


## Active free-tier model set

The response collection pipeline is currently configured to run only these models:

- `deepseek-v3.2`
- `deepseek-r1-distill-llama-70b`
- `gemma-3-12b`
- `phi-4`

Required environment variables for this set:

```bash
export DEEPSEEK_API_KEY=...   # for DeepSeek models
export HF_TOKEN=...           # for Hugging Face models
```
