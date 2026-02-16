#!/usr/bin/env python3
"""
Export Research Data Script

This script exports all evaluation data to various formats suitable for research
analysis and publication. It can create CSV files, HuggingFace datasets, 
research paper tables, and comprehensive documentation.

Usage:
    python scripts/export_research_data.py --format csv --output research_data.csv
    python scripts/export_research_data.py --format huggingface --output stereowipe_dataset
    python scripts/export_research_data.py --format all --output research_export/
"""

import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import csv
import zipfile
import shutil

# Add the parent directory to the path so we can import biaswipe
sys.path.insert(0, str(Path(__file__).parent.parent))

from biaswipe.database import Database
from biaswipe import data_loader

# Try to import HuggingFace datasets
try:
    from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: HuggingFace datasets not available. Install with: pip install datasets")


class ResearchDataExporter:
    """Export research data in various formats for analysis and publication."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.db = Database()
        self.export_metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "exporter_version": "1.0.0",
            "data_version": "1.0.0",
            "files_created": []
        }
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data from database and files."""
        data = {}
        
        # Load from database
        print("Loading evaluations from database...")
        evaluations = self.db.get_all_evaluations()
        data['evaluations'] = pd.DataFrame(evaluations)
        
        print("Loading human annotations from database...")
        human_annotations = self.db.get_human_annotations()
        data['human_annotations'] = pd.DataFrame(human_annotations)
        
        print("Loading arena battles from database...")
        arena_battles = self.db.get_arena_battles()
        data['arena_battles'] = pd.DataFrame(arena_battles)
        
        # Load additional data from files if available
        try:
            sample_data_dir = Path(__file__).parent.parent / "sample_data"
            if sample_data_dir.exists():
                print("Loading sample data files...")
                
                # Load prompts
                prompts_file = sample_data_dir / "prompts.json"
                if prompts_file.exists():
                    prompts = data_loader.load_prompts(str(prompts_file))
                    if prompts:
                        prompts_df = pd.DataFrame(
                            [(k, v['text'], v['category']) for k, v in prompts.items()],
                            columns=['prompt_id', 'prompt_text', 'category']
                        )
                        data['prompts'] = prompts_df
                
                # Load category weights
                weights_file = sample_data_dir / "category_weights.json"
                if weights_file.exists():
                    with open(weights_file, 'r') as f:
                        category_weights = json.load(f)
                        weights_df = pd.DataFrame(
                            [(k, v) for k, v in category_weights.items()],
                            columns=['category', 'weight']
                        )
                        data['category_weights'] = weights_df
        except Exception as e:
            print(f"Warning: Could not load sample data files: {e}")
        
        return data
    
    def export_csv(self, data: Dict[str, pd.DataFrame], filename: str):
        """Export data to CSV format."""
        print(f"Exporting to CSV: {filename}")
        
        # Create comprehensive CSV with all evaluation data
        if 'evaluations' in data and len(data['evaluations']) > 0:
            eval_df = data['evaluations'].copy()
            
            # Add prompt categories if available
            if 'prompts' in data and len(data['prompts']) > 0:
                eval_df = eval_df.merge(
                    data['prompts'][['prompt_id', 'category']], 
                    on='prompt_id', 
                    how='left'
                )
            
            # Add category weights if available
            if 'category_weights' in data and len(data['category_weights']) > 0:
                eval_df = eval_df.merge(
                    data['category_weights'], 
                    on='category', 
                    how='left'
                )
            
            csv_path = os.path.join(self.output_dir, filename)
            eval_df.to_csv(csv_path, index=False)
            self.export_metadata['files_created'].append(csv_path)
            print(f"Evaluations exported to: {csv_path}")
        
        # Export each table separately
        for table_name, df in data.items():
            if len(df) > 0:
                table_csv = os.path.join(self.output_dir, f"{table_name}.csv")
                df.to_csv(table_csv, index=False)
                self.export_metadata['files_created'].append(table_csv)
                print(f"{table_name} exported to: {table_csv}")
    
    def export_huggingface_dataset(self, data: Dict[str, pd.DataFrame], dataset_name: str):
        """Export data as HuggingFace dataset."""
        if not HF_AVAILABLE:
            print("HuggingFace datasets not available. Skipping HF export.")
            return
        
        print(f"Creating HuggingFace dataset: {dataset_name}")
        
        # Create dataset from evaluations
        if 'evaluations' in data and len(data['evaluations']) > 0:
            eval_df = data['evaluations'].copy()
            
            # Add prompt categories if available
            if 'prompts' in data and len(data['prompts']) > 0:
                eval_df = eval_df.merge(
                    data['prompts'][['prompt_id', 'category']], 
                    on='prompt_id', 
                    how='left'
                )
            
            # Define features
            features = Features({
                'prompt_id': Value('string'),
                'prompt_text': Value('string'),
                'model_name': Value('string'),
                'response_text': Value('string'),
                'judge_name': Value('string'),
                'is_stereotype': Value('bool'),
                'severity_score': Value('float32'),
                'reasoning': Value('string'),
                'category': Value('string'),
                'created_at': Value('string')
            })
            
            # Create dataset
            dataset = Dataset.from_pandas(eval_df, features=features)
            
            # Create dataset dict with splits
            dataset_dict = DatasetDict({
                'train': dataset,
                'validation': dataset.select(range(min(100, len(dataset)))),  # Small validation set
                'test': dataset.select(range(min(50, len(dataset))))  # Small test set
            })
            
            # Save dataset
            dataset_path = os.path.join(self.output_dir, dataset_name)
            dataset_dict.save_to_disk(dataset_path)
            self.export_metadata['files_created'].append(dataset_path)
            print(f"HuggingFace dataset saved to: {dataset_path}")
            
            # Create dataset card
            self._create_dataset_card(dataset_dict, dataset_path)
    
    def _create_dataset_card(self, dataset_dict: 'DatasetDict', dataset_path: str):
        """Create a dataset card for the HuggingFace dataset."""
        card_content = f"""
# StereoWipe Dataset

## Dataset Description

The StereoWipe dataset contains evaluations of Large Language Model responses for stereotypical content using an LLM-as-a-Judge paradigm.

## Dataset Structure

### Data Fields

- `prompt_id`: Unique identifier for the prompt
- `prompt_text`: The original prompt text
- `model_name`: Name of the model that generated the response
- `response_text`: The model's response to the prompt
- `judge_name`: Name of the LLM judge that evaluated the response
- `is_stereotype`: Boolean indicating if the response contains stereotypes
- `severity_score`: Numerical score (1-5) indicating severity of stereotyping
- `reasoning`: Judge's reasoning for the evaluation
- `category`: Category of the prompt (e.g., profession, nationality, gender)
- `created_at`: Timestamp of the evaluation

### Data Splits

- **Train**: {len(dataset_dict['train'])} examples
- **Validation**: {len(dataset_dict['validation'])} examples  
- **Test**: {len(dataset_dict['test'])} examples

## Usage

```python
from datasets import load_from_disk
dataset = load_from_disk("{dataset_path}")
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{{stereowipe_dataset,
    title={{StereoWipe: A Stereotyping Evaluation Benchmark for Large Language Models}},
    year={{{datetime.now().year}}},
    note={{Generated on {datetime.now().strftime('%Y-%m-%d')}}}
}}
```

## License

This dataset is released under the MIT License.

## Contact

For questions about this dataset, please contact the StereoWipe team.
"""
        
        card_path = os.path.join(dataset_path, "README.md")
        with open(card_path, 'w') as f:
            f.write(card_content)
        
        print(f"Dataset card created: {card_path}")
    
    def export_research_tables(self, data: Dict[str, pd.DataFrame], table_prefix: str = "table"):
        """Export formatted tables for research papers."""
        print("Creating research paper tables...")
        
        tables_dir = os.path.join(self.output_dir, "research_tables")
        os.makedirs(tables_dir, exist_ok=True)
        
        if 'evaluations' in data and len(data['evaluations']) > 0:
            eval_df = data['evaluations'].copy()
            
            # Table 1: Model Performance Summary
            if 'model_name' in eval_df.columns:
                model_summary = eval_df.groupby('model_name').agg({
                    'is_stereotype': ['count', 'sum', 'mean'],
                    'severity_score': ['mean', 'std']
                }).round(3)
                
                model_summary.columns = ['Total_Responses', 'Stereotypical_Responses', 'Stereotype_Rate', 'Mean_Severity', 'Std_Severity']
                model_summary = model_summary.reset_index()
                
                table1_path = os.path.join(tables_dir, f"{table_prefix}_1_model_performance.csv")
                model_summary.to_csv(table1_path, index=False)
                self.export_metadata['files_created'].append(table1_path)
                print(f"Table 1 (Model Performance) saved to: {table1_path}")
                
                # LaTeX version
                latex_path = table1_path.replace('.csv', '.tex')
                with open(latex_path, 'w') as f:
                    f.write("\\begin{table}[h]\n")
                    f.write("\\centering\n")
                    f.write("\\caption{Model Performance Summary}\n")
                    f.write("\\begin{tabular}{lrrrrr}\n")
                    f.write("\\toprule\n")
                    f.write("Model & Total & Stereotypical & Stereotype Rate & Mean Severity & Std Severity \\\\\n")
                    f.write("\\midrule\n")
                    
                    for _, row in model_summary.iterrows():
                        f.write(f"{row['model_name']} & {row['Total_Responses']} & {row['Stereotypical_Responses']} & {row['Stereotype_Rate']:.3f} & {row['Mean_Severity']:.3f} & {row['Std_Severity']:.3f} \\\\\n")
                    
                    f.write("\\bottomrule\n")
                    f.write("\\end{tabular}\n")
                    f.write("\\end{table}\n")
                print(f"LaTeX table saved to: {latex_path}")
            
            # Table 2: Category Breakdown
            if 'prompts' in data and len(data['prompts']) > 0:
                eval_with_categories = eval_df.merge(
                    data['prompts'][['prompt_id', 'category']], 
                    on='prompt_id', 
                    how='left'
                )
                
                if 'category' in eval_with_categories.columns:
                    category_summary = eval_with_categories.groupby(['category', 'model_name']).agg({
                        'is_stereotype': ['count', 'mean'],
                        'severity_score': 'mean'
                    }).round(3)
                    
                    category_summary.columns = ['Total_Responses', 'Stereotype_Rate', 'Mean_Severity']
                    category_summary = category_summary.reset_index()
                    
                    table2_path = os.path.join(tables_dir, f"{table_prefix}_2_category_breakdown.csv")
                    category_summary.to_csv(table2_path, index=False)
                    self.export_metadata['files_created'].append(table2_path)
                    print(f"Table 2 (Category Breakdown) saved to: {table2_path}")
            
            # Table 3: Judge Comparison (if multiple judges)
            if 'judge_name' in eval_df.columns:
                judge_counts = eval_df['judge_name'].value_counts()
                if len(judge_counts) > 1:
                    judge_summary = eval_df.groupby('judge_name').agg({
                        'is_stereotype': ['count', 'mean'],
                        'severity_score': ['mean', 'std']
                    }).round(3)
                    
                    judge_summary.columns = ['Total_Evaluations', 'Stereotype_Rate', 'Mean_Severity', 'Std_Severity']
                    judge_summary = judge_summary.reset_index()
                    
                    table3_path = os.path.join(tables_dir, f"{table_prefix}_3_judge_comparison.csv")
                    judge_summary.to_csv(table3_path, index=False)
                    self.export_metadata['files_created'].append(table3_path)
                    print(f"Table 3 (Judge Comparison) saved to: {table3_path}")
    
    def export_analysis_notebooks(self):
        """Create Jupyter notebooks for data analysis."""
        print("Creating analysis notebooks...")
        
        notebooks_dir = os.path.join(self.output_dir, "analysis_notebooks")
        os.makedirs(notebooks_dir, exist_ok=True)
        
        # Basic analysis notebook
        basic_notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# StereoWipe Dataset Analysis\\n", "\\n", "This notebook provides basic analysis of the StereoWipe evaluation dataset."]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import pandas as pd\\n",
                        "import numpy as np\\n",
                        "import matplotlib.pyplot as plt\\n",
                        "import seaborn as sns\\n",
                        "\\n",
                        "# Load the exported data\\n",
                        "evaluations = pd.read_csv('evaluations.csv')\\n",
                        "print(f'Loaded {len(evaluations)} evaluations')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Data Overview"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Basic statistics\\n",
                        "print('Dataset shape:', evaluations.shape)\\n",
                        "print('\\nColumn types:')\\n",
                        "print(evaluations.dtypes)\\n",
                        "print('\\nBasic statistics:')\\n",
                        "print(evaluations.describe())"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Stereotype Rate Analysis"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Stereotype rate by model\\n",
                        "model_stats = evaluations.groupby('model_name').agg({\\n",
                        "    'is_stereotype': ['count', 'sum', 'mean'],\\n",
                        "    'severity_score': ['mean', 'std']\\n",
                        "}).round(3)\\n",
                        "\\n",
                        "print(model_stats)"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Visualizations"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Create visualizations\\n",
                        "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\\n",
                        "\\n",
                        "# Stereotype rate by model\\n",
                        "model_stereotype_rate = evaluations.groupby('model_name')['is_stereotype'].mean()\\n",
                        "model_stereotype_rate.plot(kind='bar', ax=axes[0, 0])\\n",
                        "axes[0, 0].set_title('Stereotype Rate by Model')\\n",
                        "axes[0, 0].set_ylabel('Stereotype Rate')\\n",
                        "\\n",
                        "# Severity score distribution\\n",
                        "evaluations['severity_score'].hist(bins=20, ax=axes[0, 1])\\n",
                        "axes[0, 1].set_title('Severity Score Distribution')\\n",
                        "axes[0, 1].set_xlabel('Severity Score')\\n",
                        "\\n",
                        "# Additional plots can be added here\\n",
                        "\\n",
                        "plt.tight_layout()\\n",
                        "plt.show()"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        notebook_path = os.path.join(notebooks_dir, "basic_analysis.ipynb")
        with open(notebook_path, 'w') as f:
            json.dump(basic_notebook, f, indent=2)
        
        self.export_metadata['files_created'].append(notebook_path)
        print(f"Analysis notebook created: {notebook_path}")
    
    def create_documentation(self):
        """Create comprehensive documentation for the exported data."""
        print("Creating documentation...")
        
        # Data dictionary
        data_dict = {
            "evaluations": {
                "description": "LLM judge evaluations of model responses for stereotypical content",
                "fields": {
                    "id": "Primary key",
                    "prompt_id": "Unique identifier for the prompt",
                    "prompt_text": "The original prompt text",
                    "model_name": "Name of the model that generated the response",
                    "response_text": "The model's response to the prompt",
                    "judge_name": "Name of the LLM judge that evaluated the response",
                    "is_stereotype": "Boolean indicating if the response contains stereotypes",
                    "severity_score": "Numerical score (1-5) indicating severity of stereotyping",
                    "reasoning": "Judge's reasoning for the evaluation",
                    "created_at": "Timestamp of the evaluation"
                }
            },
            "human_annotations": {
                "description": "Human annotator evaluations of model responses",
                "fields": {
                    "id": "Primary key",
                    "session_id": "Unique identifier for the annotation session",
                    "prompt_id": "Unique identifier for the prompt",
                    "response_text": "The model's response being annotated",
                    "is_stereotype": "Boolean indicating if the response contains stereotypes",
                    "severity_score": "Numerical score (1-5) indicating severity of stereotyping",
                    "annotator_comments": "Human annotator's comments",
                    "created_at": "Timestamp of the annotation"
                }
            },
            "arena_battles": {
                "description": "Preference comparisons between model responses",
                "fields": {
                    "id": "Primary key",
                    "session_id": "Unique identifier for the comparison session",
                    "prompt_id": "Unique identifier for the prompt",
                    "model_a": "Name of the first model",
                    "response_a": "Response from the first model",
                    "model_b": "Name of the second model",
                    "response_b": "Response from the second model",
                    "winner": "Winner of the comparison ('a', 'b', or 'tie')",
                    "created_at": "Timestamp of the comparison"
                }
            }
        }
        
        # Save data dictionary
        dict_path = os.path.join(self.output_dir, "data_dictionary.json")
        with open(dict_path, 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        # Create README
        readme_content = f"""# StereoWipe Research Data Export

This directory contains exported data from the StereoWipe stereotyping evaluation benchmark.

## Export Information

- **Export Date**: {self.export_metadata['export_timestamp']}
- **Exporter Version**: {self.export_metadata['exporter_version']}
- **Data Version**: {self.export_metadata['data_version']}

## Files Included

"""
        
        for file_path in self.export_metadata['files_created']:
            rel_path = os.path.relpath(file_path, self.output_dir)
            readme_content += f"- `{rel_path}`\n"
        
        readme_content += f"""

## Data Description

The StereoWipe dataset contains evaluations of Large Language Model responses for stereotypical content using an LLM-as-a-Judge paradigm. The dataset includes:

1. **LLM Judge Evaluations**: Automated assessments of model responses
2. **Human Annotations**: Human expert evaluations for comparison
3. **Arena Battles**: Preference comparisons between model responses

## Usage

### CSV Files
Load the CSV files using any data analysis tool:

```python
import pandas as pd
evaluations = pd.read_csv('evaluations.csv')
```

### HuggingFace Dataset
If exported in HuggingFace format:

```python
from datasets import load_from_disk
dataset = load_from_disk('stereowipe_dataset')
```

### Research Tables
Pre-formatted tables for research papers are available in the `research_tables/` directory.

### Analysis Notebooks
Jupyter notebooks for data analysis are provided in the `analysis_notebooks/` directory.

## Data Dictionary

See `data_dictionary.json` for detailed field descriptions.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{{stereowipe_dataset,
    title={{StereoWipe: A Stereotyping Evaluation Benchmark for Large Language Models}},
    year={{{datetime.now().year}}},
    note={{Exported on {datetime.now().strftime('%Y-%m-%d')}}}
}}
```

## License

This dataset is released under the MIT License.

## Contact

For questions about this dataset, please contact the StereoWipe team.
"""
        
        readme_path = os.path.join(self.output_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        self.export_metadata['files_created'].extend([dict_path, readme_path])
        print(f"Documentation created: {readme_path}")
    
    def create_export_archive(self, archive_name: str = "stereowipe_research_data"):
        """Create a ZIP archive of all exported data."""
        print(f"Creating archive: {archive_name}.zip")
        
        archive_path = os.path.join(self.output_dir, f"{archive_name}.zip")
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.export_metadata['files_created']:
                if os.path.exists(file_path):
                    arcname = os.path.relpath(file_path, self.output_dir)
                    zipf.write(file_path, arcname)
        
        print(f"Archive created: {archive_path}")
        return archive_path
    
    def export_all_formats(self, base_name: str = "stereowipe_data"):
        """Export data in all available formats."""
        print("Loading all data...")
        data = self.load_all_data()
        
        print(f"Loaded {len(data)} data tables:")
        for table_name, df in data.items():
            print(f"  {table_name}: {len(df)} rows")
        
        # Export to CSV
        print("\\nExporting to CSV...")
        self.export_csv(data, f"{base_name}.csv")
        
        # Export to HuggingFace format
        if HF_AVAILABLE:
            print("\\nExporting to HuggingFace format...")
            self.export_huggingface_dataset(data, f"{base_name}_dataset")
        
        # Create research tables
        print("\\nCreating research tables...")
        self.export_research_tables(data, "research_table")
        
        # Create analysis notebooks
        print("\\nCreating analysis notebooks...")
        self.export_analysis_notebooks()
        
        # Create documentation
        print("\\nCreating documentation...")
        self.create_documentation()
        
        # Save export metadata
        metadata_path = os.path.join(self.output_dir, "export_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.export_metadata, f, indent=2)
        
        print(f"\\nExport complete! All files saved to: {self.output_dir}")
        
        return self.export_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Export StereoWipe research data in various formats"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "huggingface", "research_tables", "notebooks", "all"],
        default="all",
        help="Export format (default: all)"
    )
    parser.add_argument(
        "--output",
        default="research_data_export",
        help="Output directory or filename (default: research_data_export)"
    )
    parser.add_argument(
        "--base-name",
        default="stereowipe_data",
        help="Base name for exported files (default: stereowipe_data)"
    )
    parser.add_argument(
        "--create-archive",
        action="store_true",
        help="Create ZIP archive of all exported data"
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw database dumps in export"
    )
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = ResearchDataExporter(args.output)
    
    # Load data
    print("Loading data from database...")
    data = exporter.load_all_data()
    
    print(f"Loaded {len(data)} data tables:")
    for table_name, df in data.items():
        print(f"  {table_name}: {len(df)} rows")
    
    # Export based on format
    if args.format == "csv":
        exporter.export_csv(data, f"{args.base_name}.csv")
    elif args.format == "huggingface":
        if HF_AVAILABLE:
            exporter.export_huggingface_dataset(data, f"{args.base_name}_dataset")
        else:
            print("HuggingFace datasets not available. Please install with: pip install datasets")
    elif args.format == "research_tables":
        exporter.export_research_tables(data, "research_table")
    elif args.format == "notebooks":
        exporter.export_analysis_notebooks()
    elif args.format == "all":
        exporter.export_all_formats(args.base_name)
    
    # Create archive if requested
    if args.create_archive:
        archive_path = exporter.create_export_archive(f"{args.base_name}_export")
        print(f"Archive created: {archive_path}")
    
    print("\\nExport completed successfully!")


if __name__ == "__main__":
    main()