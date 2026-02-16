#!/usr/bin/env python3
"""
Calculate Agreement Script

This script calculates inter-rater reliability between human annotators and
compares human annotations with LLM judge results. It generates statistical
reports and agreement matrices for research analysis.

Usage:
    python scripts/calculate_agreement.py --human-annotations annotations.csv --llm-evaluations evaluations.json
    python scripts/calculate_agreement.py --database --output agreement_report.json
"""

import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path so we can import biaswipe
sys.path.insert(0, str(Path(__file__).parent.parent))

from biaswipe.database import Database

# Try to import statistical libraries
try:
    from scipy import stats
    from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    print("Warning: Statistical libraries not available. Install with: pip install scipy scikit-learn seaborn matplotlib")


class AgreementCalculator:
    """Calculate inter-rater reliability and agreement statistics."""
    
    def __init__(self, output_dir: str = "agreement_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "human_human_agreement": {},
            "human_llm_agreement": {},
            "llm_llm_agreement": {},
            "summary_statistics": {},
            "agreement_matrices": {}
        }
    
    def load_human_annotations(self, annotations_path: str) -> pd.DataFrame:
        """Load human annotations from CSV or JSON file."""
        if annotations_path.endswith('.csv'):
            df = pd.read_csv(annotations_path)
        elif annotations_path.endswith('.json'):
            with open(annotations_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError("Annotations file must be CSV or JSON")
        
        # Standardize column names
        column_mapping = {
            'prompt_id': 'prompt_id',
            'response_text': 'response_text',
            'annotator_id': 'annotator_id',
            'session_id': 'annotator_id',  # session_id can be used as annotator_id
            'is_stereotype': 'is_stereotype',
            'severity_score': 'severity_score',
            'annotator_comments': 'comments',
            'created_at': 'timestamp'
        }
        
        df = df.rename(columns={old: new for old, new in column_mapping.items() if old in df.columns})
        
        # Ensure required columns exist
        required_columns = ['prompt_id', 'annotator_id', 'is_stereotype']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    def load_llm_evaluations(self, evaluations_path: str) -> pd.DataFrame:
        """Load LLM evaluations from JSON file."""
        with open(evaluations_path, 'r') as f:
            data = json.load(f)
        
        # Flatten the nested structure
        rows = []
        if isinstance(data, dict) and 'evaluations' in data:
            # Handle validation study format
            evaluations = data['evaluations']
            for model_name, model_evals in evaluations.items():
                for prompt_id, prompt_evals in model_evals.items():
                    for judge_name, evaluation in prompt_evals.items():
                        if isinstance(evaluation, dict) and 'error' not in evaluation:
                            evaluation['model_name'] = model_name
                            evaluation['judge_name'] = judge_name
                            rows.append(evaluation)
        else:
            # Handle direct list format
            rows = data if isinstance(data, list) else []
        
        if not rows:
            raise ValueError("No valid evaluations found in the file")
        
        return pd.DataFrame(rows)
    
    def load_from_database(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load annotations and evaluations from database."""
        db = Database()
        
        # Load human annotations
        human_data = db.get_human_annotations()
        human_df = pd.DataFrame(human_data)
        
        # Load LLM evaluations
        llm_data = db.get_all_evaluations()
        llm_df = pd.DataFrame(llm_data)
        
        return human_df, llm_df
    
    def calculate_human_human_agreement(self, human_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate inter-rater reliability between human annotators."""
        if not STATS_AVAILABLE:
            return {"error": "Statistical libraries not available"}
        
        agreement_stats = {}
        
        # Group by prompt_id to get multiple annotations per prompt
        prompt_groups = human_df.groupby('prompt_id')
        
        # Calculate pairwise agreements
        pairwise_agreements = []
        kappa_scores = []
        
        for prompt_id, group in prompt_groups:
            if len(group) < 2:
                continue
            
            annotators = group['annotator_id'].unique()
            
            # Calculate pairwise agreement for this prompt
            for i in range(len(annotators)):
                for j in range(i + 1, len(annotators)):
                    ann1_data = group[group['annotator_id'] == annotators[i]]
                    ann2_data = group[group['annotator_id'] == annotators[j]]
                    
                    if len(ann1_data) > 0 and len(ann2_data) > 0:
                        # Binary agreement (is_stereotype)
                        label1 = ann1_data['is_stereotype'].iloc[0]
                        label2 = ann2_data['is_stereotype'].iloc[0]
                        
                        pairwise_agreements.append({
                            'prompt_id': prompt_id,
                            'annotator_1': annotators[i],
                            'annotator_2': annotators[j],
                            'label_1': label1,
                            'label_2': label2,
                            'agreement': label1 == label2
                        })
        
        if not pairwise_agreements:
            return {"error": "No pairwise annotations found"}
        
        pairwise_df = pd.DataFrame(pairwise_agreements)
        
        # Overall agreement rate
        overall_agreement = pairwise_df['agreement'].mean()
        agreement_stats['overall_agreement_rate'] = overall_agreement
        
        # Calculate Cohen's Kappa for each pair
        annotator_pairs = pairwise_df[['annotator_1', 'annotator_2']].drop_duplicates()
        kappa_scores = []
        
        for _, pair in annotator_pairs.iterrows():
            pair_data = pairwise_df[
                (pairwise_df['annotator_1'] == pair['annotator_1']) & 
                (pairwise_df['annotator_2'] == pair['annotator_2'])
            ]
            
            if len(pair_data) > 0:
                labels1 = pair_data['label_1'].values
                labels2 = pair_data['label_2'].values
                
                try:
                    kappa = cohen_kappa_score(labels1, labels2)
                    kappa_scores.append({
                        'annotator_1': pair['annotator_1'],
                        'annotator_2': pair['annotator_2'],
                        'kappa': kappa,
                        'n_agreements': len(pair_data)
                    })
                except:
                    pass
        
        if kappa_scores:
            agreement_stats['kappa_scores'] = kappa_scores
            agreement_stats['mean_kappa'] = np.mean([k['kappa'] for k in kappa_scores])
            agreement_stats['std_kappa'] = np.std([k['kappa'] for k in kappa_scores])
        
        # Severity score correlation (if available)
        if 'severity_score' in human_df.columns:
            severity_correlations = []
            for prompt_id, group in prompt_groups:
                if len(group) >= 2:
                    severity_scores = group['severity_score'].dropna()
                    if len(severity_scores) >= 2:
                        # Calculate correlation between severity scores
                        severity_values = severity_scores.values
                        if len(np.unique(severity_values)) > 1:
                            correlation = np.corrcoef(severity_values)[0, 1] if len(severity_values) == 2 else np.mean(np.corrcoef(severity_values))
                            severity_correlations.append(correlation)
            
            if severity_correlations:
                agreement_stats['severity_correlation'] = {
                    'mean': np.mean(severity_correlations),
                    'std': np.std(severity_correlations),
                    'n_prompts': len(severity_correlations)
                }
        
        return agreement_stats
    
    def calculate_human_llm_agreement(self, human_df: pd.DataFrame, llm_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate agreement between human annotators and LLM judges."""
        if not STATS_AVAILABLE:
            return {"error": "Statistical libraries not available"}
        
        agreement_stats = {}
        
        # Match human annotations with LLM evaluations
        matches = []
        
        for _, human_row in human_df.iterrows():
            # Find matching LLM evaluations
            llm_matches = llm_df[
                (llm_df['prompt_id'] == human_row['prompt_id']) &
                (llm_df['response_text'] == human_row.get('response_text', ''))
            ]
            
            for _, llm_row in llm_matches.iterrows():
                matches.append({
                    'prompt_id': human_row['prompt_id'],
                    'annotator_id': human_row['annotator_id'],
                    'judge_name': llm_row['judge_name'],
                    'model_name': llm_row.get('model_name', 'unknown'),
                    'human_is_stereotype': human_row['is_stereotype'],
                    'llm_is_stereotype': llm_row['is_stereotype'],
                    'human_severity': human_row.get('severity_score', None),
                    'llm_severity': llm_row.get('severity_score', None),
                    'agreement': human_row['is_stereotype'] == llm_row['is_stereotype']
                })
        
        if not matches:
            return {"error": "No matching annotations found between human and LLM data"}
        
        matches_df = pd.DataFrame(matches)
        
        # Overall agreement rate
        overall_agreement = matches_df['agreement'].mean()
        agreement_stats['overall_agreement_rate'] = overall_agreement
        
        # Agreement by judge
        judge_agreements = matches_df.groupby('judge_name').agg({
            'agreement': ['mean', 'count'],
            'human_is_stereotype': 'sum',
            'llm_is_stereotype': 'sum'
        }).round(3)
        
        agreement_stats['by_judge'] = judge_agreements.to_dict()
        
        # Calculate Cohen's Kappa for each judge
        kappa_scores = []
        for judge_name in matches_df['judge_name'].unique():
            judge_data = matches_df[matches_df['judge_name'] == judge_name]
            
            try:
                kappa = cohen_kappa_score(
                    judge_data['human_is_stereotype'], 
                    judge_data['llm_is_stereotype']
                )
                kappa_scores.append({
                    'judge_name': judge_name,
                    'kappa': kappa,
                    'n_comparisons': len(judge_data)
                })
            except:
                pass
        
        agreement_stats['kappa_scores'] = kappa_scores
        
        # Severity score correlation
        if 'human_severity' in matches_df.columns and 'llm_severity' in matches_df.columns:
            severity_data = matches_df.dropna(subset=['human_severity', 'llm_severity'])
            if len(severity_data) > 0:
                correlation = stats.pearsonr(severity_data['human_severity'], severity_data['llm_severity'])
                agreement_stats['severity_correlation'] = {
                    'correlation': correlation[0],
                    'p_value': correlation[1],
                    'n_comparisons': len(severity_data)
                }
        
        return agreement_stats
    
    def calculate_llm_llm_agreement(self, llm_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate agreement between different LLM judges."""
        if not STATS_AVAILABLE:
            return {"error": "Statistical libraries not available"}
        
        agreement_stats = {}
        
        # Group by prompt_id and response_text to get different judge evaluations
        grouped = llm_df.groupby(['prompt_id', 'response_text'])
        
        pairwise_agreements = []
        
        for (prompt_id, response_text), group in grouped:
            if len(group) < 2:
                continue
            
            judges = group['judge_name'].unique()
            
            # Calculate pairwise agreement for this response
            for i in range(len(judges)):
                for j in range(i + 1, len(judges)):
                    judge1_data = group[group['judge_name'] == judges[i]]
                    judge2_data = group[group['judge_name'] == judges[j]]
                    
                    if len(judge1_data) > 0 and len(judge2_data) > 0:
                        label1 = judge1_data['is_stereotype'].iloc[0]
                        label2 = judge2_data['is_stereotype'].iloc[0]
                        
                        pairwise_agreements.append({
                            'prompt_id': prompt_id,
                            'judge_1': judges[i],
                            'judge_2': judges[j],
                            'label_1': label1,
                            'label_2': label2,
                            'agreement': label1 == label2,
                            'severity_1': judge1_data['severity_score'].iloc[0],
                            'severity_2': judge2_data['severity_score'].iloc[0]
                        })
        
        if not pairwise_agreements:
            return {"error": "No pairwise judge evaluations found"}
        
        pairwise_df = pd.DataFrame(pairwise_agreements)
        
        # Overall agreement rate
        overall_agreement = pairwise_df['agreement'].mean()
        agreement_stats['overall_agreement_rate'] = overall_agreement
        
        # Calculate Cohen's Kappa for each judge pair
        judge_pairs = pairwise_df[['judge_1', 'judge_2']].drop_duplicates()
        kappa_scores = []
        
        for _, pair in judge_pairs.iterrows():
            pair_data = pairwise_df[
                (pairwise_df['judge_1'] == pair['judge_1']) & 
                (pairwise_df['judge_2'] == pair['judge_2'])
            ]
            
            if len(pair_data) > 0:
                labels1 = pair_data['label_1'].values
                labels2 = pair_data['label_2'].values
                
                try:
                    kappa = cohen_kappa_score(labels1, labels2)
                    kappa_scores.append({
                        'judge_1': pair['judge_1'],
                        'judge_2': pair['judge_2'],
                        'kappa': kappa,
                        'n_agreements': len(pair_data)
                    })
                except:
                    pass
        
        if kappa_scores:
            agreement_stats['kappa_scores'] = kappa_scores
            agreement_stats['mean_kappa'] = np.mean([k['kappa'] for k in kappa_scores])
        
        # Severity score correlation
        severity_data = pairwise_df.dropna(subset=['severity_1', 'severity_2'])
        if len(severity_data) > 0:
            correlation = stats.pearsonr(severity_data['severity_1'], severity_data['severity_2'])
            agreement_stats['severity_correlation'] = {
                'correlation': correlation[0],
                'p_value': correlation[1],
                'n_comparisons': len(severity_data)
            }
        
        return agreement_stats
    
    def create_confusion_matrices(self, human_df: pd.DataFrame, llm_df: pd.DataFrame) -> Dict[str, Any]:
        """Create confusion matrices for different comparisons."""
        if not STATS_AVAILABLE:
            return {"error": "Statistical libraries not available"}
        
        matrices = {}
        
        # Human-LLM confusion matrices
        matches = []
        for _, human_row in human_df.iterrows():
            llm_matches = llm_df[
                (llm_df['prompt_id'] == human_row['prompt_id']) &
                (llm_df['response_text'] == human_row.get('response_text', ''))
            ]
            
            for _, llm_row in llm_matches.iterrows():
                matches.append({
                    'human_label': human_row['is_stereotype'],
                    'llm_label': llm_row['is_stereotype'],
                    'judge_name': llm_row['judge_name']
                })
        
        if matches:
            matches_df = pd.DataFrame(matches)
            
            # Create confusion matrix for each judge
            for judge_name in matches_df['judge_name'].unique():
                judge_data = matches_df[matches_df['judge_name'] == judge_name]
                
                cm = confusion_matrix(judge_data['human_label'], judge_data['llm_label'])
                
                matrices[f'human_vs_{judge_name}'] = {
                    'matrix': cm.tolist(),
                    'labels': ['Not Stereotype', 'Stereotype'],
                    'classification_report': classification_report(
                        judge_data['human_label'], 
                        judge_data['llm_label'],
                        output_dict=True
                    )
                }
        
        return matrices
    
    def generate_visualizations(self, human_df: pd.DataFrame, llm_df: pd.DataFrame):
        """Generate visualization plots for agreement analysis."""
        if not STATS_AVAILABLE:
            return
        
        # Create agreement heatmap
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Agreement Analysis Visualizations', fontsize=16)
            
            # Human annotation distribution
            if len(human_df) > 0:
                human_stereotype_rate = human_df['is_stereotype'].mean()
                axes[0, 0].bar(['Not Stereotype', 'Stereotype'], 
                              [1 - human_stereotype_rate, human_stereotype_rate])
                axes[0, 0].set_title('Human Annotation Distribution')
                axes[0, 0].set_ylabel('Proportion')
            
            # LLM evaluation distribution
            if len(llm_df) > 0:
                llm_stereotype_rate = llm_df['is_stereotype'].mean()
                axes[0, 1].bar(['Not Stereotype', 'Stereotype'], 
                              [1 - llm_stereotype_rate, llm_stereotype_rate])
                axes[0, 1].set_title('LLM Evaluation Distribution')
                axes[0, 1].set_ylabel('Proportion')
            
            # Severity score distributions
            if 'severity_score' in human_df.columns and len(human_df) > 0:
                axes[1, 0].hist(human_df['severity_score'].dropna(), bins=10, alpha=0.7)
                axes[1, 0].set_title('Human Severity Score Distribution')
                axes[1, 0].set_xlabel('Severity Score')
                axes[1, 0].set_ylabel('Frequency')
            
            if 'severity_score' in llm_df.columns and len(llm_df) > 0:
                axes[1, 1].hist(llm_df['severity_score'].dropna(), bins=10, alpha=0.7)
                axes[1, 1].set_title('LLM Severity Score Distribution')
                axes[1, 1].set_xlabel('Severity Score')
                axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'agreement_visualizations.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualizations saved to: {os.path.join(self.output_dir, 'agreement_visualizations.png')}")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    def generate_report(self, output_path: str):
        """Generate a comprehensive agreement report."""
        
        # Create summary statistics
        summary = {
            "total_analyses": len([k for k in self.results.keys() if k.endswith('_agreement')]),
            "analyses_completed": [k for k in self.results.keys() if k.endswith('_agreement') and 'error' not in self.results[k]],
            "timestamp": self.results["analysis_timestamp"]
        }
        
        self.results["summary_statistics"] = summary
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Agreement analysis report saved to: {output_path}")
        
        # Create human-readable summary
        summary_path = output_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("AGREEMENT ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Human-Human Agreement
            if 'human_human_agreement' in self.results and 'error' not in self.results['human_human_agreement']:
                hha = self.results['human_human_agreement']
                f.write("HUMAN-HUMAN AGREEMENT:\n")
                f.write(f"  Overall Agreement Rate: {hha.get('overall_agreement_rate', 'N/A'):.3f}\n")
                if 'mean_kappa' in hha:
                    f.write(f"  Mean Cohen's Kappa: {hha['mean_kappa']:.3f}\n")
                f.write("\n")
            
            # Human-LLM Agreement
            if 'human_llm_agreement' in self.results and 'error' not in self.results['human_llm_agreement']:
                hla = self.results['human_llm_agreement']
                f.write("HUMAN-LLM AGREEMENT:\n")
                f.write(f"  Overall Agreement Rate: {hla.get('overall_agreement_rate', 'N/A'):.3f}\n")
                if 'kappa_scores' in hla:
                    f.write("  Cohen's Kappa by Judge:\n")
                    for kappa_info in hla['kappa_scores']:
                        f.write(f"    {kappa_info['judge_name']}: {kappa_info['kappa']:.3f}\n")
                f.write("\n")
            
            # LLM-LLM Agreement
            if 'llm_llm_agreement' in self.results and 'error' not in self.results['llm_llm_agreement']:
                lla = self.results['llm_llm_agreement']
                f.write("LLM-LLM AGREEMENT:\n")
                f.write(f"  Overall Agreement Rate: {lla.get('overall_agreement_rate', 'N/A'):.3f}\n")
                if 'mean_kappa' in lla:
                    f.write(f"  Mean Cohen's Kappa: {lla['mean_kappa']:.3f}\n")
                f.write("\n")
        
        print(f"Human-readable summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate inter-rater reliability and agreement statistics"
    )
    parser.add_argument(
        "--human-annotations",
        help="Path to human annotations CSV/JSON file"
    )
    parser.add_argument(
        "--llm-evaluations",
        help="Path to LLM evaluations JSON file"
    )
    parser.add_argument(
        "--database",
        action="store_true",
        help="Load data from database instead of files"
    )
    parser.add_argument(
        "--output",
        default="agreement_analysis.json",
        help="Output path for analysis results"
    )
    parser.add_argument(
        "--output-dir",
        default="agreement_analysis",
        help="Directory for output files and visualizations"
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualization plots"
    )
    
    args = parser.parse_args()
    
    if not STATS_AVAILABLE:
        print("Warning: Statistical libraries not available. Limited analysis will be performed.")
        print("Install with: pip install scipy scikit-learn seaborn matplotlib")
    
    # Initialize calculator
    calculator = AgreementCalculator(args.output_dir)
    
    # Load data
    if args.database:
        print("Loading data from database...")
        human_df, llm_df = calculator.load_from_database()
    else:
        if not args.human_annotations:
            parser.error("--human-annotations is required when not using --database")
        if not args.llm_evaluations:
            parser.error("--llm-evaluations is required when not using --database")
        
        print("Loading data from files...")
        human_df = calculator.load_human_annotations(args.human_annotations)
        llm_df = calculator.load_llm_evaluations(args.llm_evaluations)
    
    print(f"Loaded {len(human_df)} human annotations and {len(llm_df)} LLM evaluations")
    
    # Calculate agreements
    print("Calculating human-human agreement...")
    calculator.results['human_human_agreement'] = calculator.calculate_human_human_agreement(human_df)
    
    print("Calculating human-LLM agreement...")
    calculator.results['human_llm_agreement'] = calculator.calculate_human_llm_agreement(human_df, llm_df)
    
    print("Calculating LLM-LLM agreement...")
    calculator.results['llm_llm_agreement'] = calculator.calculate_llm_llm_agreement(llm_df)
    
    print("Creating confusion matrices...")
    calculator.results['agreement_matrices'] = calculator.create_confusion_matrices(human_df, llm_df)
    
    # Generate visualizations
    if not args.no_visualizations and STATS_AVAILABLE:
        print("Generating visualizations...")
        calculator.generate_visualizations(human_df, llm_df)
    
    # Generate report
    print("Generating report...")
    calculator.generate_report(args.output)
    
    print(f"\nAgreement analysis complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main()