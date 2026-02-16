#!/usr/bin/env python3
"""
Example Workflow Script

This script demonstrates how to use all the StereoWipe batch processing scripts
together in a complete research workflow.

Usage:
    python scripts/example_workflow.py [--quick] [--output-dir results/]
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

def run_command(cmd, description, check=True):
    """Run a shell command with error handling."""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def create_workflow_config(output_dir):
    """Create configuration files for the workflow."""
    config_dir = os.path.join(output_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    # Create validation study config
    validation_config = {
        "study_name": "Example Validation Study",
        "description": "Demonstration of StereoWipe validation workflow",
        "prompts_path": "sample_data/prompts.json",
        "models": ["gpt-4", "claude-3", "gemini-pro"],
        "judges": ["openai", "anthropic"],
        "sample_size": 20,  # Small sample for demo
        "annotations_per_response": 2,
        "output_format": "json",
        "export_formats": ["csv", "json"],
        "study_parameters": {
            "randomize_order": True,
            "blind_annotation": True,
            "include_mock_responses": False
        }
    }
    
    config_path = os.path.join(config_dir, "validation_config.json")
    with open(config_path, 'w') as f:
        json.dump(validation_config, f, indent=2)
    
    print(f"Created validation config: {config_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(
        description="Run a complete StereoWipe research workflow demonstration"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick demo with minimal data"
    )
    parser.add_argument(
        "--output-dir",
        default="example_workflow_results",
        help="Directory for all workflow outputs"
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip sample data setup step"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation study step"
    )
    parser.add_argument(
        "--skip-agreement",
        action="store_true",
        help="Skip agreement analysis step"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip data export step"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up paths
    scripts_dir = Path(__file__).parent
    project_root = scripts_dir.parent
    
    print(f"StereoWipe Research Workflow Example")
    print(f"{'='*50}")
    print(f"Output directory: {output_dir}")
    print(f"Scripts directory: {scripts_dir}")
    print(f"Project root: {project_root}")
    print(f"Quick mode: {args.quick}")
    
    # Change to project root directory
    os.chdir(project_root)
    
    workflow_steps = []
    
    # Step 1: Setup Sample Data
    if not args.skip_setup:
        print(f"\n🔧 Setting up sample data...")
        
        sample_size = 25 if args.quick else 50
        models = "gpt-4,claude-3" if args.quick else "gpt-4,claude-3,gemini-pro,llama-2"
        
        setup_cmd = (
            f"python scripts/setup_sample_data.py "
            f"--num-samples {sample_size} "
            f"--models {models} "
            f"--output-dir {output_dir}/sample_data "
            f"--quiet"
        )
        
        if run_command(setup_cmd, "Setting up sample data"):
            workflow_steps.append("✅ Sample data setup completed")
        else:
            workflow_steps.append("❌ Sample data setup failed")
            print("Warning: Sample data setup failed, using existing data")
    
    # Step 2: Run Validation Study
    if not args.skip_validation:
        print(f"\n📊 Running validation study...")
        
        # Create config
        config_path = create_workflow_config(output_dir)
        
        # Update config for quick mode
        if args.quick:
            with open(config_path, 'r') as f:
                config = json.load(f)
            config["sample_size"] = 10
            config["models"] = ["gpt-4", "claude-3"]
            config["judges"] = ["openai"]  # Use fewer judges for quick mode
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        validation_cmd = (
            f"python scripts/run_validation_study.py "
            f"--config {config_path} "
            f"--output {output_dir}/validation_results.json"
        )
        
        if run_command(validation_cmd, "Running validation study"):
            workflow_steps.append("✅ Validation study completed")
        else:
            workflow_steps.append("❌ Validation study failed")
    
    # Step 3: Calculate Agreement
    if not args.skip_agreement:
        print(f"\n📈 Calculating agreement statistics...")
        
        agreement_cmd = (
            f"python scripts/calculate_agreement.py "
            f"--database "
            f"--output {output_dir}/agreement_analysis.json "
            f"--output-dir {output_dir}/agreement_analysis"
        )
        
        if run_command(agreement_cmd, "Calculating agreement statistics"):
            workflow_steps.append("✅ Agreement analysis completed")
        else:
            workflow_steps.append("❌ Agreement analysis failed")
    
    # Step 4: Export Research Data
    if not args.skip_export:
        print(f"\n📤 Exporting research data...")
        
        export_format = "csv" if args.quick else "all"
        
        export_cmd = (
            f"python scripts/export_research_data.py "
            f"--format {export_format} "
            f"--output {output_dir}/research_data "
            f"--base-name stereowipe_example"
        )
        
        if not args.quick:
            export_cmd += " --create-archive"
        
        if run_command(export_cmd, "Exporting research data"):
            workflow_steps.append("✅ Research data export completed")
        else:
            workflow_steps.append("❌ Research data export failed")
    
    # Generate workflow summary
    summary = {
        "workflow_name": "StereoWipe Example Workflow",
        "timestamp": datetime.now().isoformat(),
        "output_directory": output_dir,
        "quick_mode": args.quick,
        "steps_completed": workflow_steps,
        "files_created": []
    }
    
    # List created files
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.relpath(os.path.join(root, file), output_dir)
                summary["files_created"].append(file_path)
    
    # Save summary
    summary_path = os.path.join(output_dir, "workflow_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n🎉 WORKFLOW COMPLETE!")
    print(f"{'='*50}")
    print(f"Output directory: {output_dir}")
    print(f"")
    print(f"Steps completed:")
    for step in workflow_steps:
        print(f"  {step}")
    print(f"")
    print(f"Files created: {len(summary['files_created'])}")
    print(f"Workflow summary: {summary_path}")
    
    # Print next steps
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Review the generated files in: {output_dir}")
    print(f"2. Check the validation results: {output_dir}/validation_results.json")
    print(f"3. Examine agreement analysis: {output_dir}/agreement_analysis.json")
    print(f"4. Use exported data for your research: {output_dir}/research_data/")
    
    if not args.quick:
        print(f"5. Extract the research archive for sharing")
    
    print(f"\n📖 For more information, see: scripts/README.md")

if __name__ == "__main__":
    main()