import click
import os
import json
from typing import List, Optional
from tqdm import tqdm

from biaswipe import data_loader, scoring, report
from biaswipe.judge import Judge, MockJudge, OpenAIJudge, AnthropicJudge, MissingApiKeyError

try:
    from biaswipe.run_petri_audit import main as petri_main
except ImportError as e:
    # Petri might not be installed or compatible
    petri_main = None
    # We delay the warning until the command is actually used, or we could log it here debug-style.

@click.group()
def cli():
    """Main entry point for the BiasWipe CLI."""
    pass

@cli.command("run-benchmark")
@click.option(
    '--prompts', 'prompts_path',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the JSON file containing prompts."
)
@click.option(
    '--annotations', 'annotations_path',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the JSON file containing human annotations."
)
@click.option(
    '--model-responses-dir', 'model_responses_dir_path',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help="Path to the directory containing model response JSON files."
)
@click.option(
    '--category-weights', 'category_weights_path',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    help="Path to the JSON file containing category weights for WOSI calculation."
)
@click.option(
    '--report-output', 'report_output_path',
    type=click.Path(dir_okay=False, writable=True),
    default='./report.json',
    show_default=True,
    help="Path to save the generated JSON report."
)
@click.option(
    '--judges', 'judges_str',
    type=str,
    default='mock',
    show_default=True,
    help='Comma-separated list of judge names to use (e.g., "openai,anthropic", "mock", "openai"). Available: mock, openai, anthropic.'
)
@click.option(
    '--no-db',
    is_flag=True,
    default=False,
    help='Disable saving evaluation results to the database.'
)
@click.option(
    '--quiet', 'quiet',
    is_flag=True,
    default=False,
    help='Disable progress bars and reduce output verbosity.'
)
def run_benchmark(
    prompts_path: str,
    annotations_path: str,
    model_responses_dir_path: str,
    category_weights_path: Optional[str],
    report_output_path: str,
    judges_str: str,
    no_db: bool,
    quiet: bool
):
    """
    Runs the BiasWipe benchmark to score models for stereotyping.
    """
    if not quiet:
        click.echo("Starting BiasWipe benchmark process...")
    else:
        click.echo("Running BiasWipe benchmark...")

    # --- Instantiate Judges ---
    selected_judges: List[Judge] = []
    judge_names = [name.strip().lower() for name in judges_str.split(',')]

    available_judge_constructors = {
        "mock": MockJudge,
        "openai": OpenAIJudge,
        "anthropic": AnthropicJudge
    }

    if not quiet:
        click.echo(f"Selected judge models: {', '.join(judge_names)}")
    
    # Progress bar for judge initialization
    judge_progress = tqdm(judge_names, desc="Initializing judges", disable=quiet, leave=True)
    for name in judge_progress:
        if name in available_judge_constructors:
            try:
                # You could add specific args here if needed, e.g. model variants
                if name == "mock":
                     selected_judges.append(MockJudge(name=f"CLI_{name}")) # Give CLI mocks distinct name
                else:
                    selected_judges.append(available_judge_constructors[name]())
                if not quiet:
                    click.echo(f"Successfully instantiated judge: {name}")
            except MissingApiKeyError as e:
                if not quiet:
                    click.echo(f"Warning: Could not instantiate judge '{name}' due to Missing API Key: {e}. This judge will be skipped.")
                else:
                    click.echo(f"Warning: Missing API key for judge '{name}'")
            except Exception as e:
                if not quiet:
                    click.echo(f"Warning: Could not instantiate judge '{name}' due to an unexpected error: {e}. This judge will be skipped.")
                else:
                    click.echo(f"Warning: Failed to instantiate judge '{name}'")
        else:
            if not quiet:
                click.echo(f"Warning: Unknown judge name '{name}' provided. It will be skipped. Available: {', '.join(available_judge_constructors.keys())}")
            else:
                click.echo(f"Warning: Unknown judge '{name}'")

    if not selected_judges:
        if not quiet:
            click.echo("Warning: No valid judges were instantiated (or none were selected). Defaulting to a single MockJudge.")
        else:
            click.echo("Warning: No valid judges. Using MockJudge.")
        selected_judges.append(MockJudge(name="CLI_DefaultMock"))

    if not quiet:
        click.echo(f"Using judges: {[type(j).__name__ + (f'({j.name})' if hasattr(j, 'name') else '') for j in selected_judges]}")

    # --- Load Data ---
    if not quiet:
        click.echo("Loading data files...")
    
    with tqdm(total=3, desc="Loading data", disable=quiet, leave=True) as pbar:
        prompts = data_loader.load_prompts(prompts_path)
        pbar.update(1)
        if not prompts:
            click.echo(f"Failed to load prompts from {prompts_path}. Exiting.")
            return

        annotations = data_loader.load_annotations(annotations_path) # Not used in scoring currently, but loaded
        pbar.update(1)
        if not annotations:
            if not quiet:
                click.echo(f"Warning: Failed to load annotations from {annotations_path}. This may affect parts of the benchmark not yet implemented.")

        category_weights = {}
        if category_weights_path:
            if not quiet:
                click.echo(f"Loading category weights from: {category_weights_path}")
            category_weights = data_loader.load_json_data(category_weights_path)
            if not category_weights:
                if not quiet:
                    click.echo(f"Warning: Failed to load category weights from '{category_weights_path}' or the file is empty.")
        else:
            if not quiet:
                click.echo("No category weights file provided. WOSI may be 0.0 or based on unweighted CSSS.")
        pbar.update(1)

    # --- Process Model Responses ---
    all_model_results = {}
    if not quiet:
        click.echo(f"Loading model responses from: {model_responses_dir_path}")
    
    try:
        # Get list of JSON files
        json_files = [f for f in os.listdir(model_responses_dir_path) if f.endswith(".json")]
        
        # Progress bar for processing model files
        file_progress = tqdm(json_files, desc="Processing models", disable=quiet, leave=True)
        for filename in file_progress:
            model_response_file_path = os.path.join(model_responses_dir_path, filename)
            model_name = os.path.splitext(filename)[0]
            
            # Update progress bar description
            file_progress.set_description(f"Processing {model_name}")
            
            if not quiet:
                click.echo(f"\nProcessing model response file: {model_response_file_path}")
            
            model_responses = data_loader.load_model_responses(model_response_file_path)

            if model_responses:
                if not quiet:
                    click.echo(f"Scoring responses for model: {model_name}...")

                scores = scoring.score_model_responses(
                    prompts=prompts,
                    model_responses=model_responses,
                    category_weights=category_weights,
                    judges=selected_judges, # Pass the instantiated judges
                    model_name=model_name,  # Pass model name for database storage
                    save_to_db=not no_db,  # Save to database unless --no-db flag is set
                    quiet=quiet  # Pass quiet parameter for progress control
                )
                all_model_results[model_name] = scores
                
                if not quiet:
                    click.echo(f"Finished scoring for model: {model_name}. Stereotype Rate (SR): {scores.get('SR', 'N/A')}, Stereotype Severity Score (SSS): {scores.get('SSS', 'N/A')}, Weighted Overall Stereotyping Index (WOSI): {scores.get('WOSI', 'N/A')}")
                else:
                    # In quiet mode, just show summary results
                    sr_val = scores.get('SR', 'N/A')
                    sss_val = scores.get('SSS', 'N/A')
                    wosi_val = scores.get('WOSI', 'N/A')
                    # Format numeric values if they exist
                    sr_str = f"{sr_val:.3f}" if isinstance(sr_val, (int, float)) else sr_val
                    sss_str = f"{sss_val:.3f}" if isinstance(sss_val, (int, float)) else sss_val
                    wosi_str = f"{wosi_val:.3f}" if isinstance(wosi_val, (int, float)) else wosi_val
                    click.echo(f"{model_name}: SR={sr_str}, SSS={sss_str}, WOSI={wosi_str}")
            else:
                if not quiet:
                    click.echo(f"Warning: Could not load responses from {model_response_file_path} or file is empty. Skipping.")
    except OSError as e:
        click.echo(f"Error reading from model responses directory {model_responses_dir_path}: {e}. Exiting.")
        return

    # --- Generate Report ---
    if not all_model_results:
        click.echo("No model responses were successfully processed. Skipping report generation.")
    else:
        if not quiet:
            click.echo("\nGenerating benchmark report...")
        report.generate_report(all_model_results, report_output_path)

    if not quiet:
        click.echo(f"\nBiasWipe benchmark process completed. Report available at: {report_output_path if all_model_results else 'N/A'}")
    else:
        click.echo(f"\nCompleted. Report: {report_output_path if all_model_results else 'N/A'}")

@cli.command("generate-responses")
@click.option(
    '--prompts-path', 
    type=click.Path(exists=True, dir_okay=False), 
    required=True, 
    help="Path to the prompts.json file."
)
@click.option(
    '--output-dir', 
    type=click.Path(file_okay=False, writable=True), 
    required=True, 
    help="Directory to save the generated model responses."
)
@click.option(
    '--target-model', 
    type=str, 
    default="anthropic/claude-3-opus-20240229", 
    show_default=True, 
    help="The target model to be audited by petri."
)
def generate_responses(prompts_path: str, output_dir: str, target_model: str):
    """
    Generates model responses for a set of prompts using the petri audit tool.
    This is a wrapper around the `run_petri_audit.py` script.
    """
    if petri_main is None:
        click.echo("Error: The 'petri' library or 'biaswipe.run_petri_audit' module could not be loaded. Please ensure petri is installed correctly.", err=True)
        return

    click.echo("--- Starting Petri Response Generation ---")
    
    # Manually create a context for the petri_main command
    # We need to construct the args list carefully
    args = [
        '--prompts-path', prompts_path,
        '--output-dir', output_dir,
        '--target-model', target_model
    ]
    
    # Execute the command
    try:
        # We use make_context and invoke to run the command within the current process
        ctx = petri_main.make_context(
            info_name='run-petri', 
            args=args,
            parent=click.get_current_context()
        )
        with ctx:
            petri_main.invoke(ctx)
    except Exception as e:
        click.echo(f"Error running Petri audit: {e}", err=True)
    
    click.echo("--- Finished Petri Response Generation ---")

if __name__ == '__main__':
    cli()