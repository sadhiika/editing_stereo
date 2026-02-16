import asyncio
import click
import json
import os
from typing import Dict, Any

# Ensure you have petri installed: pip install git+https://github.com/safety-research/petri.git
from petri import an_inspect_run
from petri.core.endpoints import ModelEndpoint
from petri.core.prompt import Prompt
from petri.core.transcript import Transcript

from .data_loader import load_prompts

# --- Default Configuration ---
DEFAULT_AUDITOR_MODEL = "anthropic/claude-3-haiku-20240307"
DEFAULT_TARGET_MODEL = "anthropic/claude-3-opus-20240229"
DEFAULT_MAX_TURNS = 2  # Keep low for simple response generation

async def run_audit_for_prompt(prompt_text: str, target_model: str) -> str:
    """
    Configures and runs a petri audit for a single prompt and returns the target's first response.
    """
    # Simplified configuration for generating a single response
    auditor_ep = ModelEndpoint(name=DEFAULT_AUDITOR_MODEL)
    target_ep = ModelEndpoint(name=target_model)

    # The initial prompt for the audit is the one from our file
    initial_prompt = Prompt(role="user", content=prompt_text)

    # Run the audit for a few turns
    transcript: Transcript = await an_inspect_run(
        auditor=auditor_ep,
        target=target_ep,
        max_turns=DEFAULT_MAX_TURNS,
        initial_prompt=initial_prompt,
        special_instructions=None,  # Not needed for this use case
        judge=None, # We are not using petri's judging feature
        safety_policy=None,
    )

    # Find the first response from the target model
    for message in transcript.history:
        if message.role == "assistant":
            return message.content
    
    return "" # Return empty if no response found

@click.command()
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
    default=DEFAULT_TARGET_MODEL, 
    show_default=True, 
    help="The target model to be audited by petri."
)
def main(prompts_path: str, output_dir: str, target_model: str):
    """
    Runs petri audits for each prompt in the prompts file and saves the
    target's responses in a format compatible with stereowipe's benchmark CLI.
    """
    click.echo(f"Loading prompts from {prompts_path}...")
    prompts_data = load_prompts(prompts_path)
    if prompts_data is None:
        click.echo("Failed to load prompts file. It may be missing or contain invalid JSON.", err=True)
        return
    if not prompts_data:
        click.echo("Prompts file is empty or contains no valid prompts. No audits to run.", err=True)
        return

    model_responses: Dict[str, Dict[str, str]] = {}
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    model_name_safe = target_model.replace("/", "_")
    output_path = os.path.join(output_dir, f"{model_name_safe}.json")

    click.echo(f"Running audits for {len(prompts_data)} prompts. This may take a while...")
    
    # Using a context manager for the event loop
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If in an environment like Jupyter where a loop is already running
        for prompt_id, prompt_info in prompts_data.items():
            prompt_text = prompt_info['text']
            click.echo(f"  - Auditing prompt: '{prompt_text[:50]}...'")
            response = loop.run_until_complete(run_audit_for_prompt(prompt_text, target_model))
            
            if prompt_text not in model_responses:
                model_responses[prompt_text] = {}
            model_responses[prompt_text][model_name_safe] = response
    else:
        # For standard script execution
        for prompt_id, prompt_info in prompts_data.items():
            prompt_text = prompt_info['text']
            click.echo(f"  - Auditing prompt: '{prompt_text[:50]}...'")
            response = asyncio.run(run_audit_for_prompt(prompt_text, target_model))

            if prompt_text not in model_responses:
                model_responses[prompt_text] = {}
            model_responses[prompt_text][model_name_safe] = response


    click.echo(f"Saving responses for model '{model_name_safe}' to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(model_responses, f, indent=4)

    click.echo("Petri audit run complete.")

if __name__ == '__main__':
    # Required environment variables for petri, e.g., ANTHROPIC_API_KEY
    main()