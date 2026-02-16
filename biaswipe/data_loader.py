import json
from typing import Dict, Any, Optional

def load_prompts(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads prompts from a JSON file.

    Args:
        file_path: Path to the JSON file containing prompts.

    Returns:
        A dictionary where keys are prompt IDs and values are dictionaries containing prompt data
        (e.g., {"text": "...", "category": "..."}),
        or None if a major error occurs (file not found, JSON decode error).
    """
    try:
        with open(file_path, 'r') as f:
            prompts_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prompts file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from prompts file at {file_path}")
        return None

    if not isinstance(prompts_data, dict):
        print(f"Error: Prompts file at {file_path} does not contain a valid JSON object (dictionary).")
        return None

    processed_prompts = {}
    for prompt_id, prompt_content in prompts_data.items():
        if not isinstance(prompt_content, dict):
            print(f"Warning: Prompt ID '{prompt_id}' in {file_path} is not a valid dictionary. Skipping.")
            continue
        
        text = prompt_content.get("text")
        category = prompt_content.get("category")

        if text is None:
            print(f"Warning: Prompt ID '{prompt_id}' in {file_path} is missing 'text' key. Skipping.")
            continue
        if category is None:
            print(f"Warning: Prompt ID '{prompt_id}' in {file_path} is missing 'category' key. Skipping.")
            continue
        
        if not isinstance(text, str):
            print(f"Warning: Prompt ID '{prompt_id}' in {file_path} has a 'text' value that is not a string. Skipping.")
            continue
        if not isinstance(category, str):
             print(f"Warning: Prompt ID '{prompt_id}' in {file_path} has a 'category' value that is not a string. Skipping.")
             continue
            
        processed_prompts[prompt_id] = {"text": text, "category": category}
    return processed_prompts

def load_json_data(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Generic function to load JSON data from a file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        A dictionary containing the loaded JSON data, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
             print(f"Error: JSON file at {file_path} does not contain a valid JSON object (dictionary).")
             return None
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None

def load_annotations(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads human annotations from a JSON file.

    Args:
        file_path: Path to the JSON file containing annotations.

    Returns:
        A dictionary containing the loaded annotations,
        or None if an error occurs.
    """
    return load_json_data(file_path)

def load_model_responses(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads model responses from a JSON file.

    Args:
        file_path: Path to the JSON file containing model responses.

    Returns:
        A dictionary mapping prompt ID to response text, or None if an error occurs.
    """
    return load_json_data(file_path)