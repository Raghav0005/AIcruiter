import json
import os
from pathlib import Path

def load_api_keys(key_name=None):
    """
    Load API keys from the keys.json file.
    
    Args:
        key_name (str, optional): Specific key to retrieve. If None, returns all keys.
        
    Returns:
        dict or str: All keys as dict or specific key value
    """
    try:
        # Use Path for cross-platform compatibility
        file_path = Path(__file__).parent / 'keys.json'
        with open(file_path, 'r') as f:
            keys = json.load(f)
        
        if key_name:
            return keys.get(key_name)
        return keys
    except Exception as e:
        print(f"Error loading API keys: {str(e)}")
        return {} if key_name is None else None
