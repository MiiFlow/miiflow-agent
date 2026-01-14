"""Environment variable utilities."""

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def load_env_file(env_path: Optional[str] = None) -> None:
    """Load environment variables from .env file using python-dotenv.

    Falls back to basic parsing if python-dotenv is not installed.
    """
    if DOTENV_AVAILABLE:
        load_dotenv(env_path)
    else:
        # Fallback: basic parsing (does not handle edge cases)
        from pathlib import Path
        if env_path is None:
            env_path = Path.cwd() / ".env"
        else:
            env_path = Path(env_path)

        if not env_path.exists():
            return

        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for provider from environment."""
    key_map = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY', 
        'groq': 'GROQ_API_KEY',
        'xai': 'XAI_API_KEY',
        'gemini': 'GEMINI_API_KEY',
        'openrouter': 'OPENROUTER_API_KEY',
        'mistral': 'MISTRAL_API_KEY',
        'deepseek': 'DEEPSEEK_API_KEY',
        'ollama': 'OLLAMA_API_KEY',  # Optional for local Ollama
    }
    
    env_var = key_map.get(provider.lower())
    if not env_var:
        return None
    
    # For Ollama, API key is optional (local usage)
    if provider.lower() == 'ollama':
        return os.getenv(env_var, "")  # Return empty string instead of None
        
    return os.getenv(env_var)