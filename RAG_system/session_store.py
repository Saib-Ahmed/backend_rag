import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def load_sessions(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load sessions from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        logger.error("Failed to decode session file: %s", e)
        return {}
    except Exception as e:
        logger.error("Error loading sessions from %s: %s", file_path, e)
        return {}

def save_sessions(file_path: str, sessions: Dict[str, List[Dict[str, Any]]]) -> None:
    """Save sessions to a JSON file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Error saving sessions to %s: %s", file_path, e)
