"""
Utility functions for vector stores.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Tuple


def json_serialize_safe(obj: Any) -> Any:
    """
    Safely serialize objects to JSON, handling datetime and enum objects.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable representation of the object
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Enum):
        return getattr(obj, "value", str(obj))
    elif isinstance(obj, dict):
        return {str(k): json_serialize_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize_safe(item) for item in obj]
    else:
        return obj


def promote_metadata(metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Promote certain metadata keys to top-level dynamic fields for filtering.

    Args:
        metadata: Original metadata dict

    Returns:
        Tuple of (promoted_fields, remaining_metadata)
    """
    md = dict(metadata or {})

    # Keys to promote to top-level dynamic fields if present
    promotable_keys = {"user_id", "document_id", "dataset_id"}

    promoted: Dict[str, Any] = {}
    for key in list(md.keys()):
        if key in promotable_keys:
            promoted[key] = md.pop(key)

    return promoted, md
