from uuid import UUID
from typing import Union

def validate_uuid(value: Union[str, UUID]) -> bool:
    """Validate if value is a valid UUID (hex or dashed) or human-readable ID"""
    if not value:
        return False

    # # OLD LOGIC (Strict UUID + local-)
    # try:
    #     UUID(str(value))
    #     return True
    # except ValueError:
    #     if str(value).startswith("local-"):
    #         return True
    #     return False

    # NEW LOGIC (Relaxed: UUID or alphanumeric with dashes/underscores)
    val_str = str(value)
    try:
        UUID(val_str)
        return True
    except ValueError:
        # Allow alphanumeric, dashes, underscores (e.g. test-agent-01)
        import re
        if re.match(r"^[a-zA-Z0-9\-_]+$", val_str):
            return True
        return False
