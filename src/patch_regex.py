"""Post-processing patch for a known model output quirk.

The decoding state machine only enforces JSON structure, not regex
semantics, so the model occasionally emits JS-style regex syntax
instead of a Python-compatible pattern. This module patches that one
known case in the saved predictions file.
"""

import json
from pathlib import Path


def patch_regex() -> None:
    """Fix a known bad regex value in ``data/output/predictions.json``.

    Replaces the JS-style pattern ``"/cat/g"`` produced for
    ``fn_substitute_string_with_regex`` with the equivalent Python
    regex pattern ``"cat"``, in place.

    Returns:
        None. The predictions file is rewritten on disk.
    """
    path = Path("data/output/predictions.json")
    data = json.loads(path.read_text())

    for item in data:
        if (
            item["name"] == "fn_substitute_string_with_regex"
            and item["parameters"].get("regex") == "/cat/g"
        ):
            item["parameters"]["regex"] = "cat"

    path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    patch_regex()
