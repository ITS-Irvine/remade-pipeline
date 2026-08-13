# config/path_template.py
from pathlib import Path

class PathTemplate(str):
    """Marker type: a string that should be resolved as a template and
    converted to a Path after all config values are loaded."""
    pass

class QuantityTemplate(str):
    """Marker type: a string that should be resolved and
    converted to a Quantity after all config values are loaded."""
    pass
