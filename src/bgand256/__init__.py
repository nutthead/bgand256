"""Background and 256 Colors - Find 256 foreground colors with good contrast"""

__version__ = "0.1.0"
__author__ = "Behrang Saeedzadeh"
__email__ = "hello@behrang.org"

from .color_utils import format_color_output, parse_color
from .colors import generate_readable_colors
from .contrast_algorithms import AlgorithmType, generate_contrasting_colors

__all__ = [
    "generate_readable_colors",
    "generate_contrasting_colors",
    "AlgorithmType",
    "parse_color",
    "format_color_output",
]
