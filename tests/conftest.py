"""Test configuration and fixtures for bgand256 tests."""

import pytest
import numpy as np
from typing import List, Tuple


@pytest.fixture
def sample_colors() -> List[Tuple[float, float, float]]:
    """Provide sample RGB colors for testing."""
    return [
        (0.0, 0.0, 0.0),      # Black
        (1.0, 1.0, 1.0),      # White
        (1.0, 0.0, 0.0),      # Red
        (0.0, 1.0, 0.0),      # Green
        (0.0, 0.0, 1.0),      # Blue
        (0.5, 0.5, 0.5),      # Gray
        (1.0, 1.0, 0.0),      # Yellow
        (1.0, 0.0, 1.0),      # Magenta
        (0.0, 1.0, 1.0),      # Cyan
    ]


@pytest.fixture
def sample_backgrounds() -> List[List[float]]:
    """Provide sample background colors for testing."""
    return [
        [0, 0, 0],            # Black
        [255, 255, 255],      # White  
        [128, 128, 128],      # Gray (255 scale)
        [0.5, 0.5, 0.5],      # Gray (normalized)
        [255, 0, 0],          # Red
        [0, 255, 0],          # Green
        [0, 0, 255],          # Blue
    ]


@pytest.fixture
def known_luminance_values() -> List[Tuple[Tuple[float, float, float], float]]:
    """Provide colors with known luminance values for testing."""
    return [
        ((0.0, 0.0, 0.0), 0.0),           # Black
        ((1.0, 1.0, 1.0), 1.0),           # White
        ((1.0, 0.0, 0.0), 0.2126),        # Red
        ((0.0, 1.0, 0.0), 0.7152),        # Green
        ((0.0, 0.0, 1.0), 0.0722),        # Blue
    ]


@pytest.fixture
def color_format_examples() -> List[Tuple[str, Tuple[float, float, float]]]:
    """Provide examples of different color formats with expected RGB values."""
    return [
        ("#FF0000", (1.0, 0.0, 0.0)),     # Hex red
        ("#00FF00", (0.0, 1.0, 0.0)),     # Hex green
        ("#0000FF", (0.0, 0.0, 1.0)),     # Hex blue
        ("#FFFFFF", (1.0, 1.0, 1.0)),     # Hex white
        ("#000000", (0.0, 0.0, 0.0)),     # Hex black
        ("rgb(255, 0, 0)", (1.0, 0.0, 0.0)),      # RGB red
        ("rgb(0, 255, 0)", (0.0, 1.0, 0.0)),      # RGB green
        ("rgb(0, 0, 255)", (0.0, 0.0, 1.0)),      # RGB blue
        ("rgb(128, 128, 128)", (128/255, 128/255, 128/255)),  # RGB gray
    ]


@pytest.fixture
def invalid_color_formats() -> List[str]:
    """Provide examples of invalid color format strings."""
    return [
        "invalid",
        "#GG0000",             # Invalid hex characters
        "#FF00",               # Too short hex
        "#FF000000",           # Too long hex
        "FF0000",              # Missing # in hex
        "rgb(256, 0, 0)",      # RGB value out of range
        "rgb(-1, 0, 0)",       # Negative RGB value
        "rgb(255, 0)",         # Missing RGB component
        "hsl(361, 50%, 50%)",  # HSL hue out of range
        "hsl(180, 101%, 50%)", # HSL saturation out of range
        "hsl(180, 50%, 101%)", # HSL lightness out of range
        "hsv(361, 50%, 50%)",  # HSV hue out of range
        "hsv(180, 101%, 50%)", # HSV saturation out of range
        "hsv(180, 50%, 101%)", # HSV value out of range
        "",                    # Empty string
        "   ",                 # Whitespace only
    ]


@pytest.fixture
def contrast_test_pairs() -> List[Tuple[float, float, float]]:
    """Provide luminance pairs with known contrast ratios."""
    return [
        (0.0, 1.0, 21.0),      # Black vs White = 21:1
        (0.5, 0.5, 1.0),       # Same luminance = 1:1
        (0.1, 0.6, 4.33),      # Approximately 4.33:1
        (0.0, 0.18, 4.0),      # Approximately 4:1 (close to WCAG AA threshold)
    ]


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset numpy random seed before each test for reproducibility."""
    np.random.seed(42)


class ColorTestHelpers:
    """Helper class with utility methods for color testing."""
    
    @staticmethod
    def rgb_to_255_scale(rgb: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert RGB from [0,1] to [0,255] scale."""
        return tuple(int(round(c * 255)) for c in rgb)
    
    @staticmethod
    def is_valid_rgb(rgb: Tuple[float, float, float]) -> bool:
        """Check if RGB values are in valid [0,1] range."""
        return all(0.0 <= c <= 1.0 for c in rgb)
    
    @staticmethod
    def colors_approximately_equal(
        color1: Tuple[float, float, float], 
        color2: Tuple[float, float, float], 
        tolerance: float = 0.01
    ) -> bool:
        """Check if two colors are approximately equal within tolerance."""
        return all(abs(c1 - c2) < tolerance for c1, c2 in zip(color1, color2))


@pytest.fixture
def color_helpers() -> ColorTestHelpers:
    """Provide helper methods for color testing."""
    return ColorTestHelpers()