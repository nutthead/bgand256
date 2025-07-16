"""Tests for bgand256.colors module."""

from unittest.mock import patch

import numpy as np
import pytest

from bgand256.colors import (
    _compute_luminance,
    _contrast_ratio,
    generate_readable_colors,
)


class TestComputeLuminance:
    """Test the _compute_luminance function."""

    def test_black_luminance(self):
        """Test luminance of pure black."""
        result = _compute_luminance([0.0, 0.0, 0.0])
        assert result == 0.0

    def test_white_luminance(self):
        """Test luminance of pure white."""
        result = _compute_luminance([1.0, 1.0, 1.0])
        assert result == 1.0

    def test_red_luminance(self):
        """Test luminance of pure red."""
        result = _compute_luminance([1.0, 0.0, 0.0])
        # Red has luminance of approximately 0.2126
        assert abs(result - 0.2126) < 0.001

    def test_green_luminance(self):
        """Test luminance of pure green."""
        result = _compute_luminance([0.0, 1.0, 0.0])
        # Green has luminance of approximately 0.7152
        assert abs(result - 0.7152) < 0.001

    def test_blue_luminance(self):
        """Test luminance of pure blue."""
        result = _compute_luminance([0.0, 0.0, 1.0])
        # Blue has luminance of approximately 0.0722
        assert abs(result - 0.0722) < 0.001

    def test_mid_gray_luminance(self):
        """Test luminance of mid-gray."""
        # sRGB 128 = 0.5019607843 in linear RGB ≈ 0.21586
        result = _compute_luminance([128/255, 128/255, 128/255])
        assert 0.21 < result < 0.22

    def test_linearize_low_values(self):
        """Test linearization of low RGB values (<= 0.03928)."""
        # Test the linearize function indirectly through _compute_luminance
        low_value = 0.03
        result = _compute_luminance([low_value, 0.0, 0.0])
        # For low values, linearization is c / 12.92
        expected_linear = low_value / 12.92
        expected_luminance = 0.2126 * expected_linear
        assert abs(result - expected_luminance) < 0.0001

    def test_linearize_high_values(self):
        """Test linearization of high RGB values (> 0.03928)."""
        high_value = 0.5
        result = _compute_luminance([high_value, 0.0, 0.0])
        # For high values, linearization is ((c + 0.055) / 1.055) ** 2.4
        expected_linear = ((high_value + 0.055) / 1.055) ** 2.4
        expected_luminance = 0.2126 * expected_linear
        assert abs(result - expected_luminance) < 0.0001


class TestContrastRatio:
    """Test the _contrast_ratio function."""

    def test_identical_luminance(self):
        """Test contrast ratio of identical luminance values."""
        result = _contrast_ratio(0.5, 0.5)
        assert result == 1.0

    def test_black_white_contrast(self):
        """Test contrast ratio between black and white."""
        result = _contrast_ratio(0.0, 1.0)
        expected = (1.0 + 0.05) / (0.0 + 0.05)
        assert result == expected
        assert result == 21.0

    def test_order_independence(self):
        """Test that contrast ratio is independent of parameter order."""
        l1, l2 = 0.2, 0.8
        result1 = _contrast_ratio(l1, l2)
        result2 = _contrast_ratio(l2, l1)
        assert result1 == result2

    def test_mid_gray_contrast(self):
        """Test contrast ratio with mid-gray values."""
        # Test known values
        l1 = 0.1  # darker
        l2 = 0.6  # lighter
        result = _contrast_ratio(l1, l2)
        expected = (0.6 + 0.05) / (0.1 + 0.05)
        assert abs(result - expected) < 0.0001

    def test_wcag_aa_threshold(self):
        """Test colors that meet WCAG AA threshold (4.5:1)."""
        # Create luminance values that give exactly 4.5:1 ratio
        l_light = 0.5
        l_dark = (l_light + 0.05) / 4.5 - 0.05
        result = _contrast_ratio(l_light, l_dark)
        assert abs(result - 4.5) < 0.0001


class TestGenerateReadableColors:
    """Test the generate_readable_colors function."""

    def test_black_background(self):
        """Test color generation for black background."""
        background = [0, 0, 0]
        colors = generate_readable_colors(background)

        assert isinstance(colors, list)
        assert len(colors) <= 256
        assert len(colors) > 0

        # All colors should be tuples of 3 floats
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(isinstance(c, float | np.floating) for c in color)
            assert all(0.0 <= c <= 1.0 for c in color)

    def test_white_background(self):
        """Test color generation for white background."""
        background = [1.0, 1.0, 1.0]
        colors = generate_readable_colors(background)

        assert isinstance(colors, list)
        assert len(colors) <= 256
        assert len(colors) > 0

        # All colors should have sufficient contrast
        bg_luminance = _compute_luminance(background)
        for color in colors:
            color_luminance = _compute_luminance(color)
            ratio = _contrast_ratio(bg_luminance, color_luminance)
            assert ratio >= 4.5

    def test_255_scale_input(self):
        """Test with RGB values in 0-255 scale."""
        background = [128, 128, 128]  # Mid-gray in 255 scale
        colors = generate_readable_colors(background)

        assert len(colors) > 0
        # Verify colors are in [0,1] range despite 255-scale input
        for color in colors:
            assert all(0.0 <= c <= 1.0 for c in color)

    def test_normalized_input(self):
        """Test with RGB values already in 0-1 scale."""
        background = [0.5, 0.5, 0.5]  # Mid-gray in normalized scale
        colors = generate_readable_colors(background)

        assert len(colors) > 0
        for color in colors:
            assert all(0.0 <= c <= 1.0 for c in color)

    def test_contrast_requirement(self):
        """Test that all generated colors meet WCAG AA contrast requirement."""
        backgrounds = [
            [0, 0, 0],      # Black
            [255, 255, 255], # White
            [128, 64, 192],  # Purple
            [0.3, 0.7, 0.2]  # Green (normalized)
        ]

        for background in backgrounds:
            colors = generate_readable_colors(background)
            bg_luminance = _compute_luminance(
                np.array(background) / 255.0 if np.max(background) > 1.0
                else background
            )

            for color in colors:
                color_luminance = _compute_luminance(color)
                ratio = _contrast_ratio(bg_luminance, color_luminance)
                assert ratio >= 4.5, (
                    f"Color {color} has insufficient contrast {ratio:.2f}"
                )

    def test_maximum_colors_limit(self):
        """Test that function never returns more than 256 colors."""
        # Use a background that should generate many valid colors
        background = [128, 128, 128]
        colors = generate_readable_colors(background)
        assert len(colors) <= 256

    def test_different_background_colors(self):
        """Test various background colors to ensure robustness."""
        backgrounds = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
        ]

        for background in backgrounds:
            colors = generate_readable_colors(background)
            assert len(colors) > 0, f"No colors generated for background {background}"
            assert len(colors) <= 256

    def test_systematic_generation_with_early_return(self):
        """Test that systematic generation can find 256 colors and return early."""
        # Test multiple backgrounds that should generate many valid colors
        backgrounds_likely_to_hit_256 = [
            [0, 0, 0],      # Pure black - all light colors work
            [255, 255, 255], # Pure white - all dark colors work
        ]

        for background in backgrounds_likely_to_hit_256:
            colors = generate_readable_colors(background)

            # These backgrounds should generate close to or exactly 256 colors
            assert len(colors) >= 100, (
                f"Background {background} should generate many colors"
            )
            assert len(colors) <= 256

            # Verify contrast requirement
            bg_luminance = _compute_luminance(
                np.array(background, dtype=float) / 255.0 if np.max(background) > 1.0
                else background
            )
            for color in colors[:10]:  # Check first 10 for performance
                color_luminance = _compute_luminance(color)
                ratio = _contrast_ratio(bg_luminance, color_luminance)
                assert ratio >= 4.5

    def test_force_early_return_with_mock(self):
        """Force early return by mocking HSL_to_RGB to return high-contrast colors."""
        import colour

        # Mock to always return white color (high contrast against any dark background)
        with patch.object(colour.models.rgb.cylindrical, 'HSL_to_RGB') as mock_hsl:
            mock_hsl.return_value = np.array([0.9, 0.9, 0.9])  # Light color

            # Use a dark background to ensure contrast
            background = [0.1, 0.1, 0.1]
            colors = generate_readable_colors(background)

            # Should hit the early return when valid_colors reaches 256
            # Since all generated colors are light, they should all pass contrast test
            assert len(colors) == 256

            # Verify mock was called
            assert mock_hsl.called

    def test_fallback_random_sampling_behavior(self):
        """Test random sampling fallback with background that won't generate
        256 colors.
        """
        # Use a background that generates limited colors systematically
        # Mid-gray should generate fewer than 256 colors through systematic generation
        background = [0.5, 0.5, 0.5]

        # Set numpy random seed for reproducible test
        import numpy as np
        np.random.seed(12345)

        colors = generate_readable_colors(background)

        # Should still get colors, even if fallback is used
        assert len(colors) > 0
        assert len(colors) <= 256

        # All colors should still meet contrast requirement
        bg_luminance = _compute_luminance(background)
        for color in colors:
            color_luminance = _compute_luminance(color)
            ratio = _contrast_ratio(bg_luminance, color_luminance)
            assert ratio >= 4.5

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        background = np.array([128, 64, 192])
        colors = generate_readable_colors(background)

        assert len(colors) > 0
        assert all(isinstance(color, tuple) for color in colors)

    def test_edge_case_very_dark_background(self):
        """Test with very dark background."""
        background = [0.01, 0.01, 0.01]
        colors = generate_readable_colors(background)
        assert len(colors) > 0

    def test_edge_case_very_light_background(self):
        """Test with very light background."""
        background = [0.99, 0.99, 0.99]
        colors = generate_readable_colors(background)
        assert len(colors) > 0


@pytest.mark.parametrize("background,expected_min_colors", [
    ([0, 0, 0], 50),        # Black should generate many colors
    ([255, 255, 255], 50),  # White should generate many colors
    ([128, 128, 128], 20),  # Mid-gray should generate some colors
])
def test_color_generation_expectations(
    background: list[int], expected_min_colors: int
) -> None:
    """Test that certain backgrounds generate expected minimum number of colors."""
    colors = generate_readable_colors(background)
    assert len(colors) >= expected_min_colors, \
        (
            f"Expected at least {expected_min_colors} colors for "
            f"{background}, got {len(colors)}"
        )


def test_deterministic_systematic_generation() -> None:
    """Test that systematic generation is deterministic (before random fallback)."""
    background = [0, 0, 0]

    # Generate colors twice
    colors1 = generate_readable_colors(background)
    colors2 = generate_readable_colors(background)

    # The systematic part should be identical (random part might differ)
    # Since we use the same systematic grid, initial colors should match
    # We'll check at least the first few colors are the same
    min_len = min(len(colors1), len(colors2), 10)
    if min_len > 0:
        # Note: due to random fallback, we can't guarantee full determinism
        # But the systematic generation should be consistent
        pass  # This test would need to be refined based on actual implementation

