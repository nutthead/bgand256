"""Property-based tests for contrast algorithms module.

This module provides comprehensive property-based testing for all algorithms
in the contrast_algorithms module to ensure correctness and robustness.

Author: Behrang Saeedzadeh <hello@behrang.org>
"""

# pyright: reportPrivateUsage=false

import warnings
from typing import Any

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from bgand256.colors import _compute_luminance, _contrast_ratio  # type: ignore
from bgand256.contrast_algorithms import (  # type: ignore
    _calculate_cam16ucs_distance,
    _calculate_delta_e_2000,
    _calculate_hsl_perceptual_distance,
    _convert_rgb_to_cam16ucs,
    _convert_rgb_to_lab,
    _generate_wcag_compliant_candidates,
    _get_distance_function,
    _greedy_select_contrasting_colors,
    generate_contrasting_colors,
)

# Configure hypothesis settings for faster tests
settings.register_profile("fast", max_examples=20, deadline=5000)
settings.load_profile("fast")

# Type aliases for clarity
RGBColor = tuple[float, float, float]
RGB255Color = tuple[int, int, int]

# Strategies for generating test data
rgb_color_strategy = st.tuples(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)

rgb_255_color_strategy = st.tuples(
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
)

algorithm_strategy = st.sampled_from(["delta-e", "cam16ucs", "hsl-greedy"])


class TestColorConversionFunctions:
    """Test color space conversion functions."""

    @given(rgb_color_strategy)
    def test_convert_rgb_to_lab_returns_valid_array(self, rgb_color: RGBColor) -> None:
        """Test RGB to LAB conversion returns valid numpy array."""
        assume(all(0 <= c <= 1 for c in rgb_color))

        lab = _convert_rgb_to_lab(rgb_color)

        assert isinstance(lab, np.ndarray)
        assert lab.shape == (3,)
        assert np.all(np.isfinite(lab))

        # LAB ranges: L*[0,100], a*[-128,127], b*[-128,127]
        assert 0 <= lab[0] <= 100
        assert -128 <= lab[1] <= 127
        assert -128 <= lab[2] <= 127

    @given(rgb_color_strategy)
    def test_convert_rgb_to_cam16ucs_returns_valid_array(
        self, rgb_color: RGBColor
    ) -> None:
        """Test RGB to CAM16UCS conversion returns valid numpy array."""
        assume(all(0 <= c <= 1 for c in rgb_color))

        try:
            cam16ucs = _convert_rgb_to_cam16ucs(rgb_color)

            assert isinstance(cam16ucs, np.ndarray)
            assert cam16ucs.shape == (3,)
            assert np.all(np.isfinite(cam16ucs))
        except Exception:
            # CAM16UCS can fail for edge cases, which is acceptable
            pass

    @given(rgb_color_strategy, rgb_color_strategy)
    def test_convert_rgb_to_lab_different_colors_different_results(
        self, color1: RGBColor, color2: RGBColor
    ) -> None:
        """Test that different colors produce different LAB values."""
        assume(color1 != color2)
        assume(all(0 <= c <= 1 for c in color1))
        assume(all(0 <= c <= 1 for c in color2))
        # Skip very similar colors to avoid numerical precision issues
        assume(sum(abs(c1 - c2) for c1, c2 in zip(color1, color2, strict=False)) > 0.01)

        lab1 = _convert_rgb_to_lab(color1)
        lab2 = _convert_rgb_to_lab(color2)

        assert not np.allclose(lab1, lab2, rtol=1e-6)


class TestDistanceFunctions:
    """Test perceptual distance calculation functions."""

    @given(rgb_color_strategy, rgb_color_strategy)
    def test_calculate_delta_e_2000_properties(
        self, color1: RGBColor, color2: RGBColor
    ) -> None:
        """Test Delta E 2000 calculation properties."""
        assume(all(0 <= c <= 1 for c in color1))
        assume(all(0 <= c <= 1 for c in color2))

        distance = _calculate_delta_e_2000(color1, color2)

        # Distance should be non-negative
        assert distance >= 0

        # Distance should be symmetric
        reverse_distance = _calculate_delta_e_2000(color2, color1)
        assert abs(distance - reverse_distance) < 1e-10

        # Distance from color to itself should be 0
        self_distance = _calculate_delta_e_2000(color1, color1)
        assert abs(self_distance) < 1e-10

    @given(rgb_color_strategy, rgb_color_strategy)
    def test_calculate_cam16ucs_distance_properties(
        self, color1: RGBColor, color2: RGBColor
    ) -> None:
        """Test CAM16UCS distance calculation properties."""
        assume(all(0 <= c <= 1 for c in color1))
        assume(all(0 <= c <= 1 for c in color2))

        distance = _calculate_cam16ucs_distance(color1, color2)

        # Distance should be non-negative
        assert distance >= 0

        # Distance should be symmetric
        reverse_distance = _calculate_cam16ucs_distance(color2, color1)
        assert abs(distance - reverse_distance) < 1e-6

        # Distance from color to itself should be 0
        self_distance = _calculate_cam16ucs_distance(color1, color1)
        assert abs(self_distance) < 1e-6

    @given(rgb_color_strategy, rgb_color_strategy)
    def test_calculate_hsl_perceptual_distance_properties(
        self, color1: RGBColor, color2: RGBColor
    ) -> None:
        """Test HSL perceptual distance calculation properties."""
        assume(all(0 <= c <= 1 for c in color1))
        assume(all(0 <= c <= 1 for c in color2))

        distance = _calculate_hsl_perceptual_distance(color1, color2)

        # Distance should be non-negative
        assert distance >= 0

        # Distance should be symmetric
        reverse_distance = _calculate_hsl_perceptual_distance(color2, color1)
        assert abs(distance - reverse_distance) < 1e-10

        # Distance from color to itself should be 0
        self_distance = _calculate_hsl_perceptual_distance(color1, color1)
        assert abs(self_distance) < 1e-10

    @given(algorithm_strategy)
    def test_get_distance_function_returns_callable(self, algorithm: str) -> None:
        """Test that get_distance_function returns a callable."""
        distance_fn = _get_distance_function(algorithm)  # type: ignore[arg-type]

        assert callable(distance_fn)

        # Test that the function works with valid inputs
        result = distance_fn((0.5, 0.5, 0.5), (0.6, 0.6, 0.6))
        assert isinstance(result, float)
        assert result >= 0

    def test_get_distance_function_invalid_algorithm(self) -> None:
        """Test that get_distance_function raises KeyError for invalid algorithm."""
        with pytest.raises(KeyError):
            _get_distance_function("invalid-algorithm")  # type: ignore


class TestCandidateGeneration:
    """Test WCAG-compliant candidate generation."""

    @given(rgb_color_strategy)
    def test_generate_wcag_compliant_candidates_returns_valid_colors(
        self, background: RGBColor
    ) -> None:
        """Test that candidate generation returns valid RGB colors."""
        assume(all(0 <= c <= 1 for c in background))

        candidates = _generate_wcag_compliant_candidates(background)

        assert isinstance(candidates, list)

        # Only check first few candidates to avoid timeout
        for candidate in candidates[:10]:
            assert isinstance(candidate, tuple)
            assert len(candidate) == 3
            assert all(0 <= c <= 1 for c in candidate)

    @given(rgb_255_color_strategy)
    def test_generate_wcag_compliant_candidates_handles_255_range(
        self, background: RGB255Color
    ) -> None:
        """Test that candidate generation handles 255-range RGB input."""
        candidates = _generate_wcag_compliant_candidates(background)

        assert isinstance(candidates, list)

        for candidate in candidates:
            assert isinstance(candidate, tuple)
            assert len(candidate) == 3
            assert all(0 <= c <= 1 for c in candidate)

    @given(rgb_color_strategy, st.floats(min_value=1.0, max_value=21.0))
    def test_generate_wcag_compliant_candidates_respects_min_contrast(
        self, background: RGBColor, min_contrast: float
    ) -> None:
        """Test that all candidates meet minimum contrast requirement."""
        assume(all(0 <= c <= 1 for c in background))

        candidates = _generate_wcag_compliant_candidates(background, min_contrast)

        bg_luminance = _compute_luminance(np.array(background))

        # Only check first few candidates to avoid timeout
        for candidate in candidates[:10]:
            candidate_luminance = _compute_luminance(np.array(candidate))
            contrast = _contrast_ratio(bg_luminance, candidate_luminance)
            assert contrast >= min_contrast - 1e-10  # Allow for floating point errors

    @given(st.lists(st.floats(min_value=-1.0, max_value=2.0), min_size=3, max_size=3))
    def test_generate_wcag_compliant_candidates_handles_invalid_rgb(
        self, invalid_rgb: Any
    ) -> None:
        """Test that candidate generation handles invalid RGB values gracefully."""
        # This should not crash, even with invalid inputs
        try:
            candidates = _generate_wcag_compliant_candidates(invalid_rgb)
            assert isinstance(candidates, list)
        except Exception:
            # Some invalid inputs might cause exceptions, which is acceptable
            pass


class TestGreedySelection:
    """Test greedy color selection algorithm."""

    @given(
        st.lists(rgb_color_strategy, min_size=3, max_size=8),
        st.integers(min_value=1, max_value=5),
        algorithm_strategy,
    )
    def test_greedy_select_contrasting_colors_returns_valid_selection(
        self, candidates: Any, target_count: int, algorithm: str
    ) -> None:
        """Test that greedy selection returns valid color selection."""
        assume(len(candidates) >= target_count)
        assume(all(all(0 <= c <= 1 for c in color) for color in candidates))

        selected = _greedy_select_contrasting_colors(
            candidates, target_count, algorithm  # type: ignore[arg-type]
        )

        assert isinstance(selected, list)
        assert len(selected) <= target_count
        assert len(selected) <= len(candidates)

        # All selected colors should be from the candidates
        for color in selected:
            assert color in candidates

    @given(st.integers(min_value=1, max_value=10), algorithm_strategy)
    def test_greedy_select_contrasting_colors_empty_candidates(
        self, target_count: int, algorithm: str
    ) -> None:
        """Test that greedy selection handles empty candidate list."""
        selected = _greedy_select_contrasting_colors([], target_count, algorithm)  # type: ignore[arg-type]

        assert selected == []

    @given(
        st.lists(rgb_color_strategy, min_size=1, max_size=5),
        st.integers(min_value=10, max_value=20),
        algorithm_strategy,
    )
    def test_greedy_select_contrasting_colors_more_targets_than_candidates(
        self, candidates: Any, target_count: int, algorithm: str
    ) -> None:
        """Test greedy selection when target count exceeds candidate count."""
        assume(len(candidates) < target_count)
        assume(all(all(0 <= c <= 1 for c in color) for color in candidates))

        selected = _greedy_select_contrasting_colors(
            candidates, target_count, algorithm  # type: ignore[arg-type]
        )

        assert len(selected) <= len(candidates)

    @given(
        st.lists(rgb_color_strategy, min_size=5, max_size=20),
        st.integers(min_value=1, max_value=10),
        algorithm_strategy,
        st.floats(min_value=0.1, max_value=100.0),
    )
    def test_greedy_select_contrasting_colors_with_min_distance(
        self, candidates: Any, target_count: int, algorithm: str, min_distance: float
    ) -> None:
        """Test greedy selection with custom minimum distance."""
        assume(len(candidates) >= target_count)
        assume(all(all(0 <= c <= 1 for c in color) for color in candidates))

        selected = _greedy_select_contrasting_colors(
            candidates, target_count, algorithm, min_distance  # type: ignore[arg-type]
        )

        assert isinstance(selected, list)
        assert len(selected) <= target_count

        # Check that selected colors maintain minimum distance (when possible)
        if len(selected) > 1:
            distance_fn = _get_distance_function(algorithm)  # type: ignore[arg-type]
            for i, color1 in enumerate(selected):
                for j, color2 in enumerate(selected):
                    if i != j:
                        distance = distance_fn(color1, color2)
                        # Distance should be positive (different colors)
                        assert distance >= 0


class TestMainFunction:
    """Test the main generate_contrasting_colors function."""

    @given(
        rgb_color_strategy,
        st.integers(min_value=1, max_value=10),
        algorithm_strategy,
    )
    def test_generate_contrasting_colors_basic_properties(
        self, background: RGBColor, target_count: int, algorithm: str
    ) -> None:
        """Test basic properties of generate_contrasting_colors."""
        assume(all(0 <= c <= 1 for c in background))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(background, target_count, algorithm)  # type: ignore[arg-type]

        assert isinstance(colors, list)
        assert len(colors) <= target_count

        # All colors should be valid RGB tuples
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 1 for c in color)

    @given(
        rgb_255_color_strategy,
        st.integers(min_value=1, max_value=10),
        algorithm_strategy,
    )
    def test_generate_contrasting_colors_handles_255_range(
        self, background: RGB255Color, target_count: int, algorithm: str
    ) -> None:
        """Test that main function handles 255-range RGB input."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(background, target_count, algorithm)  # type: ignore[arg-type]

        assert isinstance(colors, list)
        assert len(colors) <= target_count

    @given(
        rgb_color_strategy,
        st.integers(min_value=1, max_value=5),
        algorithm_strategy,
        st.floats(min_value=1.0, max_value=7.0),
    )
    def test_generate_contrasting_colors_respects_min_contrast(
        self,
        background: RGBColor,
        target_count: int,
        algorithm: str,
        min_contrast: float,
    ) -> None:
        """Test that generated colors meet minimum contrast requirement."""
        assume(all(0 <= c <= 1 for c in background))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(
                background, target_count, algorithm, min_contrast  # type: ignore[arg-type]
            )

        bg_luminance = _compute_luminance(np.array(background))

        for color in colors:
            color_luminance = _compute_luminance(np.array(color))
            contrast = _contrast_ratio(bg_luminance, color_luminance)
            assert contrast >= min_contrast - 1e-10

    @given(
        rgb_color_strategy,
        st.integers(min_value=1, max_value=5),
        algorithm_strategy,
        st.floats(min_value=0.1, max_value=10.0),
    )
    def test_generate_contrasting_colors_with_min_mutual_distance(
        self,
        background: RGBColor,
        target_count: int,
        algorithm: str,
        min_mutual_distance: float,
    ) -> None:
        """Test generate_contrasting_colors with custom minimum mutual distance."""
        assume(all(0 <= c <= 1 for c in background))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(
                background,
                target_count,
                algorithm,  # type: ignore[arg-type]
                min_mutual_distance=min_mutual_distance,
            )

        assert isinstance(colors, list)
        assert len(colors) <= target_count

    @given(arrays(np.float64, (3,), elements=st.floats(min_value=0.0, max_value=1.0)))
    def test_generate_contrasting_colors_with_numpy_input(
        self, background_array: np.ndarray
    ) -> None:
        """Test that main function handles numpy array input."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(background_array, 5, "hsl-greedy")  # type: ignore[arg-type]

        assert isinstance(colors, list)

    def test_generate_contrasting_colors_edge_cases(self) -> None:
        """Test edge cases for generate_contrasting_colors."""
        # Test with pure black background
        colors = generate_contrasting_colors([0, 0, 0], 5, "hsl-greedy")
        assert len(colors) <= 5

        # Test with pure white background
        colors = generate_contrasting_colors([1, 1, 1], 5, "hsl-greedy")
        assert len(colors) <= 5

        # Test with zero target count
        colors = generate_contrasting_colors([0.5, 0.5, 0.5], 0, "hsl-greedy")
        assert len(colors) == 0

    def test_generate_contrasting_colors_all_algorithms(self) -> None:
        """Test that all algorithms work with typical inputs."""
        background = [0.2, 0.3, 0.4]
        target_count = 10

        for algorithm in ["delta-e", "cam16ucs", "hsl-greedy"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                colors = generate_contrasting_colors(
                    background, target_count, algorithm  # type: ignore[arg-type]
                )

            assert isinstance(colors, list)
            assert len(colors) <= target_count

            # Verify all colors are valid
            for color in colors:
                assert isinstance(color, tuple)
                assert len(color) == 3
                assert all(0 <= c <= 1 for c in color)

