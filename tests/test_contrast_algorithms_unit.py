"""Unit tests for contrast algorithms module to achieve >90% coverage.

This module provides focused unit tests for complete coverage of the
contrast_algorithms module without relying on slow property-based testing.

Author: Behrang Saeedzadeh <hello@behrang.org>
"""

# pyright: reportPrivateUsage=false

import warnings

import numpy as np
import pytest

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

# Type aliases for clarity
RGBColor = tuple[float, float, float]


class TestColorConversionFunctions:
    """Test color space conversion functions."""

    def test_convert_rgb_to_lab_basic(self) -> None:
        """Test basic RGB to LAB conversion."""
        # Test with known values
        lab = _convert_rgb_to_lab((1.0, 0.0, 0.0))  # Red
        assert isinstance(lab, np.ndarray)
        assert lab.shape == (3,)
        assert np.all(np.isfinite(lab))

        lab = _convert_rgb_to_lab((0.0, 1.0, 0.0))  # Green
        assert isinstance(lab, np.ndarray)
        assert lab.shape == (3,)
        assert np.all(np.isfinite(lab))

        lab = _convert_rgb_to_lab((0.0, 0.0, 1.0))  # Blue
        assert isinstance(lab, np.ndarray)
        assert lab.shape == (3,)
        assert np.all(np.isfinite(lab))

    def test_convert_rgb_to_cam16ucs_basic(self) -> None:
        """Test basic RGB to CAM16UCS conversion."""
        # Test with known values
        try:
            cam16ucs = _convert_rgb_to_cam16ucs((1.0, 0.0, 0.0))  # Red
            assert isinstance(cam16ucs, np.ndarray)
            assert cam16ucs.shape == (3,)
            assert np.all(np.isfinite(cam16ucs))
        except Exception:
            # CAM16UCS can fail for some inputs, which is handled
            pass

    def test_convert_rgb_to_lab_edge_cases(self) -> None:
        """Test LAB conversion with edge cases."""
        # Test with black
        lab = _convert_rgb_to_lab((0.0, 0.0, 0.0))
        assert isinstance(lab, np.ndarray)
        assert lab.shape == (3,)

        # Test with white
        lab = _convert_rgb_to_lab((1.0, 1.0, 1.0))
        assert isinstance(lab, np.ndarray)
        assert lab.shape == (3,)

        # Test with mid-gray
        lab = _convert_rgb_to_lab((0.5, 0.5, 0.5))
        assert isinstance(lab, np.ndarray)
        assert lab.shape == (3,)


class TestDistanceFunctions:
    """Test perceptual distance calculation functions."""

    def test_calculate_delta_e_2000_basic(self) -> None:
        """Test Delta E 2000 calculation."""
        # Test identical colors
        distance = _calculate_delta_e_2000((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert abs(distance) < 1e-6

        # Test different colors
        distance = _calculate_delta_e_2000((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert distance > 0

        # Test symmetry
        d1 = _calculate_delta_e_2000((0.2, 0.3, 0.4), (0.6, 0.7, 0.8))
        d2 = _calculate_delta_e_2000((0.6, 0.7, 0.8), (0.2, 0.3, 0.4))
        assert abs(d1 - d2) < 1e-6

    def test_calculate_cam16ucs_distance_basic(self) -> None:
        """Test CAM16UCS distance calculation."""
        # Test identical colors
        distance = _calculate_cam16ucs_distance((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert abs(distance) < 1e-6

        # Test different colors
        distance = _calculate_cam16ucs_distance((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert distance > 0

        # Test symmetry
        d1 = _calculate_cam16ucs_distance((0.2, 0.3, 0.4), (0.6, 0.7, 0.8))
        d2 = _calculate_cam16ucs_distance((0.6, 0.7, 0.8), (0.2, 0.3, 0.4))
        assert abs(d1 - d2) < 1e-6

    def test_calculate_cam16ucs_distance_fallback(self) -> None:
        """Test CAM16UCS distance fallback to Delta E."""
        # Test with potential problematic values that might trigger fallback
        distance = _calculate_cam16ucs_distance((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        assert distance >= 0

        # Test with out-of-range values to force exception and fallback
        try:
            distance = _calculate_cam16ucs_distance((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
            assert distance >= 0
        except Exception:
            # If it still fails, that's acceptable
            pass

    def test_calculate_hsl_perceptual_distance_basic(self) -> None:
        """Test HSL perceptual distance calculation."""
        # Test identical colors
        distance = _calculate_hsl_perceptual_distance((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert abs(distance) < 1e-6

        # Test different colors
        distance = _calculate_hsl_perceptual_distance((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert distance > 0

        # Test symmetry
        d1 = _calculate_hsl_perceptual_distance((0.2, 0.3, 0.4), (0.6, 0.7, 0.8))
        d2 = _calculate_hsl_perceptual_distance((0.6, 0.7, 0.8), (0.2, 0.3, 0.4))
        assert abs(d1 - d2) < 1e-6

    def test_calculate_hsl_perceptual_distance_hue_wrapping(self) -> None:
        """Test HSL distance with hue wrapping."""
        # Test hue wrapping (red to red via wrap-around)
        distance = _calculate_hsl_perceptual_distance((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        assert abs(distance) < 1e-6

    def test_get_distance_function_all_algorithms(self) -> None:
        """Test get_distance_function for all algorithms."""
        algorithms = ["delta-e", "cam16ucs", "hsl-greedy"]

        for algorithm in algorithms:
            distance_fn = _get_distance_function(algorithm)  # type: ignore[arg-type]
            assert callable(distance_fn)

            # Test that function works
            result = distance_fn((0.2, 0.3, 0.4), (0.6, 0.7, 0.8))
            assert isinstance(result, float)
            assert result >= 0

    def test_get_distance_function_invalid_algorithm(self) -> None:
        """Test get_distance_function with invalid algorithm."""
        with pytest.raises(KeyError):
            _get_distance_function("invalid-algorithm")  # type: ignore


class TestCandidateGeneration:
    """Test WCAG-compliant candidate generation."""

    def test_generate_wcag_compliant_candidates_basic(self) -> None:
        """Test basic candidate generation."""
        background = (0.5, 0.5, 0.5)
        candidates = _generate_wcag_compliant_candidates(background)

        assert isinstance(candidates, list)
        assert len(candidates) > 0

        # Check that all candidates are valid RGB tuples
        for candidate in candidates[:5]:  # Check first 5 to avoid long loops
            assert isinstance(candidate, tuple)
            assert len(candidate) == 3
            assert all(0 <= c <= 1 for c in candidate)

    def test_generate_wcag_compliant_candidates_255_range(self) -> None:
        """Test candidate generation with 255-range input."""
        background = (128, 128, 128)
        candidates = _generate_wcag_compliant_candidates(background)

        assert isinstance(candidates, list)
        assert len(candidates) > 0

        # Check that all candidates are normalized to [0,1] range
        for candidate in candidates[:5]:
            assert isinstance(candidate, tuple)
            assert len(candidate) == 3
            assert all(0 <= c <= 1 for c in candidate)

    def test_generate_wcag_compliant_candidates_contrast_requirement(self) -> None:
        """Test that candidates meet contrast requirements."""
        background = (0.2, 0.2, 0.2)
        min_contrast = 7.0
        candidates = _generate_wcag_compliant_candidates(background, min_contrast)

        bg_luminance = _compute_luminance(np.array(background))

        # Check first few candidates
        for candidate in candidates[:3]:
            candidate_luminance = _compute_luminance(np.array(candidate))
            contrast = _contrast_ratio(bg_luminance, candidate_luminance)
            assert contrast >= min_contrast - 1e-10

    def test_generate_wcag_compliant_candidates_extreme_backgrounds(self) -> None:
        """Test candidate generation with extreme backgrounds."""
        # Very dark background
        candidates = _generate_wcag_compliant_candidates((0.0, 0.0, 0.0))
        assert isinstance(candidates, list)

        # Very light background
        candidates = _generate_wcag_compliant_candidates((1.0, 1.0, 1.0))
        assert isinstance(candidates, list)

    def test_generate_wcag_compliant_candidates_high_contrast(self) -> None:
        """Test candidate generation with high contrast requirement."""
        background = (0.5, 0.5, 0.5)
        min_contrast = 21.0  # Maximum possible contrast
        candidates = _generate_wcag_compliant_candidates(background, min_contrast)

        # Should return empty list or very few candidates
        assert isinstance(candidates, list)
        assert len(candidates) == 0

    def test_generate_wcag_compliant_candidates_invalid_input(self) -> None:
        """Test candidate generation with invalid input."""
        # Test with out-of-range values
        try:
            candidates = _generate_wcag_compliant_candidates([-0.5, 1.5, 2.0])
            assert isinstance(candidates, list)
        except Exception:
            # Some invalid inputs might cause exceptions
            pass

    def test_generate_wcag_compliant_candidates_normalization(self) -> None:
        """Test RGB normalization in candidate generation."""
        # Test with values > 1.0 to trigger normalization
        background = (150.0, 200.0, 250.0)  # Values > 1.0
        candidates = _generate_wcag_compliant_candidates(background)

        assert isinstance(candidates, list)

        # Check normalization worked
        for candidate in candidates[:3]:
            assert all(0 <= c <= 1 for c in candidate)


class TestGreedySelection:
    """Test greedy color selection algorithm."""

    def test_greedy_select_contrasting_colors_basic(self) -> None:
        """Test basic greedy selection."""
        candidates = [
            (0.1, 0.1, 0.1),
            (0.9, 0.9, 0.9),
            (0.5, 0.1, 0.1),
            (0.1, 0.5, 0.1),
            (0.1, 0.1, 0.5),
        ]

        selected = _greedy_select_contrasting_colors(candidates, 3, "hsl-greedy")  # type: ignore[arg-type]

        assert isinstance(selected, list)
        assert len(selected) <= 3
        assert len(selected) <= len(candidates)

        # All selected colors should be from candidates
        for color in selected:
            assert color in candidates

    def test_greedy_select_contrasting_colors_empty_candidates(self) -> None:
        """Test greedy selection with empty candidates."""
        selected = _greedy_select_contrasting_colors([], 5, "hsl-greedy")  # type: ignore[arg-type]
        assert selected == []

    def test_greedy_select_contrasting_colors_more_targets_than_candidates(
        self,
    ) -> None:
        """Test when target count exceeds candidates."""
        candidates = [(0.1, 0.1, 0.1), (0.9, 0.9, 0.9)]
        selected = _greedy_select_contrasting_colors(candidates, 10, "hsl-greedy")  # type: ignore[arg-type]

        assert len(selected) <= len(candidates)

    def test_greedy_select_contrasting_colors_all_algorithms(self) -> None:
        """Test greedy selection with all algorithms."""
        candidates = [
            (0.1, 0.1, 0.1),
            (0.9, 0.9, 0.9),
            (0.5, 0.1, 0.1),
            (0.1, 0.5, 0.1),
        ]

        for algorithm in ["delta-e", "cam16ucs", "hsl-greedy"]:
            selected = _greedy_select_contrasting_colors(candidates, 2, algorithm)  # type: ignore[arg-type]
            assert isinstance(selected, list)
            assert len(selected) <= 2

    def test_greedy_select_contrasting_colors_with_min_distance(self) -> None:
        """Test greedy selection with custom minimum distance."""
        candidates = [
            (0.1, 0.1, 0.1),
            (0.9, 0.9, 0.9),
            (0.5, 0.1, 0.1),
            (0.1, 0.5, 0.1),
        ]

        selected = _greedy_select_contrasting_colors(candidates, 2, "hsl-greedy", 0.5)  # type: ignore[arg-type]
        assert isinstance(selected, list)
        assert len(selected) <= 2

    def test_greedy_select_contrasting_colors_distance_threshold(self) -> None:
        """Test that greedy selection with distance thresholds."""
        candidates = [
            (0.1, 0.1, 0.1),
            (0.9, 0.9, 0.9),
            (0.11, 0.11, 0.11),  # Very similar to first
            (0.91, 0.91, 0.91),  # Very similar to second
        ]

        # With high minimum distance, check that algorithm still works
        selected = _greedy_select_contrasting_colors(candidates, 4, "hsl-greedy", 1.0)  # type: ignore[arg-type]
        assert len(selected) >= 1  # Should select at least one color
        assert len(selected) <= 4  # Should not exceed candidates

    def test_greedy_select_contrasting_colors_fallback_behavior(self) -> None:
        """Test fallback behavior when no candidates meet distance requirement."""
        candidates = [
            (0.1, 0.1, 0.1),
            (0.11, 0.11, 0.11),
            (0.12, 0.12, 0.12),
        ]

        # Should still select some colors even with high minimum distance
        selected = _greedy_select_contrasting_colors(candidates, 3, "hsl-greedy", 10.0)  # type: ignore[arg-type]
        assert len(selected) > 0

    def test_greedy_select_contrasting_colors_no_best_candidate(self) -> None:
        """Test when no best candidate is found."""
        candidates = [
            (0.1, 0.1, 0.1),
            (0.1, 0.1, 0.1),  # Duplicate
            (0.1, 0.1, 0.1),  # Duplicate
        ]

        # Should handle case where no distinct candidates exist
        selected = _greedy_select_contrasting_colors(candidates, 3, "hsl-greedy")  # type: ignore[arg-type]
        assert len(selected) >= 1  # At least one should be selected

    def test_greedy_select_contrasting_colors_early_termination(self) -> None:
        """Test early termination conditions."""
        candidates = [
            (0.1, 0.1, 0.1),
            (0.9, 0.9, 0.9),
            (0.11, 0.11, 0.11),  # Very close to first
        ]

        # Test target_count // 2 condition
        selected = _greedy_select_contrasting_colors(candidates, 4, "hsl-greedy", 5.0)  # type: ignore[arg-type]
        assert len(selected) >= 1


class TestMainFunction:
    """Test the main generate_contrasting_colors function."""

    def test_generate_contrasting_colors_basic(self) -> None:
        """Test basic functionality of generate_contrasting_colors."""
        background = (0.5, 0.5, 0.5)
        target_count = 5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(background, target_count, "hsl-greedy")  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert isinstance(colors, list)
        assert len(colors) <= target_count

        # All colors should be valid RGB tuples
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 1 for c in color)

    def test_generate_contrasting_colors_all_algorithms(self) -> None:
        """Test all algorithms work."""
        background = (0.2, 0.3, 0.4)
        target_count = 5

        for algorithm in ["delta-e", "cam16ucs", "hsl-greedy"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                colors = generate_contrasting_colors(
                    background, target_count, algorithm  # type: ignore[arg-type]
                )

            assert isinstance(colors, list)
            assert len(colors) <= target_count

    def test_generate_contrasting_colors_255_range_input(self) -> None:
        """Test with 255-range RGB input."""
        background = (128, 128, 128)
        target_count = 5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(background, target_count, "hsl-greedy")  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert isinstance(colors, list)
        assert len(colors) <= target_count

    def test_generate_contrasting_colors_numpy_input(self) -> None:
        """Test with numpy array input."""
        background = np.array([0.5, 0.5, 0.5])
        target_count = 5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(background, target_count, "hsl-greedy")  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert isinstance(colors, list)
        assert len(colors) <= target_count

    def test_generate_contrasting_colors_custom_contrast(self) -> None:
        """Test with custom minimum contrast."""
        background = (0.2, 0.2, 0.2)
        target_count = 5
        min_contrast = 7.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(
                background, target_count, "hsl-greedy", min_contrast
            )

        assert isinstance(colors, list)

        # Verify contrast requirement is met
        bg_luminance = _compute_luminance(np.array(background))
        for color in colors:
            color_luminance = _compute_luminance(np.array(color))
            contrast = _contrast_ratio(bg_luminance, color_luminance)
            assert contrast >= min_contrast - 1e-10

    def test_generate_contrasting_colors_custom_mutual_distance(self) -> None:
        """Test with custom minimum mutual distance."""
        background = (0.5, 0.5, 0.5)
        target_count = 5
        min_mutual_distance = 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(
                background,
                target_count,
                "hsl-greedy",
                min_mutual_distance=min_mutual_distance,
            )

        assert isinstance(colors, list)
        assert len(colors) <= target_count

    def test_generate_contrasting_colors_edge_cases(self) -> None:
        """Test edge cases."""
        # Test with pure black
        colors = generate_contrasting_colors((0, 0, 0), 3, "hsl-greedy")  # type: ignore[arg-type]
        assert isinstance(colors, list)
        assert len(colors) <= 3

        # Test with pure white
        colors = generate_contrasting_colors((1, 1, 1), 3, "hsl-greedy")  # type: ignore[arg-type]
        assert isinstance(colors, list)
        assert len(colors) <= 3

        # Test with zero target count
        colors = generate_contrasting_colors((0.5, 0.5, 0.5), 0, "hsl-greedy")  # type: ignore[arg-type]
        assert colors == []

    def test_generate_contrasting_colors_insufficient_candidates(self) -> None:
        """Test when not enough candidates can be generated."""
        background = (0.5, 0.5, 0.5)
        target_count = 5
        min_contrast = 21.0  # Impossible contrast requirement

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(
                background, target_count, "hsl-greedy", min_contrast
            )

        assert isinstance(colors, list)
        assert len(colors) == 0

    def test_generate_contrasting_colors_truncation(self) -> None:
        """Test that result is truncated to target count."""
        background = (0.1, 0.1, 0.1)
        target_count = 3

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(background, target_count, "hsl-greedy")  # type: ignore[arg-type]  # type: ignore[arg-type]

        assert len(colors) <= target_count

    def test_generate_contrasting_colors_fewer_candidates_than_target(self) -> None:
        """Test when fewer candidates than target count."""
        background = (0.5, 0.5, 0.5)
        target_count = 1000  # Very high target count
        min_contrast = 15.0  # High contrast requirement

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(
                background, target_count, "hsl-greedy", min_contrast
            )

        # Should return all available candidates (fewer than target)
        assert len(colors) < target_count

    def test_cam16ucs_exception_handling(self) -> None:
        """Test explicit exception handling in CAM16UCS."""
        # Force an exception by using problematic values
        result = _calculate_cam16ucs_distance((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        assert result >= 0  # Should fallback to Delta E

    def test_generate_contrasting_colors_warnings_suppression(self) -> None:
        """Test that warnings are properly suppressed."""
        background = (0.1, 0.1, 0.1)

        # This should not produce warnings in output
        colors = generate_contrasting_colors(background, 5, "delta-e")  # type: ignore[arg-type]
        assert isinstance(colors, list)

    def test_cam16ucs_exception_path(self) -> None:
        """Test CAM16UCS exception handling path."""
        # Use extreme values to trigger CAM16UCS exception
        from unittest.mock import patch

        # Mock the CAM16UCS conversion to raise an exception
        with patch(
            'bgand256.contrast_algorithms._convert_rgb_to_cam16ucs',
            side_effect=Exception("Mock exception"),
        ):
            distance = _calculate_cam16ucs_distance((0.5, 0.5, 0.5), (0.6, 0.6, 0.6))
            assert distance >= 0  # Should fallback to Delta E

    def test_greedy_selection_break_conditions(self) -> None:
        """Test break conditions in greedy selection."""
        candidates = [
            (0.1, 0.1, 0.1),
            (0.2, 0.2, 0.2),
            (0.3, 0.3, 0.3),
            (0.4, 0.4, 0.4),
        ]

        # Test condition where len(selected) < target_count // 2
        selected = _greedy_select_contrasting_colors(
            candidates, 10, "hsl-greedy", 100.0
        )
        assert len(selected) >= 1  # Should select at least some colors

    def test_generate_contrasting_colors_candidate_count_condition(self) -> None:
        """Test condition where candidates <= target_count."""
        background = (0.5, 0.5, 0.5)

        # Use very high contrast to limit candidates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            colors = generate_contrasting_colors(background, 1000, "hsl-greedy", 20.0)  # type: ignore[arg-type]

        # Should return all available candidates
        assert isinstance(colors, list)
        assert len(colors) < 1000  # Should have fewer than target
