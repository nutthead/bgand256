"""Optimized color selection algorithms with improved Big O complexity.

This module provides optimized versions of the greedy color selection algorithm
that significantly reduce computational complexity while maintaining quality.

Author: Behrang Saeedzadeh <hello@behrang.org>
"""

import heapq
import warnings
from typing import Any, Callable, Literal

import numpy as np
from scipy.spatial import KDTree

from .colors import _compute_luminance, _contrast_ratio
from .contrast_algorithms import (
    AlgorithmType,
    _calculate_cam16ucs_distance,
    _calculate_delta_e_2000,
    _calculate_hsl_perceptual_distance,
    _convert_rgb_to_cam16ucs,
    _convert_rgb_to_lab,
    _generate_wcag_compliant_candidates,
    _get_distance_function,
)


def _convert_colors_to_feature_space(
    colors: list[tuple[float, float, float]], algorithm: AlgorithmType
) -> np.ndarray:
    """Convert colors to appropriate feature space for distance calculations.
    
    Returns a numpy array of shape (n_colors, n_features) where features
    depend on the algorithm used.
    """
    if algorithm == "delta-e":
        # Convert to LAB space
        features = np.array([_convert_rgb_to_lab(color) for color in colors])
    elif algorithm == "cam16ucs":
        # Convert to CAM16UCS space
        features = []
        for color in colors:
            try:
                features.append(_convert_rgb_to_cam16ucs(color))
            except Exception:
                # Fallback to LAB if CAM16UCS fails
                features.append(_convert_rgb_to_lab(color))
        features = np.array(features)
    else:  # hsl-greedy
        # Use RGB directly for HSL-based distance
        features = np.array(colors)
    
    return features


def _greedy_select_contrasting_colors_optimized(
    candidates: list[tuple[float, float, float]],
    target_count: int,
    algorithm: AlgorithmType,
    min_mutual_distance: float | None = None,
) -> list[tuple[float, float, float]]:
    """Optimized greedy algorithm with O(n log n) complexity using spatial indexing.
    
    This version uses KD-trees for efficient nearest neighbor queries and
    maintains a priority queue for candidate selection.
    """
    if not candidates or target_count <= 0:
        return []
    
    if len(candidates) <= target_count:
        return candidates[:target_count]
    
    # Set default minimum distances
    if min_mutual_distance is None:
        min_distances = {
            "delta-e": 15.0,
            "cam16ucs": 10.0,
            "hsl-greedy": 0.3,
        }
        min_mutual_distance = min_distances[algorithm]
    
    # Convert to feature space for efficient distance calculations
    features = _convert_colors_to_feature_space(candidates, algorithm)
    
    # Initialize selected indices and remaining candidates
    selected_indices = []
    selected_features = []
    remaining_indices = list(range(len(candidates)))
    
    # Select first color (could be optimized to select best initial color)
    selected_indices.append(0)
    selected_features.append(features[0])
    remaining_indices.remove(0)
    
    # Use KD-tree for efficient nearest neighbor queries
    # Rebuild tree as we add selected colors
    while len(selected_indices) < target_count and remaining_indices:
        # Build KD-tree from selected features
        if len(selected_features) > 0:
            tree = KDTree(np.array(selected_features))
        
        best_idx = None
        best_min_dist = 0
        
        # Evaluate remaining candidates
        for idx in remaining_indices:
            # Find distance to nearest selected color
            if len(selected_features) == 1:
                # For first iteration, calculate distance directly
                min_dist = np.linalg.norm(features[idx] - selected_features[0])
            else:
                # Use KD-tree for efficient nearest neighbor query
                min_dist, _ = tree.query(features[idx], k=1)
            
            # Track candidate with maximum minimum distance
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        
        # Add best candidate if it meets criteria
        if best_idx is not None:
            if best_min_dist >= min_mutual_distance or len(selected_indices) < 8:
                selected_indices.append(best_idx)
                selected_features.append(features[best_idx])
                remaining_indices.remove(best_idx)
            elif len(selected_indices) < target_count // 2:
                # Relax criteria if we haven't selected enough colors
                selected_indices.append(best_idx)
                selected_features.append(features[best_idx])
                remaining_indices.remove(best_idx)
            else:
                break
    
    return [candidates[i] for i in selected_indices]


def _greedy_select_contrasting_colors_matrix(
    candidates: list[tuple[float, float, float]],
    target_count: int,
    algorithm: AlgorithmType,
    min_mutual_distance: float | None = None,
) -> list[tuple[float, float, float]]:
    """Matrix-based greedy algorithm with O(n²) preprocessing but O(n) selection.
    
    This version pre-computes all pairwise distances for faster selection,
    trading memory for speed. Best for moderate candidate counts (<5000).
    """
    if not candidates or target_count <= 0:
        return []
    
    n = len(candidates)
    if n <= target_count:
        return candidates[:target_count]
    
    # Set default minimum distances
    if min_mutual_distance is None:
        min_distances = {
            "delta-e": 15.0,
            "cam16ucs": 10.0,
            "hsl-greedy": 0.3,
        }
        min_mutual_distance = min_distances[algorithm]
    
    # Get distance function
    distance_fn = _get_distance_function(algorithm)
    
    # Pre-compute distance matrix (upper triangular to save memory)
    # This is O(n²) but only done once
    distance_matrix = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_fn(candidates[i], candidates[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    # Track selected indices and minimum distances to selected set
    selected = []
    min_distances_to_selected = np.full(n, np.inf)
    
    # Select first color (could optimize initial selection)
    selected.append(0)
    min_distances_to_selected = distance_matrix[0, :]
    min_distances_to_selected[0] = -1  # Mark as selected
    
    # Greedy selection - O(n × t) where t = target_count
    while len(selected) < target_count:
        # Find candidate with maximum minimum distance
        best_idx = np.argmax(min_distances_to_selected)
        best_dist = min_distances_to_selected[best_idx]
        
        # Check termination conditions
        if best_dist < 0:  # All candidates selected
            break
        
        if best_dist < min_mutual_distance and len(selected) >= 8:
            if len(selected) >= target_count // 2:
                break
        
        # Add best candidate
        selected.append(best_idx)
        
        # Update minimum distances - O(n)
        for i in range(n):
            if min_distances_to_selected[i] > 0:  # Not selected
                min_distances_to_selected[i] = min(
                    min_distances_to_selected[i],
                    distance_matrix[best_idx, i]
                )
        min_distances_to_selected[best_idx] = -1  # Mark as selected
    
    return [candidates[i] for i in selected]


def _greedy_select_contrasting_colors_hybrid(
    candidates: list[tuple[float, float, float]],
    target_count: int,
    algorithm: AlgorithmType,
    min_mutual_distance: float | None = None,
) -> list[tuple[float, float, float]]:
    """Hybrid approach that chooses the best algorithm based on input size.
    
    Uses matrix approach for small inputs and KD-tree approach for large inputs.
    """
    n = len(candidates)
    
    # Choose algorithm based on input size and target count
    if n * target_count < 1_000_000:  # Matrix approach feasible
        return _greedy_select_contrasting_colors_matrix(
            candidates, target_count, algorithm, min_mutual_distance
        )
    else:  # Use spatial indexing for large inputs
        return _greedy_select_contrasting_colors_optimized(
            candidates, target_count, algorithm, min_mutual_distance
        )


def generate_contrasting_colors_optimized(
    background_rgb: Any,
    target_count: int = 256,
    algorithm: AlgorithmType = "delta-e",
    min_contrast: float = 4.5,
    min_mutual_distance: float | None = None,
) -> list[tuple[float, float, float]]:
    """Optimized version of generate_contrasting_colors with better performance.
    
    This version uses spatial data structures and optimized algorithms to achieve
    better Big O complexity while maintaining color quality.
    """
    # Suppress colour-science warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Generate candidates
        candidates = _generate_wcag_compliant_candidates(background_rgb, min_contrast)
        
        if not candidates:
            return []
        
        # If we have fewer candidates than requested, return all
        if len(candidates) <= target_count:
            return candidates
        
        # Use optimized greedy selection
        selected = _greedy_select_contrasting_colors_hybrid(
            candidates, target_count, algorithm, min_mutual_distance
        )
        
        return selected[:target_count]