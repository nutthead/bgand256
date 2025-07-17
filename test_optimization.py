#!/usr/bin/env python3
"""Compare performance of original vs optimized algorithms."""

import time
import sys
sys.path.insert(0, 'src')

from bgand256.contrast_algorithms import (
    generate_contrasting_colors,
    _greedy_select_contrasting_colors,
    _generate_wcag_compliant_candidates,
)
from bgand256.contrast_algorithms_optimized import (
    generate_contrasting_colors_optimized,
    _greedy_select_contrasting_colors_optimized,
    _greedy_select_contrasting_colors_matrix,
    _greedy_select_contrasting_colors_hybrid,
)

# Convert #faf4ed to RGB [0,1]
background = (0xfa/255, 0xf4/255, 0xed/255)  # #faf4ed

print(f"Testing with background color: RGB{tuple(int(c*255) for c in background)}")
print("=" * 60)

# Generate candidates once
print("\nGenerating WCAG-compliant candidates...")
start = time.time()
candidates = _generate_wcag_compliant_candidates(background, 4.5)
candidate_time = time.time() - start
print(f"Generated {len(candidates)} candidates in {candidate_time:.2f}s")

# Test different target counts
test_configs = [
    (10, candidates[:500]),
    (50, candidates[:1000]),
    (100, candidates[:2000]),
    (256, candidates),
]

for target_count, test_candidates in test_configs:
    print(f"\n{'='*60}")
    print(f"Testing with {target_count} target colors from {len(test_candidates)} candidates")
    print(f"{'='*60}")
    
    # Test original algorithm (with timeout)
    print("\nOriginal greedy algorithm:")
    start = time.time()
    try:
        if target_count <= 50:  # Only test small counts for original
            selected_orig = _greedy_select_contrasting_colors(
                test_candidates, target_count, "hsl-greedy"
            )
            orig_time = time.time() - start
            print(f"  Time: {orig_time:.2f}s")
            print(f"  Selected: {len(selected_orig)} colors")
        else:
            print("  Skipped (too slow for large counts)")
            orig_time = None
    except KeyboardInterrupt:
        orig_time = time.time() - start
        print(f"  Interrupted after {orig_time:.2f}s")
        orig_time = None
    
    # Test optimized KD-tree algorithm
    print("\nOptimized KD-tree algorithm:")
    start = time.time()
    selected_kd = _greedy_select_contrasting_colors_optimized(
        test_candidates, target_count, "hsl-greedy"
    )
    kd_time = time.time() - start
    print(f"  Time: {kd_time:.2f}s")
    print(f"  Selected: {len(selected_kd)} colors")
    
    # Test matrix algorithm
    print("\nMatrix-based algorithm:")
    start = time.time()
    selected_matrix = _greedy_select_contrasting_colors_matrix(
        test_candidates, target_count, "hsl-greedy"
    )
    matrix_time = time.time() - start
    print(f"  Time: {matrix_time:.2f}s")
    print(f"  Selected: {len(selected_matrix)} colors")
    
    # Test hybrid algorithm
    print("\nHybrid algorithm:")
    start = time.time()
    selected_hybrid = _greedy_select_contrasting_colors_hybrid(
        test_candidates, target_count, "hsl-greedy"
    )
    hybrid_time = time.time() - start
    print(f"  Time: {hybrid_time:.2f}s")
    print(f"  Selected: {len(selected_hybrid)} colors")
    
    # Calculate speedup
    if orig_time:
        print(f"\nSpeedup vs original:")
        print(f"  KD-tree: {orig_time/kd_time:.1f}x faster")
        print(f"  Matrix:  {orig_time/matrix_time:.1f}x faster")
        print(f"  Hybrid:  {orig_time/hybrid_time:.1f}x faster")

# Test full algorithm
print(f"\n{'='*60}")
print("Testing full algorithm for 256 colors")
print(f"{'='*60}")

print("\nOptimized full algorithm:")
start = time.time()
colors_opt = generate_contrasting_colors_optimized(background, 256, "hsl-greedy")
opt_total_time = time.time() - start
print(f"  Total time: {opt_total_time:.2f}s")
print(f"  Colors generated: {len(colors_opt)}")

print("\nTesting color quality...")
# Verify minimum distance between colors
from bgand256.contrast_algorithms import _calculate_hsl_perceptual_distance
min_dist = float('inf')
for i in range(min(10, len(colors_opt))):
    for j in range(i+1, min(10, len(colors_opt))):
        dist = _calculate_hsl_perceptual_distance(colors_opt[i], colors_opt[j])
        min_dist = min(min_dist, dist)
print(f"  Minimum distance between first 10 colors: {min_dist:.3f}")