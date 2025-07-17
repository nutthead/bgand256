#!/usr/bin/env python3
"""Test the improved greedy algorithm performance."""

import time
import sys
sys.path.insert(0, 'src')

from bgand256.contrast_algorithms import (
    generate_contrasting_colors,
    _generate_wcag_compliant_candidates,
)

# Convert #faf4ed to RGB [0,1]
background = (0xfa/255, 0xf4/255, 0xed/255)  # #faf4ed
print(f"Testing with background color: RGB{tuple(int(c*255) for c in background)}")
print("=" * 60)

# Generate candidates
print("\nGenerating WCAG-compliant candidates...")
start = time.time()
candidates = _generate_wcag_compliant_candidates(background, 4.5)
candidate_time = time.time() - start
print(f"Generated {len(candidates)} candidates in {candidate_time:.2f}s")

# Test different configurations
test_configs = [
    (10, "hsl-greedy"),
    (50, "hsl-greedy"), 
    (100, "hsl-greedy"),
    (256, "hsl-greedy"),
    (256, "delta-e"),
    (256, "cam16ucs"),
]

for target_count, algorithm in test_configs:
    print(f"\nGenerating {target_count} colors using {algorithm} algorithm...")
    start = time.time()
    colors = generate_contrasting_colors(background, target_count, algorithm)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Colors generated: {len(colors)}")
    
    # Check quality - minimum distance between first few colors
    if len(colors) >= 2:
        from bgand256.contrast_algorithms import _get_distance_function
        dist_fn = _get_distance_function(algorithm)
        min_dist = float('inf')
        for i in range(min(5, len(colors))):
            for j in range(i+1, min(5, len(colors))):
                dist = dist_fn(colors[i], colors[j])
                min_dist = min(min_dist, dist)
        print(f"  Min distance (first 5): {min_dist:.3f}")

# Demonstrate the color palette
print("\n" + "="*60)
print("Sample colors from the palette (RGB values):")
print("="*60)
colors = generate_contrasting_colors(background, 20, "hsl-greedy")
for i, color in enumerate(colors[:10]):
    rgb = tuple(int(c*255) for c in color)
    print(f"Color {i+1:2d}: RGB{rgb}")