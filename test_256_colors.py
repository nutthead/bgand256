#!/usr/bin/env python3
"""Test script to debug why only 128 colors are generated instead of 256."""

from bgand256.contrast_algorithms import generate_contrasting_colors

# Test with different backgrounds and algorithms
backgrounds = [
    (0.2, 0.3, 0.4),  # Dark gray-blue
    (0.5, 0.5, 0.5),  # Mid gray
    (0.8, 0.8, 0.8),  # Light gray
]

algorithms = ["delta-e", "cam16ucs", "hsl-greedy"]

for bg in backgrounds:
    print(f"\nBackground RGB: {bg}")
    print("-" * 50)
    
    for algorithm in algorithms:
        colors = generate_contrasting_colors(bg, 256, algorithm)
        print(f"{algorithm}: Generated {len(colors)} colors")
        
        # Check if we hit exactly 128 (which would be target_count // 2)
        if len(colors) == 128:
            print(f"  WARNING: Hit exactly 128 colors (target_count // 2)")