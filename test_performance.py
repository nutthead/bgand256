#!/usr/bin/env python3
"""Test performance of current algorithm."""

import time
from bgand256.contrast_algorithms import (
    _generate_wcag_compliant_candidates,
    _greedy_select_contrasting_colors,
    generate_contrasting_colors
)

# Convert #faf4ed to RGB [0,1]
background = (0xfa/255, 0xf4/255, 0xed/255)  # #faf4ed

print(f"Testing with background color: {background}")
print(f"RGB 255: ({int(background[0]*255)}, {int(background[1]*255)}, {int(background[2]*255)})")

# Test candidate generation
start = time.time()
candidates = _generate_wcag_compliant_candidates(background, 4.5)
candidate_time = time.time() - start
print(f"\nCandidate generation took: {candidate_time:.2f}s")
print(f"Number of candidates: {len(candidates)}")

# Test greedy selection with different target counts
for target_count in [10, 50, 100, 256]:
    print(f"\nTesting greedy selection for {target_count} colors...")
    start = time.time()
    selected = _greedy_select_contrasting_colors(
        candidates[:1000],  # Limit candidates for testing
        target_count, 
        "hsl-greedy"
    )
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Selected: {len(selected)} colors")
    
# Test full algorithm
print("\nTesting full algorithm for 256 colors...")
start = time.time()
colors = generate_contrasting_colors(background, 256, "hsl-greedy")
total_time = time.time() - start
print(f"Total time: {total_time:.2f}s")
print(f"Colors generated: {len(colors)}")