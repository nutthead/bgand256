#!/usr/bin/env python3
"""Simple test to debug performance issue."""

import time
import sys
sys.path.insert(0, 'src')

from bgand256.contrast_algorithms import _generate_wcag_compliant_candidates

# Convert #faf4ed to RGB [0,1]
background = (0xfa/255, 0xf4/255, 0xed/255)  # #faf4ed
print(f"Background: RGB{tuple(int(c*255) for c in background)}")

# Test candidate generation
print("\nGenerating candidates...")
start = time.time()
candidates = _generate_wcag_compliant_candidates(background, 4.5)
elapsed = time.time() - start
print(f"Generated {len(candidates)} candidates in {elapsed:.2f}s")

# Show some candidates
print("\nFirst 5 candidates:")
for i, c in enumerate(candidates[:5]):
    rgb255 = tuple(int(x*255) for x in c)
    print(f"  {i}: RGB{rgb255}")
    
print("\nLast 5 candidates:")  
for i, c in enumerate(candidates[-5:]):
    rgb255 = tuple(int(x*255) for x in c)
    print(f"  {i}: RGB{rgb255}")