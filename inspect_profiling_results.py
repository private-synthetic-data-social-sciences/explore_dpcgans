#!/usr/bin/env python
"""Inspect results of the profiler"""

import pstats
from pstats import SortKey

profiling_output = "profiling_10kobs.txt"

p = pstats.Stats(profiling_output)
p.strip_dirs().sort_stats(-1).print_stats()

# or 
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(20)
