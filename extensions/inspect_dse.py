#!/usr/bin/env python3
"""Prints the structure of kvroofline.dse / quality / layerwise outputs (read-only).
Run from the repo root:  python extensions/inspect_dse.py > extensions/outputs/inspect_dse.txt
"""
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from kvroofline import Compression, crossover_S, evaluate, get_hw, get_model, layerwise_dataframe, max_batch
from kvroofline import dse
from kvroofline.dse import Grid, Regime, iso_area_best, pareto_front, pareto_mask, sweep
from kvroofline.quality import QualityTable

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)


def sig(f):
    try:
        return f"{f.__name__}{inspect.signature(f)}"
    except Exception as e:
        return f"{f} (no signature: {e})"


if __name__ == "__main__":
    print("=== signatures ===")
    for f in (sweep, pareto_front, pareto_mask, iso_area_best, layerwise_dataframe, evaluate, crossover_S, max_batch):
        print(sig(f))
