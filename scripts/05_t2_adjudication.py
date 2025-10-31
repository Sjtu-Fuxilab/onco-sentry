"""
SENTRY-MH · Script 05: T2 Adjudication from Per-Rater Files
===========================================================
Majority voting with safety locks for high-severity vignettes.

Usage:
    python scripts/05_t2_adjudication.py
"""
import os, json, time
from pathlib import Path
from itertools import combinations
from datetime import datetime
import numpy as np
import pandas as pd

# [INSERT YOUR SCRIPT 05 CODE HERE]