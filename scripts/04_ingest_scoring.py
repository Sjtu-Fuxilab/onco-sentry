"""
SENTRY-MH · Script 04: Ingest + Scoring + Adjudication
======================================================
Ingests rater CSVs, computes agreement metrics, and performs adjudication.

Usage:
    python scripts/04_ingest_scoring.py
"""
from __future__ import annotations
import os, sys, json, glob, warnings, re
from pathlib import Path
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

# [INSERT YOUR SCRIPT 04 CODE HERE]