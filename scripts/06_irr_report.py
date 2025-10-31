"""
SENTRY-MH · Script 06: Inter-Rater Reliability Report
====================================================
Generates comprehensive IRR report with Cohen's κ and Gwet's AC1.

Usage:
    python scripts/06_irr_report.py

Options:
    FAST_MODE=True: Faster but larger CIs (default for testing)
    FAST_MODE=False: Slower but more precise estimates
"""
import os, json, glob, textwrap
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# [INSERT YOUR SCRIPT 06 CODE HERE]