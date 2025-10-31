"""
SENTRY-MH · Script 00: Project Setup
====================================
Creates directory structure, configuration files, and SMS rubric.

Usage:
    python setup/setup_project.py [optional_root_path]

Environment Variables:
    SENTRY_ROOT: Override default project root location
"""
from __future__ import annotations
import os, sys, json, csv, textwrap, datetime
from pathlib import Path

def STAMP():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# TODO: implementation to be added (placeholder removed)
# print('Script 00 complete')
