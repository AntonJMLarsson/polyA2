"""Pytest setup for polyA2.

Adds the project root to ``sys.path`` so the tests can ``import support_functions``
no matter where ``pytest`` is invoked from.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
