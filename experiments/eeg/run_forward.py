# experiments/forward_sanity.py
from __future__ import annotations

import argparse

from experiments.lib.io import load_yaml, save_yaml, make_run_dir, save_npz, save_json
from experiments.lib.utils import build_model_numpy, build_design_numpy
from experiments.lib.plotting import plot_and_save_separate, plot_summary

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"