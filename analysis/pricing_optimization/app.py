"""Compatibility entry point for hosts configured to launch this legacy path.

The canonical Streamlit app lives at the repository root. This shim keeps older
deployment settings working without duplicating the application or its data.
"""
from pathlib import Path
import os
import runpy
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_APP = REPO_ROOT / "app.py"

if not ROOT_APP.is_file():
    raise FileNotFoundError(f"Expected Streamlit entry point at {ROOT_APP}")

# Streamlit Community Cloud and the devcontainer launch this legacy path.
# The dataset and imports live at the repository root.
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
runpy.run_path(str(ROOT_APP), run_name="__main__")
