import os
import subprocess
import sys

VENV_DIR = "venv"
PYTHON_EXE = os.path.join(VENV_DIR, "Scripts", "python.exe")
SCRIPT = "finetune_sketch_classifier_og.py"

if not os.path.exists(PYTHON_EXE):
    print(
        f"Error: {PYTHON_EXE} not found. Please run install.py first to set up the virtual environment."
    )
    sys.exit(1)

subprocess.check_call([PYTHON_EXE, SCRIPT])
