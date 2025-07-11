import os
import subprocess
import sys

# Always use Python 3.9 via the Windows py launcher
PYTHON39_LAUNCHER = ["py", "-3.9"]
VENV_DIR = "venv"
VENV_ACTIVATE = os.path.join(VENV_DIR, "Scripts", "activate")
REQUIREMENTS = "requirements.txt"

# Step 1: Check if Python 3.9 is available
try:
    python39_path = (
        subprocess.check_output(
            PYTHON39_LAUNCHER + ["-c", "import sys; print(sys.executable)"]
        )
        .decode()
        .strip()
    )
except subprocess.CalledProcessError:
    print(
        "Error: Python 3.9 is required but not found. Please install Python 3.9 and ensure it is available via the 'py -3.9' launcher."
    )
    sys.exit(1)

# Step 2: Create virtual environment with Python 3.9
if not os.path.exists(VENV_DIR):
    subprocess.check_call(PYTHON39_LAUNCHER + ["-m", "venv", VENV_DIR])
else:
    print(f"Virtual environment '{VENV_DIR}' already exists.")

# Step 3: Install torch, torchvision, torchaudio with CUDA 12.1
pip_exe = os.path.join(VENV_DIR, "Scripts", "pip.exe")
subprocess.check_call(
    [
        pip_exe,
        "install",
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    ]
)

# Step 4: Install requirements.txt
subprocess.check_call([pip_exe, "install", "-r", REQUIREMENTS])

print("\nInstallation complete. To activate the environment, run:")
print(f"  {VENV_DIR}\\Scripts\\activate")
