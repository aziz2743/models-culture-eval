#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# Check if the script is running as root
if [ "$EUID" -eq 0 ]; then
    SUDO=""
    echo "Running as root. Skipping sudo."
else
    SUDO="sudo"
    echo "Running as user. Using sudo when necessary."
fi

# 1. Install nano and python3-venv (required on Ubuntu to create virtual environments)
echo "==> Updating package lists and installing nano & python3-venv..."
$SUDO apt update
$SUDO apt install -y nano python3-venv python3-pip

# 2. Make the virtual environment
VENV_NAME=".venv"
echo "==> Creating virtual environment in '$VENV_NAME'..."
python3 -m venv "$VENV_NAME"

# 3. Activate the virtual environment
echo "==> Activating virtual environment..."
source "$VENV_NAME/bin/activate"

echo "==> Done! Virtual environment is active."
echo "Python path: $(which python)"