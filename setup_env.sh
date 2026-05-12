#!/usr/bin/env bash
# Set up a clean virtual environment for the credit-scoring experiments.
# Run this once before anything else.
#
# Usage:
#   bash setup_env.sh
#
# Then in each new terminal session:
#   source .venv/bin/activate     (Linux/Mac)
#   .\.venv\Scripts\activate      (Windows PowerShell)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "[1/4] Creating virtual environment in .venv ..."
python -m venv .venv

echo "[2/4] Activating virtual environment ..."
# shellcheck disable=SC1091
source .venv/bin/activate || source .venv/Scripts/activate

echo "[3/4] Upgrading pip ..."
python -m pip install --upgrade pip

echo "[4/4] Installing requirements ..."
pip install -r requirements.txt

echo ""
echo "Done. To activate in future sessions:"
echo "  Linux/Mac:        source $PROJECT_ROOT/.venv/bin/activate"
echo "  Windows PowerShell: $PROJECT_ROOT\\.venv\\Scripts\\Activate.ps1"
