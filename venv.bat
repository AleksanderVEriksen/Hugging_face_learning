REM Description: This script sets up a Python virtual environment and installs required packages.
REM Usage: venv.bat

@echo off


.\.venv\Scripts\Activate

pip install --upgrade pip
pip install -r requirements.txt
