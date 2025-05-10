REM Description: save the current package list to requirements.txt.
REM Path: exit.bat

@echo off
pip freeze > requirements.txt
deactivate
echo Virtual environment deactivated and requirements.txt updated.
