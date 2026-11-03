@echo off
setlocal

cd /d "%~dp0"

if not exist venv (
    echo Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

rem With no arguments: the window. With arguments: the command-line
rem captioner, which takes them unchanged (run.bat <folder> --help).
if "%~1"=="" (
    python taggui\run_gui.py
) else (
    python taggui\run_cli.py %*
)
