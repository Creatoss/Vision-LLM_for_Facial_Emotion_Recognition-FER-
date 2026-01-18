@echo off
REM ============================================================
REM Streamlit Emotion Recognition Dashboard - Startup Script
REM Windows Batch File
REM ============================================================

echo.
echo ========================================
echo  Emotion Recognition Dashboard
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version

REM Check if virtual environment exists
if exist "venv\" (
    echo [2/4] Virtual environment found, activating...
    call venv\Scripts\activate.bat
) else (
    echo [2/4] Virtual environment not found
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
)

REM Check if requirements are installed
echo [3/4] Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements_streamlit.txt
)

REM Check if model exists
echo [4/4] Checking model files...
if not exist "blip2-emotion-rafce-final\" (
    echo.
    echo Warning: Fine-tuned model not found!
    echo Please download from Google Drive:
    echo   /content/drive/MyDrive/blip2-emotion-rafce-final
    echo And extract to: blip2-emotion-rafce-final/
    echo.
    pause
)

REM Launch dashboard
echo.
echo Starting Streamlit dashboard...
echo.
echo Dashboard will open at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

python -m streamlit run streamlit_app.py

pause
