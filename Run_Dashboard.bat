@echo off
setlocal
cd /d "%~dp0"

echo Launching Earnings Intelligence Terminal...

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

start "" http://localhost:8501
streamlit run app.py

pause
