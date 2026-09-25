@echo off
setlocal
cd /d "%~dp0"

echo Running Earnings Signal Engine CLI Demo...

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python main.py --tickers NVDA AAPL MSFT --holding-days 5 --ablation --demo

pause
