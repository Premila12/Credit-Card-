@echo off
echo Setting up Virtual Environment...

:: Create venv if it doesn't exist
if not exist "venv" (
    python -m venv venv
    echo Virtual environment 'venv' created.
) else (
    echo 'venv' already exists.
)

:: Activate venv and install requirements
echo Activating venv and installing dependencies...
call venv\Scripts\activate
pip install -r requirements.txt

echo.
echo Setup Complete!
echo To run the app, use: streamlit run src/app.py
pause
