@echo off
REM AI-Based Shipping Line Tracking System
REM Run script for Windows
REM Usage: run.bat <booking_id>

setlocal enabledelayedexpansion

REM Check if booking ID is provided
if "%~1"=="" (
    echo [ERROR] No booking ID provided
    echo Usage: %0 ^<booking_id^>
    echo Example: %0 SINI25432400
    exit /b 1
)

set BOOKING_ID=%1

echo [INFO] Starting AI-Based Shipping Line Tracking System
echo [INFO] Booking ID: %BOOKING_ID%

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo [INFO] Please install Python 3.8+ from https://python.org
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" if not exist ".venv" (
    echo [WARNING] No virtual environment found
    echo [INFO] Creating virtual environment...

    where uv >nul 2>&1
    if !errorlevel! equ 0 (
        echo [INFO] Using uv to create virtual environment
        uv venv --python 3.11
        if exist ".venv" (
            call .venv\\Scripts\\activate.bat
        )
    ) else (
        echo [INFO] Using standard venv
        python -m venv venv
        call venv\\Scripts\\activate.bat
    )

    echo [SUCCESS] Virtual environment created
) else (
    echo [INFO] Activating virtual environment...
    if exist ".venv" (
        call .venv\\Scripts\\activate.bat
    ) else if exist "venv" (
        call venv\\Scripts\\activate.bat
    )
    echo [SUCCESS] Virtual environment activated
)

REM Check if requirements are installed
echo [INFO] Checking dependencies...
python -c "import browser_use" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Dependencies not found, installing...

    where uv >nul 2>&1
    if !errorlevel! equ 0 (
        echo [INFO] Installing dependencies with uv...
        uv pip install -r requirements.txt
    ) else (
        echo [INFO] Installing dependencies with pip...
        pip install -r requirements.txt
    )

    echo [INFO] Installing Playwright browsers...
    playwright install

    echo [SUCCESS] Dependencies installed
) else (
    echo [SUCCESS] Dependencies already installed
)

REM Check if .env file exists
if not exist ".env" (
    echo [WARNING] No .env file found
    if exist ".env.example" (
        echo [INFO] Copying .env.example to .env
        copy ".env.example" ".env" >nul
        echo [WARNING] Please edit .env file and add your API key(s) for the desired LLM provider (e.g., GROQ_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY)
        echo [INFO] Then run this script again
        exit /b 1
    ) else (
        echo [ERROR] No .env.example file found
        exit /b 1
    )
)

REM Run the tracking system
echo [INFO] Starting shipment tracking...
echo [INFO] This may take 30-60 seconds depending on website response time

python agent.py %BOOKING_ID%
if errorlevel 1 (
    echo [ERROR] Tracking failed. Check the logs for details.
    exit /b 1
) else (
    echo [SUCCESS] Tracking completed successfully!
)
