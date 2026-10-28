@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo           NutriVision AI — Production Launcher & Health Monitor
echo =====================================================================
echo.

:: 1. Navigate to Project Directory
cd /d "%~dp0"
echo [1/5] Working directory: %CD%

:: 2. Check Python Interpreter
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please ensure Python 3.10+ is installed and added to PATH.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_EXE=%%i
echo [2/5] Python interpreter: %PYTHON_EXE%

:: 3. Check Critical Dependencies
python -c "import fastapi, uvicorn, streamlit, torch, cv2, ultralytics; print('Dependencies OK')" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some dependencies might be missing or broken.
    echo Attempting pre-flight import check:
    python -c "import fastapi, uvicorn, streamlit, torch, cv2, ultralytics"
    echo.
    echo Please install dependencies via: pip install -r requirements.txt
    pause
    exit /b 1
)
echo [3/5] Critical dependencies verified (FastAPI, Uvicorn, Streamlit, PyTorch, OpenCV, Ultralytics).

:: 4. Launch Backend API on Port 8000 (if not already running)
echo [4/5] Checking FastAPI backend on port 8000...
powershell -NoProfile -Command "$client = New-Object System.Net.Sockets.TcpClient; try { $client.Connect('127.0.0.1', 8000); exit 0 } catch { exit 1 }" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Port 8000 is already active. Checking /health...
    powershell -NoProfile -Command "try { $res = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 3; if ($res.status -eq 'healthy' -or $res.status -eq 'online') { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [PASS] Existing backend is healthy and responding!
    ) else (
        echo [WARNING] Port 8000 is in use but /health check failed.
        echo Please ensure no other conflicting software is using port 8000.
    )
) else (
    echo [INFO] Starting FastAPI Backend in separate terminal...
    start "NutriVision AI — FastAPI Backend (Port 8000)" powershell -NoExit -Command "cd '%CD%'; python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
    
    echo Waiting for backend to become ready...
    set /a attempts=0
    :wait_backend
    set /a attempts+=1
    powershell -NoProfile -Command "try { $res = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [PASS] FastAPI Backend is live at http://127.0.0.1:8000
    ) else (
        if !attempts! geq 25 (
            echo [ERROR] Backend timed out after 25 attempts.
            echo Please inspect the backend console window for error tracebacks.
            pause
            exit /b 1
        )
        timeout /t 1 /nobreak >nul
        goto wait_backend
    )
)

:: 5. Launch Streamlit Frontend on Port 8501
echo [5/5] Launching Streamlit Web Dashboard...
echo Dashboard URL: http://127.0.0.1:8501
echo OpenAPI Docs:  http://127.0.0.1:8000/docs
echo.
python -m streamlit run frontend/streamlit_app.py --server.port 8501

pause
