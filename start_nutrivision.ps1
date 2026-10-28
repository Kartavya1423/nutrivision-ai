# =====================================================================
#           NutriVision AI — PowerShell Production Launcher
# =====================================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "          NutriVision AI — Launch & Health Orchestrator               " -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "[1/5] Working Directory: $ScriptDir" -ForegroundColor Gray

# 2. Check Python
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    Write-Host "[ERROR] Python was not found in PATH!" -ForegroundColor Red
    exit 1
}
Write-Host "[2/5] Python Interpreter: $PythonExe" -ForegroundColor Gray

# 3. Pre-Flight Dependency Verification
$DepCheck = & python -c "import fastapi, uvicorn, streamlit, torch, cv2, ultralytics; print('OK')" 2>$null
if ($DepCheck -ne "OK") {
    Write-Host "[WARNING] Dependencies missing or incomplete. Attempting check..." -ForegroundColor Yellow
    & python -c "import fastapi, uvicorn, streamlit, torch, cv2, ultralytics"
    Write-Host "Please run: pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}
Write-Host "[3/5] Critical Dependencies: Verified (PyTorch, Ultralytics, OpenCV, Streamlit, FastAPI)" -ForegroundColor Green

# 4. Check Port 8000 & Launch Backend
Write-Host "[4/5] Checking FastAPI Backend on port 8000..." -ForegroundColor Gray
$PortInUse = $false
try {
    $Tcp = New-Object System.Net.Sockets.TcpClient
    $Tcp.Connect("127.0.0.1", 8000)
    $Tcp.Close()
    $PortInUse = $true
} catch {
    $PortInUse = $false
}

if ($PortInUse) {
    try {
        $Health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 3 -ErrorAction Stop
        Write-Host "[PASS] Existing Backend is healthy and responding! (Device: $($Health.device))" -ForegroundColor Green
    } catch {
        Write-Host "[WARNING] Port 8000 is occupied but /health check did not return HTTP 200." -ForegroundColor Yellow
    }
} else {
    Write-Host "[INFO] Starting FastAPI Backend in separate process..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ScriptDir'; python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
    
    Write-Host "Polling http://127.0.0.1:8000/health for readiness..." -ForegroundColor Gray
    $Attempts = 0
    $BackendReady = $false
    while ($Attempts -lt 25 -and -not $BackendReady) {
        Start-Sleep -Seconds 1
        $Attempts++
        try {
            $Health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2 -ErrorAction Stop
            $BackendReady = $true
            Write-Host "[PASS] Backend is live at http://127.0.0.1:8000 (Ready in $Attempts s)" -ForegroundColor Green
        } catch {
            Write-Host -NoNewline "."
        }
    }
    if (-not $BackendReady) {
        Write-Host "`n[ERROR] Backend failed to report ready within 25 seconds." -ForegroundColor Red
        exit 1
    }
}

# 5. Launch Streamlit Frontend
Write-Host "`n[5/5] Launching Streamlit Web Dashboard..." -ForegroundColor Green
Write-Host "Dashboard URL: http://127.0.0.1:8501" -ForegroundColor Cyan
Write-Host "OpenAPI Docs:  http://127.0.0.1:8000/docs`n" -ForegroundColor Cyan

python -m streamlit run frontend/streamlit_app.py --server.port 8501
