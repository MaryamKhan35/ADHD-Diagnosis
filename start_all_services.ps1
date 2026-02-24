# EEG ADHD Detection System - Complete Startup Script
# Starts MySQL, Backend, and Frontend in separate windows

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "EEG ADHD Detection System - Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start MySQL Database
Write-Host "Starting MySQL Database..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
  "cd `"$env:USERPROFILE\Downloads\mysql-9.6.0-winx64\mysql-9.6.0-winx64\bin`"; .\mysqld.exe --basedir=`"$env:USERPROFILE\Downloads\mysql-9.6.0-winx64\mysql-9.6.0-winx64`" --datadir=`"$env:USERPROFILE\Downloads\mysql-data`" --port=3306"
Start-Sleep -Seconds 3

# Start Backend (FastAPI)
Write-Host "Starting Backend API (FastAPI)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
  "cd 'c:\Users\Maryam\Downloads\WebApp Front-End'; .\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"
Start-Sleep -Seconds 2

# Start Frontend (Vite + React)
Write-Host "Starting Frontend (Vite + React)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
  "cd 'c:\Users\Maryam\Downloads\WebApp Front-End\frontend'; npm run dev -- --host 0.0.0.0 --port 5173"
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "All services started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Access your application:" -ForegroundColor Cyan
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor Green
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor Green
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "MySQL database running on port 3306" -ForegroundColor Green
Write-Host ""
Write-Host "Note: Each service is running in a separate window." -ForegroundColor Yellow
Write-Host "Close any window to stop that service." -ForegroundColor Yellow
