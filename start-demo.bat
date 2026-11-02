@echo off
REM Starts DeepLearnQ on ports 5080 (API) and 5173 (UI) so it can run alongside BuildSmart (3000/5000/8000).
set PYTHONUTF8=1
set PY=python
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
start "DeepLearnQ backend" cmd /k "cd /d %~dp0 && %PY% backend\app.py"
start "DeepLearnQ frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
echo Waiting for the model to load...
timeout /t 25 >nul
start http://localhost:5173
