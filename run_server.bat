@echo off
rem FastAPI server for the furniture site (http://127.0.0.1:8000)
set PYTHONDONTWRITEBYTECODE=1
cd /d "%~dp0backend"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
