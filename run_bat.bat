@echo off
setlocal

set "ROOT=%~dp0"

pushd "%ROOT%GPT-SoVITS"
start "GPT-SoVITS API" cmd /K "python run_api.py"
popd

cd /d "%ROOT%"
python run.py

pause

