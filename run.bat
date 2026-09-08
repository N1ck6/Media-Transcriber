@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM Запуск через python из вашего venv.
REM Если путь к venv другой — измените строку ниже.
set PYTHON=C:\whisper\venv\Scripts\python.exe

if not exist "%PYTHON%" (
    echo Python из venv не найден: %PYTHON%
    echo Проверьте путь в bat или установите venv в C:\whisper\venv
    pause
    exit /b 1
)

"%PYTHON%" transcribe.py %*
if errorlevel 1 pause
