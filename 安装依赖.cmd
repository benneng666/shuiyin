@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 (
    echo 请安装 Python 3.11 或更新版本，并加入 PATH。
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo 依赖安装失败，请检查网络。
  pause
  exit /b 1
)
echo 安装完成。默认使用本机 Microsoft Edge 浏览器。
pause
