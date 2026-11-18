@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal enabledelayedexpansion
rem 自动选用 dist 下最近更新的发行目录，升级版本时不必再改本脚本。
set "APP="
for /f "delims=" %%D in ('dir /b /ad /o-d "dist\买家秀采集助手-*" 2^>nul') do (
  if not defined APP if exist "dist\%%D\买家秀采集助手.exe" set "APP=dist\%%D\买家秀采集助手.exe"
)
if defined APP (
  start "" "!APP!"
  exit /b 0
)
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "%~dp0desktop.py"
  exit /b 0
)
echo 请解压完整的软件压缩包后运行其中的 EXE，或先安装源码依赖。
pause
