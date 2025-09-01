@echo off
REM ------------------------------------------------------------------
REM AI Lover Service - Discord Bot 獨立啟動腳本
REM ------------------------------------------------------------------

REM 自動切換到批次檔所在的目錄
cd /d "%~dp0"

echo.
echo =======================================================
echo      AI Lover Service - Discord Bot Mode
echo =======================================================
echo.
echo 正在啟動 Discord Bot 服務...
echo.

REM 使用系統的 python 來執行 main.py，並明確指定 'discord' 模式
python main.py discord

echo.
echo 服務已停止或發生錯誤。
pause