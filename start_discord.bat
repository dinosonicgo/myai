@echo off
:: AI Lover 啟動器 (Discord 模式) v1.0
:: 說明: 此批次檔會自動啟動位於同目錄下的 AI 戀人專案，並傳入 'discord' 參數。

:: 設置視窗標題，方便識別
title AI Lover Launcher (Discord Mode)

echo [AI Lover Launcher] 正在準備啟動環境...

:: 核心指令: 切換到此批次檔所在的目錄
:: %~dp0 會自動展開為此檔案的路徑 (例如: D:\...\ai_lover_service\)
:: cd /d 確保即使跨磁碟機也能正確切換
cd /d "%~dp0"

echo [AI Lover Launcher] 當前目錄: %cd%
echo [AI Lover Launcher] 準備執行: python launcher.py discord
echo ----------------------------------------------------

:: 執行主程式
python launcher.py discord

:: 保持視窗開啟，以便查看日誌或錯誤訊息
echo ----------------------------------------------------
echo [AI Lover Launcher] 程式已結束。您可以按任意鍵關閉此視窗。
pause