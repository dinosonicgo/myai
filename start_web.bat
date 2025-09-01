@echo off
rem 啟動 AI 戀人程式的 Web 介面 (v1.4 - 最終穩定版)
rem 更新紀錄:
rem v1.4 (2050-08-03): [根本性修正] 使用 'start' 命令在新視窗中啟動 Python。這可以將 Python 進程與批次檔進程隔離，從而避免因底層驅動或安全軟體衝突導致的 cmd 視窗崩潰（閃退）問題。
rem v1.3 (2050-08-01): [診斷] 將條件式暫停改為無條件 'pause'。
rem v1.2 (2050-08-01): [健壯性] 移除了 '> nul'。

chcp 65001
echo 正在準備啟動 AI戀人程式 (僅 Web 介面)...
echo.

rem 切換到批次檔所在的目錄
cd /d "%~dp0"

rem 檢查虛擬環境是否存在
if not exist ".venv" (
    echo 錯誤：找不到虛擬環境 (.venv) 資料夾。
    echo 請先運行一次 start.bat 來完成首次安裝。
    pause
    exit /b
)

echo 正在檢查並安裝必要的依賴套件...
.venv\Scripts\pip install -r requirements.txt
echo 依賴套件檢查完畢。
echo.

echo 即將在新視窗中啟動 Web 伺服器...
echo Web介面將在 http://localhost:8000 運行
echo.

rem [修正] 使用 start 命令在新視窗中運行 Python，並為新視窗指定標題
rem 這可以徹底解決因輸出衝突導致的閃退問題
start "AI Lover Web" .venv\Scripts\python.exe main.py web

echo 啟動指令已發送。主伺服器正在新的視窗中運行。
echo 這個啟動視窗現在可以關閉了。
timeout /t 5 > nul