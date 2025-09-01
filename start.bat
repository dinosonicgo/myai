@echo off
rem 主啟動檔，同時運行 Web 介面和 Discord Bot (v1.4 - 最終穩定版)
rem 更新紀錄:
rem v1.4 (2050-08-03): [根本性修正] 使用 'start' 命令在新視窗中分別啟動 Web 和 Discord 進程，以解決閃退問題。
rem v1.3 (2050-07-31): [重大修正] 使用 'py' 啟動器代替 'python' 來創建虛擬環境。
rem v1.2 (2050-07-31): 新增 || pause 機制。

chcp 65001
echo 正在啟動 AI戀人 程式...
echo.

rem 切換到批次檔所在的目錄
cd /d "%~dp0"

rem 檢查虛擬環境是否存在，如果不存在則進行首次設置
if not exist ".venv" (
    echo 首次運行，正在設置虛擬環境...
    py -m venv .venv
    
    if not exist ".venv" (
        echo.
        echo 錯誤：虛擬環境創建失敗！
        echo 請確認您已經從 python.org 正確安裝了 Python，並且 'py' 命令可用。
        pause
        exit /b
    )

    echo 正在升級 pip...
    .venv\Scripts\python.exe -m pip install --upgrade pip
    echo 虛擬環境設置完成！
    echo.
)

echo 正在檢查並安裝必要的依賴套件...
.venv\Scripts\pip install -r requirements.txt
echo 依賴套件檢查完畢。
echo.

echo 即將在新視窗中分別啟動 Web 伺服器與 Discord Bot...
echo 請稍候...
echo.

rem [修正] 使用 start 命令分別啟動 Web 和 Discord
start "AI Lover Web" .venv\Scripts\python.exe main.py web
timeout /t 2 > nul
start "AI Lover Discord" .venv\Scripts\python.exe main.py discord

echo 啟動指令已全部發送。
echo Web 伺服器和 Discord Bot 正在各自的新視窗中運行。
echo 這個啟動視窗現在可以關閉了。
timeout /t 5 > nul