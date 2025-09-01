@echo off
rem AI Lover 終極診斷腳本 (v1.0)
rem 此腳本會將所有輸出記錄到 diag_log.txt

rem 設定環境為 UTF-8
chcp 65001

rem 切換到腳本所在目錄
cd /d "%~dp0"

rem --- 步驟 1: 清理舊日誌並建立新檔案 ---
if exist "diag_log.txt" (
    del "diag_log.txt"
)
echo AI Lover 診斷日誌 > diag_log.txt
echo ======================================== >> diag_log.txt
echo 開始時間: %date% %time% >> diag_log.txt
echo. >> diag_log.txt


rem --- 步驟 2: 檢查虛擬環境是否存在 ---
echo 正在檢查虛擬環境...
if not exist ".venv\Scripts\python.exe" (
    echo [錯誤] 找不到 .venv\Scripts\python.exe！請先執行 start.bat 完成首次安裝。 >> diag_log.txt
    echo [錯誤] 找不到 .venv\Scripts\python.exe！請先執行 start.bat 完成首次安裝。
    goto end_script
)
echo [成功] 找到虛擬環境。 >> diag_log.txt
echo [成功] 找到虛擬環境。


rem --- 步驟 3: 檢查虛擬環境中的 Python 版本 ---
echo. >> diag_log.txt
echo --- 正在檢查 Python 版本 --- >> diag_log.txt
echo 正在檢查虛擬環境中的 Python 版本...
rem '2>&1' 會將標準錯誤(stderr)和標準輸出(stdout)都重定向到日誌檔案
.venv\Scripts\python.exe --version >> diag_log.txt 2>&1
echo [完成] Python 版本檢查完畢，結果已寫入 diag_log.txt。


rem --- 步驟 4: 執行主程式並記錄所有輸出 ---
echo. >> diag_log.txt
echo --- 正在執行 main.py web 模式 --- >> diag_log.txt
echo 正在嘗試以 web 模式執行 main.py，所有輸出都將被記錄...
rem 這是最關鍵的一步：將所有輸出（包括所有錯誤）都強制寫入日誌檔案
.venv\Scripts\python.exe main.py web >> diag_log.txt 2>&1


rem --- 步驟 5: 腳本結束 ---
:end_script
echo. >> diag_log.txt
echo 結束時間: %date% %time% >> diag_log.txt
echo ======================================== >> diag_log.txt

echo.
echo 診斷腳本執行完畢。
echo 請打開您專案資料夾中的 'diag_log.txt' 檔案。
echo 然後將該檔案的【全部內容】複製並貼上給我。
pause