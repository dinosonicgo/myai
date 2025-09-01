# launcher.py 的中文註釋(v2.0 - 在專案根目錄運行)
# 更新紀錄:
# v2.0 (2025-09-01):
# 1. [重大架構重構] 移除了 clone 和切換目錄的邏輯。此版本被設計為直接放置在專案的根目錄下運行。
# 2. [功能簡化] 啟動器現在的核心職責是：確保當前所在的 Git 倉庫是最新版本，然後執行同目錄下的 main.py。
# v1.0 (2025-09-01):
# 1. [全新創建] 創建此啟動器腳本。

import os
import sys
import subprocess
from pathlib import Path

# 函式：執行命令
# 說明：一個輔助函式，用於執行系統命令並處理潛在的錯誤。
def _run_command(command, working_dir=None):
    """執行一個 shell 命令並返回成功與否。"""
    try:
        print(f"▶️ 正在執行: {' '.join(command)}")
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        return True
    except FileNotFoundError:
        print(f"🔥 錯誤: 'git' 命令未找到。")
        print("請確保您已在系統中安裝 Git，並且其路徑已添加到環境變數中。")
        if os.name == 'nt':
            os.system("pause")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"🔥 命令執行失敗: {' '.join(command)}")
        print(f"   標準錯誤: {e.stderr.strip()}")
        if "Authentication failed" in e.stderr:
            print("\n   [可能原因] 您的 GitHub 倉庫是私有的，需要進行身份驗證。")
        if os.name == 'nt':
            os.system("pause")
        sys.exit(1)
# 函式：執行命令

# 函式：主啟動邏輯
# 說明：執行整個更新與啟動流程。
def main():
    """主啟動函式。"""
    current_dir = Path(__file__).resolve().parent

    # 步驟 1: 檢查 Git 是否安裝
    print("--- 步驟 1/3: 檢查 Git 環境 ---")
    if not _run_command(["git", "--version"]):
        return

    # 步驟 2: 更新當前倉庫
    print("\n--- 步驟 2/3: 正在從 GitHub 同步最新程式碼 ---")
    if not _run_command(["git", "pull"], working_dir=current_dir):
        # 如果 pull 失敗，通常是因為本地有未提交的修改。
        # 在此情境下，我們依然嘗試運行，讓使用者決定如何處理。
        print("   ⚠️ 警告: 'git pull' 失敗。可能是因為您在本地修改了檔案。")
        print("      將繼續使用當前的本地版本啟動程式...")
    else:
        print("✅ 程式碼已是最新版本。")

    # 步驟 3: 執行主應用程式
    print(f"\n--- 步驟 3/3: 啟動主應用程式 ---")
    main_py_path = current_dir / "main.py"
    if not main_py_path.is_file():
        print(f"🔥 致命錯誤: 在當前目錄中找不到 'main.py'。")
        if os.name == 'nt':
            os.system("pause")
        sys.exit(1)

    args_to_pass = sys.argv[1:]
    command_to_run = [sys.executable, "main.py"] + args_to_pass

    print(f"🚀 準備執行: {' '.join(command_to_run)}")
    print("-" * 50)

    try:
        process = subprocess.Popen(command_to_run, text=True, encoding='utf-8')
        process.wait()
    except KeyboardInterrupt:
        print("\n[啟動器] 偵測到使用者中斷，正在關閉...")
        if process:
            process.terminate()
    except Exception as e:
        print(f"\n[啟動器] 執行 main.py 時發生錯誤: {e}")

if __name__ == "__main__":
    main()