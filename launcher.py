# launcher.py 的中文註釋(v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-09-01):
# 1. [全新創建] 創建此啟動器腳本，作為整個應用的主入口點。
# 2. [功能] 實現了在啟動時自動從 GitHub clone 或 pull 最新版本的程式碼。
# 3. [健壯性] 使用 subprocess 執行 git 命令，並在執行主程式前切換工作目錄，確保路徑引用的正確性。

import os
import sys
import subprocess
from pathlib import Path

# --- 設定 ---
# 您的 GitHub 倉庫 URL
REPO_URL = "https://github.com/dinosonicgo/myai.git"
# 專案程式碼將被存放的本地目錄名稱
APP_DIR_NAME = "myai_app"
# --- 設定結束 ---

# 函式：執行命令
# 說明：一個輔助函式，用於執行系統命令並處理潛在的錯誤。
def _run_command(command, working_dir=None):
    """執行一個 shell 命令並返回成功與否。"""
    try:
        print(f"▶️ 正在執行: {' '.join(command)} (於: {working_dir or '.'})")
        # 使用 check=True，如果命令返回非零退出碼（表示錯誤），將會引發 CalledProcessError
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
        print(f"   錯誤碼: {e.returncode}")
        print(f"   標準輸出: {e.stdout.strip()}")
        print(f"   標準錯誤: {e.stderr.strip()}")
        if "Authentication failed" in e.stderr:
            print("\n   [可能原因] 您的 GitHub 倉庫是私有的，需要進行身份驗證。")
        elif "not a git repository" in e.stderr:
             print("\n   [可能原因] 本地目錄已存在但不是一個有效的 Git 倉庫。請嘗試刪除該目錄後重試。")
        if os.name == 'nt':
            os.system("pause")
        sys.exit(1)
# 函式：執行命令

# 函式：主啟動邏輯
# 說明：執行整個更新與啟動流程。
def main():
    """主啟動函式。"""
    # 獲取當前腳本所在的目錄
    current_dir = Path(__file__).resolve().parent
    app_path = current_dir / APP_DIR_NAME

    # 步驟 1: 檢查 Git 是否安裝
    print("--- 步驟 1/3: 檢查 Git 環境 ---")
    if not _run_command(["git", "--version"]):
        return # _run_command 內部已處理錯誤退出

    # 步驟 2: Clone 或 Pull 倉庫
    print("\n--- 步驟 2/3: 同步最新的應用程式碼 ---")
    if app_path.is_dir():
        print(f"📁 應用程式目錄 '{APP_DIR_NAME}' 已存在，嘗試更新...")
        if not _run_command(["git", "pull"], working_dir=app_path):
            return
        print("✅ 更新成功。")
    else:
        print(f"📁 應用程式目錄 '{APP_DIR_NAME}' 不存在，正在從 GitHub 下載...")
        if not _run_command(["git", "clone", REPO_URL, str(app_path)]):
            return
        print("✅ 下載成功。")

    # 步驟 3: 執行主應用程式
    print(f"\n--- 步驟 3/3: 啟動主應用程式 ---")
    main_py_path = app_path / "main.py"
    if not main_py_path.is_file():
        print(f"🔥 致命錯誤: 在下載的目錄中找不到 'main.py'。")
        print(f"   請檢查您的 GitHub 倉庫 '{REPO_URL}' 中是否包含 main.py 檔案。")
        if os.name == 'nt':
            os.system("pause")
        sys.exit(1)

    # 將工作目錄切換到應用程式目錄下，這對確保 main.py 能正確找到相對路徑的檔案至關重要
    os.chdir(app_path)
    
    # 獲取傳遞給 launcher.py 的所有參數 (例如 'web' 或 'discord')
    args_to_pass = sys.argv[1:]
    command_to_run = [sys.executable, "main.py"] + args_to_pass

    print(f"🚀 準備執行: {' '.join(command_to_run)}")
    print("-" * 50)

    try:
        # 使用 Popen 來執行，這樣可以即時看到 main.py 的輸出
        process = subprocess.Popen(command_to_run, text=True, encoding='utf-8')
        process.wait() # 等待 main.py 執行完畢
    except KeyboardInterrupt:
        print("\n[啟動器] 偵測到使用者中斷，正在關閉...")
        if process:
            process.terminate()
    except Exception as e:
        print(f"\n[啟動器] 執行 main.py 時發生錯誤: {e}")

if __name__ == "__main__":
    main()