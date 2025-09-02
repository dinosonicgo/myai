# launcher.py 的中文註釋(v2.1 - 強制同步)
# 更新紀錄:
# v2.1 (2025-09-02):
# 1. [健壯性] 在 'git pull' 之前增加了 'git reset --hard origin/main'。此命令會強制將本地倉庫與遠端同步，拋棄任何本地意外的修改，從根本上解決了因本地狀態不一致導致更新失敗的問題。
# v2.0 (2025-09-01):
# 1. [重大架構重構] 移除了 clone 和切換目錄的邏輯。

import os
import sys
import subprocess
from pathlib import Path

# 函式：執行命令
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
        if os.name == 'nt':
            os.system("pause")
        sys.exit(1)
# 函式：執行命令

# 函式：主啟動邏輯 (v3.0 - 守護進程重構)
# 更新紀錄:
# v3.0 (2025-09-03): [重大架構重構] 引入了守護進程循環。現在 launcher.py 會持續監控 main.py。當 main.py 以返回碼 0（表示需要更新）退出時，launcher 會自動重新拉起一個全新的進程。此修改確保了每次更新後的重啟都是完全乾淨的，從根本上解決了因 os.execv 硬重啟導致的異步 I/O 衝突和 Bot 無法登錄的問題。
# v2.1 (2025-09-02): [健壯性] 增加了 'git reset --hard origin/main' 來強制同步。
def main():
    """主啟動函式。"""
    current_dir = Path(__file__).resolve().parent

    print("--- AI Lover 啟動器 ---")

    # [v3.0 核心修正] 引入守護循環
    while True:
        print("\n--- 步驟 1/3: 檢查 Git 環境 ---")
        if not _run_command(["git", "--version"]):
            return

        print("\n--- 步驟 2/3: 正在從 GitHub 同步最新程式碼 ---")
        if not _run_command(["git", "fetch"], working_dir=current_dir):
            print("   ⚠️ 警告: 'git fetch' 失敗，將嘗試繼續...")
        print("   -> 正在強制同步本地倉庫至遠端最新版本...")
        if not _run_command(["git", "reset", "--hard", "origin/main"], working_dir=current_dir):
            print("   🔥 錯誤: 強制同步失敗。請手動檢查您的 Git 倉庫狀態。")
            if os.name == 'nt':
                os.system("pause")
            return
        print("✅ 程式碼已強制同步至最新版本。")

        print(f"\n--- 步驟 3/3: 啟動主應用程式 ---")
        main_py_path = current_dir / "main.py"
        if not main_py_path.is_file():
            print(f"🔥 致命錯誤: 在當前目錄中找不到 'main.py'。")
            if os.name == 'nt':
                os.system("pause")
            sys.exit(1)

        args_to_pass = sys.argv[1:]
        command_to_run = [sys.executable, "main.py"] + args_to_pass
        process = None

        try:
            print(f"🚀 準備執行: {' '.join(command_to_run)}")
            print("-" * 50)
            process = subprocess.Popen(command_to_run, text=True, encoding='utf-8')
            return_code = process.wait()

            # [v3.0 核心修正] 檢查返回碼
            if return_code == 0:
                print("\n[啟動器] 偵測到主程式正常退出 (返回碼 0)，將在 5 秒後自動重啟以應用更新...")
                time.sleep(5)
                # 循環將自動繼續
            else:
                print(f"\n[啟動器] 偵測到主程式異常退出 (返回碼: {return_code})。")
                print("[啟動器] 守護進程已停止。")
                break # 發生錯誤，跳出循環

        except KeyboardInterrupt:
            print("\n[啟動器] 偵測到使用者中斷，正在關閉...")
            if process:
                process.terminate()
            break # 使用者手動中斷，跳出循環
        except Exception as e:
            print(f"\n[啟動器] 執行 main.py 時發生錯誤: {e}")
            break # 未知錯誤，跳出循環

    if os.name == 'nt':
        print("\n----------------------------------------------------")
        print("[AI Lover Launcher] 程式已結束。您可以按任意鍵關閉此視窗。")
        os.system("pause")

if __name__ == "__main__":
    main()
# 函式：主啟動邏輯 (v3.0 - 守護進程重構)
