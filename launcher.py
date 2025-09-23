# launcher.py 的中文註釋(v3.2 - 完整性修復)
# 更新紀錄:
# v3.2 (2025-09-04): [灾难性BUG修复] 提供了完整的文件内容，确保顶层的 `import time` 语句被正确包含，解决了因 NameError 导致的启动器立即退出的问题。
# v3.1 (2025-09-04): [灾难性BUG修复] 在文件顶部增加了 `import time`，以解决因调用 `time.sleep()` 而导致的 `NameError: name 'time' is not defined` 致命错误。
# v3.0 (2025-09-03): [重大架構重構] 引入了守護進程循環。

import os
import sys
import subprocess
from pathlib import Path
import time

# 函式：執行命令
def _run_command(command, working_dir=None):
    """執行一個 shell 命令並返回成功與否。"""
    try:
        print(f"▶️ 正在執行: {' '.join(command)}")
        if working_dir and not Path(working_dir).is_dir():
            print(f"🔥 错误: 工作目录不存在: {working_dir}")
            return False
            
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            cwd=working_dir
        )
        return True
    except FileNotFoundError:
        print(f"🔥 错误: 'git' 命令未找到。")
        print("请确保您已在系统中安装 Git，并且其路径已添加到环境变量中。")
        if os.name == 'nt':
            os.system("pause")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"🔥 命令执行失败: {' '.join(command)}")
        print(f"   标准错误: {e.stderr.strip()}")
        if os.name == 'nt':
            os.system("pause")
        sys.exit(1)
    except Exception as e:
        print(f"🔥 执行命令时发生未知错误: {e}")
        if os.name == 'nt':
            os.system("pause")
        sys.exit(1)
# 函式：執行命令

# 函式：主啟動邏輯 (v4.1 - 健壯性鎖定修復)
# 更新紀錄:
# v4.1 (2025-09-23): [災難性BUG修復] 增加了自動化的 Git 鎖定檔案清理機制。在執行任何 Git 操作前，此版本會檢查並強制移除殘留的 .git/index.lock 檔案，從根本上解決了因進程衝突或崩潰導致的啟動失敗問題。
# v4.0 (2025-09-10): [架構重構] 徹底重構了熔斷機制。現在，在連續失敗後，守護進程將進入長時冷却模式並持續嘗試重啟，而不是直接退出，以確保遠程修復通道的絕對可用性。
# v3.2 (2025-09-04): [灾难性BUG修复] 提供了完整的文件内容。
def main():
    """主啟動函式，包含守護進程和熔斷機制。"""
    current_dir = Path(__file__).resolve().parent

    print("--- AI Lover 啟動器 ---")

    failure_count = 0
    last_failure_time = 0.0
    FAILURE_THRESHOLD = 5
    FAILURE_WINDOW = 60 
    COOLDOWN_SECONDS = 300 # 進入冷却模式後的等待時間 (5分鐘)

    while True:
        print("\n--- 步驟 1/3: 檢查 Git 環境 ---")
        if not _run_command(["git", "--version"], working_dir=current_dir):
            return

        # [v4.1 核心修正] Git 鎖定檔案自動清理機制
        print("\n--- 步驟 1.5/3: 檢查並清理 Git 鎖定 ---")
        lock_file = current_dir / ".git" / "index.lock"
        if lock_file.is_file():
            print("   ⚠️ 警告: 偵測到殘留的 Git 鎖定檔案 (.git/index.lock)。")
            print("   -> 可能是上一個 Git 進程崩潰導致的。")
            print("   -> 將在短暫等待後嘗試自動移除...")
            time.sleep(2) # 給予正常進程結束的機會
            try:
                if lock_file.is_file(): # 再次檢查
                    lock_file.unlink()
                    print("   ✅ 殘留的鎖定檔案已成功移除。")
            except OSError as e:
                print(f"   🔥 錯誤: 自動移除鎖定檔案失敗: {e}")
                print("   請手動刪除 'D:/DINO/SD/ComfyUI/personal_server/ai_lover_service/.git/index.lock' 檔案後再試。")
                if os.name == 'nt':
                    os.system("pause")
                return
        else:
            print("   ✅ Git 倉庫狀態正常，無殘留鎖定。")


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
        return_code = -1

        try:
            print(f"🚀 準備執行: {' '.join(command_to_run)}")
            print("-" * 50)
            process = subprocess.Popen(command_to_run, text=True, encoding='utf-8')
            return_code = process.wait()

        except KeyboardInterrupt:
            print("\n[啟動器] 偵測到使用者中斷，正在關閉...")
            if process:
                process.terminate()
            break
        except Exception as e:
            print(f"\n[啟動器] 執行 main.py 時發生嚴重錯誤: {e}")
            return_code = 1
        finally:
            current_time = time.time()
            if return_code == 0:
                print(f"\n[啟動器] 偵測到主程式正常退出 (返回碼 0)。")
                failure_count = 0 
            else:
                print(f"\n[啟動器] 偵測到主程式異常退出 (返回碼: {return_code})。")
                
                if current_time - last_failure_time < FAILURE_WINDOW:
                    failure_count += 1
                else:
                    failure_count = 1
                
                last_failure_time = current_time
                
                # [v4.0 核心修正] 重構熔斷機制為長時冷却
                if failure_count >= FAILURE_THRESHOLD:
                    print(f"🔥🔥🔥 [啟動器冷却模式] 在 {FAILURE_WINDOW} 秒內連續失敗 {failure_count} 次！")
                    print(f"[啟動器] 系統可能存在持續性BUG。為防止資源耗盡，將進入 {COOLDOWN_SECONDS} 秒的長時冷却。")
                    print(f"[啟動器] 在此期間，您可以推送修復到GitHub倉庫。冷却結束後，系統將自動拉取最新程式碼並嘗試重啟。")
                    time.sleep(COOLDOWN_SECONDS)
                    failure_count = 0 # 冷却結束後重置計數器，給予新程式碼一個完整的重試機會
                    continue # 跳過下方的短時等待，直接進入下一個循環
            
            print(f"[啟動器] 將在 5 秒後嘗試重啟...")
            time.sleep(5)

    if os.name == 'nt':
        print("\n----------------------------------------------------")
        print("[AI Lover Launcher] 程式已結束。您可以按任意鍵關閉此視窗。")
        os.system("pause")
# 函式：主啟動邏輯 (v4.1 - 健壯性鎖定修復)

if __name__ == "__main__":
    main()

