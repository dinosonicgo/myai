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

# launcher.py 的 main 函式 (v5.0 - 啟動器前置依賴檢查)
# 更新紀錄:
# v5.0 (2025-11-26): [灾难性BUG修复] 引入了「啟動器前置依賴檢查」機制。在同步程式碼之後、啟動主應用之前，啟動器現在會強制執行 `pip install -r requirements.txt`。此修改確保了即使 `main.py` 中引入了新的函式庫，也能在啟動時被自動安裝，從根本上解決了因缺少依賴而導致的 `ImportError` 啟動失敗問題。
# v3.2 (2025-09-04): [灾难性BUG修复] 提供了完整的文件内容，确保顶层的 `import time` 语句被正确包含。
# v3.1 (2025-09-04): [灾难性BUG修复] 在文件顶部增加了 `import time`。
def main():
    """主啟動函式，包含守護進程、前置依賴檢查和熔斷機制。"""
    current_dir = Path(__file__).resolve().parent
    requirements_path = current_dir / "requirements.txt"

    print("--- AI Lover 啟動器 ---")

    # ... (failure_count 和其他變數定義保持不變) ...
    failure_count = 0
    last_failure_time = 0.0
    FAILURE_THRESHOLD = 5
    FAILURE_WINDOW = 60 
    COOLDOWN_SECONDS = 300

    while True:
        print("\n--- 步驟 1/4: 檢查 Git 環境與鎖定 ---")
        # ... (Git 檢查邏輯保持不變) ...
        if not _run_command(["git", "--version"], working_dir=current_dir):
            return
        
        lock_file = current_dir / ".git" / "index.lock"
        if lock_file.is_file():
            print("   ⚠️ 警告: 偵測到殘留的 Git 鎖定檔案，將嘗試自動移除...")
            try:
                lock_file.unlink()
                print("   ✅ 殘留的鎖定檔案已成功移除。")
            except OSError as e:
                print(f"   🔥 錯誤: 自動移除鎖定檔案失敗: {e}")
                if os.name == 'nt': os.system("pause")
                return
        else:
            print("   ✅ Git 倉庫狀態正常。")


        print("\n--- 步驟 2/4: 正在從 GitHub 同步最新程式碼 ---")
        # ... (Git 同步邏輯保持不變) ...
        if not _run_command(["git", "fetch"], working_dir=current_dir):
            print("   ⚠️ 警告: 'git fetch' 失敗，將嘗試繼續...")
        if not _run_command(["git", "reset", "--hard", "origin/main"], working_dir=current_dir):
            print("   🔥 錯誤: 強制同步失敗。")
            if os.name == 'nt': os.system("pause")
            return
        print("✅ 程式碼已同步至最新版本。")

        # [v5.0 核心修正] 啟動器前置依賴檢查
        print("\n--- 步驟 3/4: 正在根據 requirements.txt 同步 Python 環境 ---")
        if not requirements_path.is_file():
            print(f"   🔥 錯誤: 找不到 'requirements.txt' 檔案。")
            if os.name == 'nt': os.system("pause")
            return
        
        if not _run_command([sys.executable, "-m", "pip", "install", "--upgrade", "-r", str(requirements_path)], working_dir=current_dir):
            print(f"   🔥 錯誤: 依賴項同步失敗。請檢查 pip 的錯誤訊息。")
            if os.name == 'nt': os.system("pause")
            return
        print("✅ Python 環境已成功同步。")

        print(f"\n--- 步驟 4/4: 啟動主應用程式 ---")
        # ... (後續的主應用啟動和守護邏輯保持不變) ...
        main_py_path = current_dir / "main.py"
        if not main_py_path.is_file():
            print(f"🔥 致命錯誤: 找不到 'main.py'。")
            if os.name == 'nt': os.system("pause")
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
                if failure_count >= FAILURE_THRESHOLD:
                    print(f"🔥🔥🔥 [啟動器冷却模式] 在 {FAILURE_WINDOW} 秒內連續失敗 {failure_count} 次！")
                    print(f"   將進入 {COOLDOWN_SECONDS} 秒的長時冷却...")
                    time.sleep(COOLDOWN_SECONDS)
                    failure_count = 0
                    continue
            
            print(f"[啟動器] 將在 5 秒後嘗試重啟...")
            time.sleep(5)

    if os.name == 'nt':
        print("\n----------------------------------------------------")
        print("[AI Lover Launcher] 程式已結束。")
        os.system("pause")
# 函式：主啟動邏輯

if __name__ == "__main__":
    main()




