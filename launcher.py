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

# 函式：主啟動邏輯 (v3.2 - 健壯性重啟)
def main():
    """主啟動函式，包含守護進程和熔斷機制。"""
    current_dir = Path(__file__).resolve().parent

    print("--- AI Lover 啟動器 ---")

    # [核心修改] 引入熔斷機制變數
    failure_count = 0
    last_failure_time = 0.0
    FAILURE_THRESHOLD = 5  # 連續失敗5次則熔斷
    FAILURE_WINDOW = 60    # 60秒內的連續失敗才計數

    while True:
        print("\n--- 步驟 1/3: 檢查 Git 環境 ---")
        if not _run_command(["git", "--version"], working_dir=current_dir):
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
            return_code = 1 # 將未知錯誤也視為失敗
        finally:
            current_time = time.time()
            if return_code == 0:
                print(f"\n[啟動器] 偵測到主程式正常退出 (返回碼 0)。")
                failure_count = 0 # 成功運行後重置失敗計數器
            else:
                print(f"\n[啟動器] 偵測到主程式異常退出 (返回碼: {return_code})。")
                # [核心修改] 熔斷邏輯
                if current_time - last_failure_time < FAILURE_WINDOW:
                    failure_count += 1
                else:
                    failure_count = 1 # 超過時間窗口，重置計數
                
                last_failure_time = current_time
                
                if failure_count >= FAILURE_THRESHOLD:
                    print(f"🔥🔥🔥 [啟動器熔斷] 在 {FAILURE_WINDOW} 秒內連續失敗 {failure_count} 次！")
                    print("[啟動器] 為防止資源耗盡，守護進程已停止。請檢查LOG以修復持續性BUG。")
                    break # 觸發熔斷，跳出 while 循環
            
            print(f"[啟動器] 將在 5 秒後嘗試重啟...")
            time.sleep(5)

    if os.name == 'nt':
        print("\n----------------------------------------------------")
        print("[AI Lover Launcher] 程式已結束。您可以按任意鍵關閉此視窗。")
        os.system("pause")
# 函式：主啟動邏輯 (v3.2 - 健壯性重啟)

if __name__ == "__main__":
    main()
