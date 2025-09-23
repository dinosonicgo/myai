# main.py 的中文註釋(v7.0 - 錯誤隔離)
# 更新紀錄:
# v7.0 (2025-10-04): [重大架構重構] 為了實現“守護任務的絕對獨立性”，徹底重構了主任務的啟動和管理邏輯。現在，核心服務（如 Discord Bot）和守護任務（如自動更新）被分離到不同的邏輯組中。核心服務的啟動器（start_discord_bot_task）被一個巨大的 try...except 塊包裹，以確保其自身的任何崩潰都不會影響到主事件循環。主函式現在使用 asyncio.gather 來並行運行所有任務，即使核心服務失敗退出，守護任務也會繼續在後台運行，從而確保了遠程修復通道的絕對可用性。
# v6.0 (2025-09-06): [災難性BUG修復] 引入了全局的 asyncio.Event 作為優雅關閉信號。
# v5.2 (2025-09-02): [根本性BUG修復] 增加了自動清理 __pycache__ 的功能。
# v8.0 (2025-10-15): [健壯性] 引入了全局的 asyncio.Lock 來保護 Git 操作，徹底解決了日誌推送和自動更新之間的競態條件問題。

import os
import sys
import shutil
from pathlib import Path
import asyncio
import uvicorn
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import subprocess
import importlib.metadata
import datetime
import traceback

# 全局的關閉事件，用於協調所有任務的優雅退出
shutdown_event = asyncio.Event()

# [v8.0 核心修正] 創建一個全局的異步鎖，用於保護 Git 操作
git_lock = asyncio.Lock()


# [核心修正] 将 PROJ_DIR 定义提升到全局作用域
PROJ_DIR = Path(__file__).resolve().parent

def _clear_pycache():
    """遞歸地查找並刪除 __pycache__ 資料夾。"""
    root_dir = Path(__file__).resolve().parent
    for path in root_dir.rglob('__pycache__'):
        if path.is_dir():
            print(f"🧹 清理舊快取: {path}")
            try:
                shutil.rmtree(path)
            except OSError as e:
                print(f"🔥 清理快取失敗: {e}")
_clear_pycache()

from src.database import init_db
from src.config import settings
from src.web_server import router as web_router
from src.discord_bot import AILoverBot

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
app.include_router(web_router)

# 函式：檢查並安裝缺失的 Python 依賴項
# 更新紀錄:
# v1.1 (2025-09-23): [災難性BUG修復] 根據 ModuleNotFoundError，補全了在 v1.0 中遺漏的、對 `spacy` 函式庫及其 `zh_core_web_sm` 中文模型的自動檢查與安裝功能。
# v1.0 (2025-09-23): [功能擴展] 引入“混合NLP”策略，但遺漏了實際的安裝邏輯。
def _check_and_install_dependencies():
    """檢查並安裝缺失的 Python 依賴項，包括 spaCy 和其模型。"""
    import importlib.util
    import subprocess
    import importlib.metadata
    
    required_packages = {
        'uvicorn': 'uvicorn', 'fastapi': 'fastapi', 'SQLAlchemy': 'sqlalchemy',
        'aiosqlite': 'aiosqlite', 'discord.py': 'discord', 'langchain': 'langchain',
        'langchain-core': 'langchain_core', 'langchain-google-genai': 'langchain_google_genai',
        'langchain-community': 'langchain_community', 'langchain-chroma': 'langchain_chroma',
        'langchain-cohere': 'langchain_cohere', 'google-generativeai': 'google.generativeai',
        'chromadb': 'chromadb', 'rank_bm25': 'rank_bm25',
        'pydantic-settings': 'pydantic_settings', 'Jinja2': 'jinja2',
        'python-Levenshtein': 'Levenshtein',
        'spacy': 'spacy' # [v1.1 新增]
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            if importlib.util.find_spec(import_name) is None:
                # 再次檢查，因為 find_spec 對某些包可能不準確
                importlib.metadata.version(package_name)
        except (ImportError, importlib.metadata.PackageNotFoundError):
            missing_packages.append(package_name)

    if missing_packages:
        print("\n⏳ 正在自動安裝缺失的 Python 依賴項...")
        for package in missing_packages:
            try:
                print(f"   - 正在安裝 {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
                print(f"   ✅ {package} 安裝成功。")
            except subprocess.CalledProcessError:
                print(f"   🔥 {package} 安裝失敗！請手動執行 'pip install {package}'。")
                if os.name == 'nt': os.system("pause")
                sys.exit(1)
        print("\n🔄 依賴項安裝完畢。")

    # [v1.1 新增] 檢查 spaCy 中文模型
    try:
        import spacy
        spacy.load('zh_core_web_sm')
        print("✅ spaCy 中文模型已安裝。")
    except (ImportError, OSError):
        # 如果 spacy 剛裝好，可能需要重新加載，或者模型不存在
        print("\n⏳ spaCy 中文模型未找到或未加載，正在自動下載...")
        try:
            # 確保 spacy 本身已安裝
            if 'spacy' in missing_packages:
                 print("   - spacy 剛安裝，請重啟啟動器以加載模型。")
                 if os.name == 'nt': os.system("pause")
                 sys.exit(0) # 正常退出，讓啟動器重啟
            
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "zh_core_web_sm"])
            print("✅ spaCy 中文模型下載成功。")
            print("\n🔄 正在重啟以應用變更...")
            time.sleep(3)
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except subprocess.CalledProcessError:
            print("   🔥 spaCy 中文模型下載失敗！請手動執行 'python -m spacy download zh_core_web_sm'。")
            if os.name == 'nt': os.system("pause")
            sys.exit(1)
            
    if not missing_packages:
        print("✅ 所有依賴項和模型均已準備就緒。")
# 函式：檢查並安裝缺失的 Python 依賴項


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})





# 函式：[守護任務] 自動推送LOG到GitHub倉庫 (v7.1 - 调整日志行数并实现静默成功)
# 更新紀錄:
# v7.1 (2025-09-23): [功能优化] 1. 将截取的日志行数从 100 行增加到 300 行，以提供更丰富的远程调试上下文。 2. 移除了成功推送到仓库时的 print() 确认消息，实现了“静默成功”，以避免在控制台中刷掉重要的错误日志。
# v7.0 (2025-10-20): [健壯性] 重构了日志推送逻辑，实现了“静默成功”的初步版本。
# v6.0 (2025-10-15): [健壯性] 增加了“静默模式”。
async def start_git_log_pusher_task(lock: asyncio.Lock):
    """一個完全獨立的背景任務，定期將最新的日誌檔案推送到GitHub倉庫。"""
    await asyncio.sleep(15)
    print("✅ [守護任務] LOG 自動推送器已啟動。")
    
    log_file_path = PROJ_DIR / "data" / "logs" / "app.log"
    upload_log_path = PROJ_DIR / "latest_log.txt"

    def run_git_commands_sync() -> bool:
        """
        同步執行Git指令的輔助函式，設計為在背景線程中運行。
        返回 True 表示有新的 commit 被推送，返回 False 表示沒有變化。
        """
        try:
            if not log_file_path.is_file(): return False
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # [v7.1 核心修正] 将截取的行数从 -100 修改为 -300
            latest_lines = lines[-300:]
            log_content_to_write = "".join(latest_lines)
            with open(upload_log_path, 'w', encoding='utf-8') as f:
                f.write(f"### AI Lover Log - Last updated at {datetime.datetime.now().isoformat()} ###\n\n")
                f.write(log_content_to_write)
            subprocess.run(["git", "add", str(upload_log_path)], check=True, cwd=PROJ_DIR, capture_output=True)
            commit_message = f"docs: Update latest_log.txt at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            commit_process = subprocess.run(
                ["git", "commit", "-m", commit_message], 
                capture_output=True, text=True, encoding='utf-8', cwd=PROJ_DIR
            )
            if commit_process.returncode != 0:
                if "nothing to commit" in commit_process.stdout or "沒有東西可以提交" in commit_process.stdout:
                    return False
                else:
                    raise subprocess.CalledProcessError(
                        commit_process.returncode, commit_process.args, commit_process.stdout, commit_process.stderr
                    )
            
            subprocess.run(["git", "push", "origin", "main"], check=True, cwd=PROJ_DIR, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            error_output = e.stderr or e.stdout
            if "nothing to commit" not in str(error_output) and "沒有東西可以提交" not in str(error_output):
                print(f"🔥 [LOG Pusher] Git指令執行失敗: {error_output}")
            return False
        except Exception as e:
            print(f"🔥 [LOG Pusher] 執行時發生未知錯誤: {e}")
            return False

    while not shutdown_event.is_set():
        try:
            async with lock:
                # [v7.1 核心修正] 移除了成功时的 print() 消息
                await asyncio.to_thread(run_git_commands_sync)
            
            await asyncio.sleep(300) 
        except asyncio.CancelledError:
            print("⚪️ [LOG Pusher] 背景任務被正常取消。")
            break
        except Exception as e:
            print(f"🔥 [LOG Pusher] 背景任務主循環發生錯誤: {e}")
            await asyncio.sleep(60)
# 函式：[守護任務] 自動推送LOG到GitHub倉庫 (v7.1 - 调整日志行数并实现静默成功)






    
# 函式：[守護任務] GitHub 自動更新檢查器 (v2.2 - 縮排修正)
# 更新纪录:
# v2.2 (2025-10-10): [災難性BUG修復] 修正了此函式定義的全局作用域缩排錯誤，解決了導致 NameError 的問題。
# v2.1 (2025-10-09): [災難性BUG修復] 修正了此函式因无法访问 PROJ_DIR 變數而導致的 NameError。
# v3.0 (2025-10-15): [健壯性] 整合了 asyncio.Lock，以防止與日誌推送任務發生 Git 競態條件。
# v4.0 (2025-10-15): [健壯性] 增加了「靜默模式」，只有在檢測到新版本時，才會打印詳細的 Git 操作日誌。
async def start_github_update_checker_task(lock: asyncio.Lock):
    """一個獨立的背景任務，檢查GitHub更新並在必要時觸發重啟。"""
    await asyncio.sleep(10)
    print("✅ [守護任務] GitHub 自動更新檢查器已啟動。")
    
    def run_git_command_sync(command: list) -> tuple[int, str, str]:
        """在背景線程中安全地執行同步的 git 命令。"""
        process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False, cwd=PROJ_DIR)
        return process.returncode, process.stdout, process.stderr
        
    while not shutdown_event.is_set():
        try:
            async with lock:
                # [v4.0 核心修正] 先檢查，再根據結果決定是否打印日誌
                await asyncio.to_thread(run_git_command_sync, ['git', 'fetch'])
                rc, stdout, _ = await asyncio.to_thread(run_git_command_sync, ['git', 'status', '-uno'])
                
                if rc == 0 and ("Your branch is behind" in stdout or "您的分支落後" in stdout):
                    print("🔵 [Auto Update] 已獲取 Git 鎖，檢測到新版本，準備更新...")
                    print("\n🔄 [自動更新] 偵測到遠端倉庫有新版本，正在更新...")
                    pull_rc, _, pull_stderr = await asyncio.to_thread(run_git_command_sync, ['git', 'reset', '--hard', 'origin/main'])
                    if pull_rc == 0:
                        print("✅ [自動更新] 程式碼強制同步成功！")
                        print("🔄 應用程式將在 3 秒後發出優雅關閉信號，由啟動器負責重啟...")
                        await asyncio.sleep(3)
                        shutdown_event.set()
                        print("🟢 [Auto Update] 更新完成，已釋放 Git 鎖。")
                        break 
                    else:
                        print(f"🔥 [自動更新] 'git reset' 失敗: {pull_stderr}")
                # else: # 在靜默模式下，如果沒有更新，則不打印任何日誌
                #     print("🟢 [Auto Update] 未檢測到更新，已釋放 Git 鎖。")
            
            await asyncio.sleep(300)

        except asyncio.CancelledError:
            print("⚪️ [自動更新] 背景任務被正常取消。")
            break
        except Exception as e:
            print(f"🔥 [自動更新] 檢查更新時發生未預期的錯誤: {type(e).__name__}: {e}")
            await asyncio.sleep(600)
# 函式：[守護任務] GitHub 自動更新檢查器 (v2.2 - 縮排修正)




# main.py 的 start_discord_bot_task 函式
# 更新紀錄:
# v4.0 (2025-11-14): [災難性BUG修復] 此函式現在接收 git_lock 並將其傳遞給 AILoverBot 的構造函數。
async def start_discord_bot_task(lock: asyncio.Lock):
    """啟動Discord Bot的核心服務。內建錯誤處理以防止其崩潰影響其他任務。"""
    try:
        if not settings.DISCORD_BOT_TOKEN:
            print("🔥 [Discord Bot] 錯誤：DISCORD_BOT_TOKEN 未在 config/.env 檔案中設定。服務無法啟動。")
            return

        print("🚀 [Discord Bot] 正在嘗試啟動核心服務...")
        # [v4.0 核心修正] 將鎖傳入
        bot = AILoverBot(shutdown_event=shutdown_event, git_lock=lock)
        
        bot_task = asyncio.create_task(bot.start(settings.DISCORD_BOT_TOKEN))
        shutdown_waiter = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            {bot_task, shutdown_waiter}, 
            return_when=asyncio.FIRST_COMPLETED
        )

        if shutdown_waiter in done:
            print("🔵 [Discord Bot] 收到外部關閉信號，正在優雅關閉...")
            await bot.close()
        
        for task in pending:
            task.cancel()

    except Exception as e:
        print(f"🔥 [Discord Bot] 核心服務在啟動或運行時發生致命錯誤: {e}")
        traceback.print_exc()
    finally:
        print("🔴 [Discord Bot] 核心服務任務已結束。守護任務將繼續獨立運行。")
# start_discord_bot_task 函式結束

# 函式：[核心服務] Web 伺服器啟動器 (v2.0 - 錯誤隔離)
async def start_web_server_task():
    """啟動 FastAPI Web 伺服器並監聽關閉信號，內建錯誤隔離。"""
    try:
        config = uvicorn.Config(app, host="localhost", port=8000, log_level="info")
        server = uvicorn.Server(config)
        
        web_task = asyncio.create_task(server.serve())
        shutdown_waiter = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            {web_task, shutdown_waiter},
            return_when=asyncio.FIRST_COMPLETED
        )

        if shutdown_waiter in done:
            print("🔵 [Web Server] 收到外部關閉信號，正在優雅關閉...")
            server.should_exit = True
        
        for task in pending:
            task.cancel()

    except Exception as e:
        print(f"🔥 [Web Server] 核心服務在啟動或運行時發生致命錯誤: {e}")
        traceback.print_exc()
    finally:
        print("🔴 [Web Server] 核心服務任務已結束。守護任務將繼續獨立運行。")
# 函式：[核心服務] Web 伺服器啟動器 (v2.0 - 錯誤隔離)


# main.py 的 main 函式
# 更新紀錄:
# v9.0 (2025-11-14): [災難性BUG修復] 為了從根本上解決死鎖問題，現在將全局的 git_lock 實例在創建 AILoverBot 時直接傳遞給它，確保 Bot 內的所有 Git 操作都能與背景守護任務共享同一個鎖。
# v8.0 (2025-10-15): [健壯性] 引入了全局的 asyncio.Lock 來保護 Git 操作。
async def main():
    MAIN_PY_VERSION = "v9.0"
    print(f"--- AI Lover 主程式 ({MAIN_PY_VERSION}) ---")
    
    _check_and_install_dependencies()

    try:
        print("初始化資料庫...")
        await init_db()
        
        core_services = []
        guardian_tasks = []
        mode = sys.argv[1] if len(sys.argv) > 1 else "all"
        
        # [v9.0 核心修正] 將 git_lock 傳遞給 AILoverBot
        if mode in ["all", "discord"]:
            core_services.append(start_discord_bot_task(git_lock))
        if mode in ["all", "web"]:
            core_services.append(start_web_server_task())

        guardian_tasks.append(start_github_update_checker_task(git_lock))
        guardian_tasks.append(start_git_log_pusher_task(git_lock))

        if not core_services and not guardian_tasks:
            print(f"錯誤：未知的運行模式 '{mode}'。請使用 'all', 'discord', 或 'web'。")
            return

        print(f"\n啟動 AI戀人系統 (模式: {mode})...")
        
        all_tasks = core_services + guardian_tasks
        await asyncio.gather(*all_tasks)

        if shutdown_event.is_set():
            print("🔄 [Main Process] 收到重啟信號，主程式即將退出以觸發 Launcher 重啟。")
            sys.exit(0) 

    except asyncio.CancelledError:
        print("主任務被取消，程式正在關閉。")
    except Exception as e:
        print(f"\n主程式運行時發生未處理的頂層錯誤: {str(e)}")
        traceback.print_exc()
    finally:
        print("主程式 main() 函式已結束。 launcher.py 將在 5 秒後嘗試重啟。")
# main 函式結束

if __name__ == "__main__":
    try:
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n偵測到使用者中斷 (Ctrl+C)，程式已停止運行。")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n【啟動失敗】致命設定錯誤: {e}")
        if os.name == 'nt': os.system("pause")
    except Exception as e:
        if isinstance(e, ImportError):
            print(f"\n【啟動失敗】致命導入錯誤: {e}")
        else:
            print(f"\n程式啟動失敗，發生致命錯誤: {e}")
        if os.name == 'nt': os.system("pause")
