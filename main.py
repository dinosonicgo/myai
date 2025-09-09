# main.py 的中文註釋(v6.0 - 優雅重啟)
# 更新紀錄:
# v6.0 (2025-09-06): [災難性BUG修復] 徹底重構了程式的關閉與重啟機制。
#    1. [新增] 引入了全局的 `asyncio.Event` 作為優雅關閉信號。
#    2. [修正] `_perform_update_and_restart` 不再調用 `sys.exit(0)`，而是設置此事件。
#    3. [修正] `main` 函式現在會等待此事件，然後再正常退出。
#    此修改遵循了異步程式設計的最佳實踐，從根本上解決了因在背景任務中使用 `sys.exit` 而導致的 `Task exception was never retrieved` 警告。
# v5.2 (2025-09-02): [根本性BUG修復] 增加了自動清理 __pycache__ 的功能。
# v5.1 (2025-09-02): [健壯性] 修改了自動更新邏輯，改為使用與啟動器相同的 'git reset --hard'。

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

# [v6.0 新增] 創建一個全局的關閉事件
shutdown_event = asyncio.Event()

def _clear_pycache():
    """遞歸地查找並刪除當前目錄及其子目錄下的所有 __pycache__ 資料夾。"""
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
# [v6.0 新增] 導入 bot 實例以傳遞關閉事件
from src.discord_bot import AILoverBot

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
app.include_router(web_router)

def _check_and_install_dependencies():
    import importlib.util
    required_packages = {
        'uvicorn': 'uvicorn', 'fastapi': 'fastapi', 'SQLAlchemy': 'sqlalchemy',
        'aiosqlite': 'aiosqlite', 'discord.py': 'discord', 'langchain': 'langchain',
        'langchain-core': 'langchain_core', 'langchain-google-genai': 'langchain_google_genai',
        'langchain-community': 'langchain_community', 'langchain-chroma': 'langchain_chroma',
        'langchain-cohere': 'langchain_cohere', 'google-generativeai': 'google.generativeai',
        'chromadb': 'chromadb', 'rank_bm25': 'rank_bm25',
        'pydantic-settings': 'pydantic_settings', 'Jinja2': 'jinja2',
        'python-Levenshtein': 'Levenshtein'
    }
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            if importlib.util.find_spec(import_name) is None:
                importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            missing_packages.append(package_name)
    if not missing_packages:
        print("✅ 所有依賴項均已安裝。")
    if missing_packages:
        print("\n⏳ 正在自動安裝缺失的依賴項，請稍候...")
        for package in missing_packages:
            try:
                print(f"   -> 正在安裝 {package}...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--quiet", package]
                )
                print(f"   ✅ {package} 安裝成功。")
            except subprocess.CalledProcessError:
                print(f"   🔥 {package} 安裝失敗！請手動執行 'pip install {package}' 後再試。")
                if os.name == 'nt': os.system("pause")
                sys.exit(1)
        print("\n🔄 所有依賴項已安裝完畢。程式將在 3 秒後自動重啟以應用變更...")
        time.sleep(3)
        os.execv(sys.executable, [sys.executable] + sys.argv)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 函式：[守護任務] 自動推送LOG到GitHub倉庫 (v2.0 - 獨立化)
async def start_git_log_pusher_task():
    """一個完全獨立的背景任務，定期將最新的日誌檔案推送到GitHub倉庫。"""
    await asyncio.sleep(15) # 初始延遲，等待其他服務啟動
    print("✅ [守護任務] LOG 自動推送器已啟動。")
    
    project_root = Path(__file__).resolve().parent
    log_file_path = project_root / "data" / "logs" / "app.log"
    upload_log_path = project_root / "latest_log.txt"

    def run_git_commands():
        """同步執行Git指令的輔助函式。"""
        try:
            if not log_file_path.is_file():
                print(f"🟡 [LOG Pusher] 等待日誌檔案創建...")
                return True

            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            latest_lines = lines[-100:]
            log_content_to_write = "".join(latest_lines)

            with open(upload_log_path, 'w', encoding='utf-8') as f:
                f.write(f"### AI Lover Log - Last updated at {datetime.datetime.now().isoformat()} ###\n\n")
                f.write(log_content_to_write)

            subprocess.run(["git", "add", str(upload_log_path)], check=True, cwd=project_root, capture_output=True)
            
            commit_message = f"docs: Update latest_log.txt at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            commit_process = subprocess.run(
                ["git", "commit", "-m", commit_message], 
                capture_output=True, text=True, encoding='utf-8', cwd=project_root
            )
            
            if commit_process.returncode != 0 and "nothing to commit" not in commit_process.stdout:
                raise subprocess.CalledProcessError(
                    commit_process.returncode, commit_process.args, commit_process.stdout, commit_process.stderr
                )

            subprocess.run(["git", "push", "origin", "main"], check=True, cwd=project_root, capture_output=True)
            
            print(f"✅ [LOG Pusher] {datetime.datetime.now().strftime('%H:%M:%S')} - 最新LOG已成功推送到GitHub。")
            return True
        except subprocess.CalledProcessError as e:
            error_output = e.stderr or e.stdout
            if "nothing to commit" in error_output:
                print(f"⚪️ [LOG Pusher] {datetime.datetime.now().strftime('%H:%M:%S')} - LOG無變更，跳過推送。")
                return True
            print(f"🔥 [LOG Pusher] Git指令執行失敗: {error_output}")
            return False
        except Exception as e:
            print(f"🔥 [LOG Pusher] 執行時發生未知錯誤: {e}")
            return False

    while not shutdown_event.is_set():
        try:
            await asyncio.to_thread(run_git_commands)
            await asyncio.sleep(300) 
        except asyncio.CancelledError:
            print("⚪️ [LOG Pusher] 背景任務被正常取消。")
            break
        except Exception as e:
            print(f"🔥 [LOG Pusher] 背景任務主循環發生錯誤: {e}")
            await asyncio.sleep(60)
# 函式：[守護任務] 自動推送LOG到GitHub倉庫 (v2.0 - 獨立化)

async def main():
    MAIN_PY_VERSION = "v6.0"
    print(f"--- AI Lover 主程式 ({MAIN_PY_VERSION}) ---")
    
    _check_and_install_dependencies()

    # 函式：[核心服務] Discord Bot 啟動器 (v2.0 - 錯誤隔離)
    async def start_discord_bot_task():
        """啟動Discord Bot的核心服務。內建錯誤處理以防止其崩潰影響其他任務。"""
        try:
            if not settings.DISCORD_BOT_TOKEN:
                print("🔥 [Discord Bot] 錯誤：DISCORD_BOT_TOKEN 未在 config/.env 檔案中設定。服務無法啟動。")
                await asyncio.sleep(3600) # 等待一小時，避免在日誌中刷屏
                return

            print("🚀 [Discord Bot] 正在嘗試啟動核心服務...")
            bot = AILoverBot(shutdown_event=shutdown_event)
            async with bot:
                await bot.start(settings.DISCORD_BOT_TOKEN)
        except Exception as e:
            print(f"🔥 [Discord Bot] 核心服務啟動失敗或在運行時發生致命錯誤: {e}")
            # 打印更詳細的追蹤訊息，以便除錯
            import traceback
            traceback.print_exc()
            print("🔴 [Discord Bot] 核心服務已停止。守護任務將繼續運行。")
    # 函式：[核心服務] Discord Bot 啟動器 (v2.0 - 錯誤隔離)

    async def start_web_server_task():
        config = uvicorn.Config(app, host="localhost", port=8000, log_level="info")
        server = uvicorn.Server(config)
        web_task = asyncio.create_task(server.serve())
        await shutdown_event.wait()
        server.should_exit = True
        await web_task

    # 函式：[守護任務] GitHub 自動更新檢查器
    async def start_github_update_checker_task():
        """一個獨立的背景任務，檢查GitHub更新並在必要時觸發重啟。"""
        await asyncio.sleep(10)
        print("✅ [守護任務] GitHub 自動更新檢查器已啟動。")
        def run_git_command(command: list) -> tuple[int, str, str]:
            process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False)
            return process.returncode, process.stdout, process.stderr
        while not shutdown_event.is_set():
            try:
                await asyncio.to_thread(run_git_command, ['git', 'fetch'])
                rc, stdout, _ = await asyncio.to_thread(run_git_command, ['git', 'status', '-uno'])
                if rc == 0 and ("Your branch is behind" in stdout or "您的分支落後" in stdout):
                    print("\n🔄 [自動更新] 偵測到遠端倉庫有新版本，正在更新...")
                    pull_rc, _, pull_stderr = await asyncio.to_thread(run_git_command, ['git', 'reset', '--hard', 'origin/main'])
                    if pull_rc == 0:
                        print("✅ [自動更新] 程式碼強制同步成功！")
                        print("🔄 應用程式將在 3 秒後發出優雅關閉信號，由啟動器負責重啟...")
                        await asyncio.sleep(3)
                        shutdown_event.set()
                        break 
                    else:
                        print(f"🔥 [自動更新] 'git reset' 失敗: {pull_stderr}")
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                print("⚪️ [自動更新] 背景任務被正常取消。")
                break
            except FileNotFoundError:
                print("🔥 [自動更新] 錯誤: 'git' 命令未找到。自動更新功能已停用。")
                break
            except Exception as e:
                print(f"🔥 [自動更新] 檢查更新時發生未預期的錯誤: {type(e).__name__}: {e}")
                await asyncio.sleep(600)
    # 函式：[守護任務] GitHub 自動更新檢查器

    try:
        print("初始化資料庫...")
        await init_db()
        
        tasks = []
        mode = sys.argv[1] if len(sys.argv) > 1 else "all"
        
        # 核心服務
        if mode in ["all", "discord"]:
            tasks.append(start_discord_bot_task())
        if mode in ["all", "web"]:
            tasks.append(start_web_server_task())

        # 守護任務 (始終運行，除非被模式排除)
        if mode in ["all", "discord"]:
            tasks.append(start_github_update_checker_task())
            tasks.append(start_git_log_pusher_task())

        if not tasks:
            print(f"錯誤：未知的運行模式 '{mode}'。請使用 'all', 'discord', 或 'web'。")
            return

        print(f"\n啟動 AI戀人系統 (模式: {mode})...")
        
        # [v6.0 核心修正] 使用 asyncio.wait 實現優雅關閉
        # 1. 創建一個專門等待關閉信號的任務
        shutdown_waiter = asyncio.create_task(shutdown_event.wait())
        
        # 2. 將所有要運行的任務（包括 shutdown_waiter）轉換為 Task 物件
        all_tasks = {asyncio.create_task(t) for t in tasks}
        all_tasks.add(shutdown_waiter)

        # 3. 等待任何一個任務完成
        done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)

        # 4. 如果是 shutdown_waiter 完成了，說明收到了關閉信號
        if shutdown_waiter in done:
            print("收到關閉信號，正在優雅地終止所有背景任務...")
        else:
            # 如果是其他任務意外結束，也觸發關閉流程
            print("一個核心任務意外終止，正在關閉其他任務...")
            shutdown_event.set() # 確保其他任務也能收到信號

        # 5. 取消所有仍在運行的任務
        for task in pending:
            task.cancel()
        
        # 6. 等待所有任務的取消操作完成
        await asyncio.gather(*pending, return_exceptions=True)
        print("所有任務已清理完畢。")

    except Exception as e:
        print(f"\n主程式運行時發生未處理的錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        await asyncio.sleep(5)
    finally:
        print("主程式 main() 函式已結束。")


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
            print("這通常是因為循環導入 (Circular Import) 導致的。")
        else:
            print(f"\n程式啟動失敗，發生致命錯誤: {e}")
        if os.name == 'nt': os.system("pause")
