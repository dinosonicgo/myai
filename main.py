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

async def main():
    MAIN_PY_VERSION = "v6.0"
    print(f"--- AI Lover 主程式 ({MAIN_PY_VERSION}) ---")
    
    _check_and_install_dependencies()

    async def start_discord_bot_task():
        if not settings.DISCORD_BOT_TOKEN:
            print("錯誤：DISCORD_BOT_TOKEN 未在 config/.env 檔案中設定。")
            await asyncio.sleep(10)
            return
        try:
            # [v6.0 修正] 傳入關閉事件
            bot = AILoverBot(shutdown_event=shutdown_event)
            async with bot:
                await bot.start(settings.DISCORD_BOT_TOKEN)
        except Exception as e:
            print(f"啟動 Discord Bot 時發生錯誤: {e}")

    async def start_web_server_task():
        config = uvicorn.Config(app, host="localhost", port=8000, log_level="info")
        server = uvicorn.Server(config)
        # [v6.0 新增] 讓 web server 也能響應關閉事件
        web_task = asyncio.create_task(server.serve())
        await shutdown_event.wait()
        server.should_exit = True
        await web_task

    async def start_github_update_checker_task():
        await asyncio.sleep(10)
        print("✅ 背景任務：GitHub 自動更新檢查器已啟動。")
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
                        # [v6.0 核心修正] 設置事件，而不是退出
                        shutdown_event.set()
                        break 
                    else:
                        print(f"🔥 [自動更新] 'git reset' 失敗: {pull_stderr}")
                # [v6.0 修正] 使用 asyncio.sleep 進行非阻塞等待
                await asyncio.sleep(300)
            except FileNotFoundError:
                print("🔥 [自動更新] 錯誤: 'git' 命令未找到。自動更新功能已停用。")
                break
            except Exception as e:
                print(f"🔥 [自動更新] 檢查更新時發生未預期的錯誤: {type(e).__name__}: {e}")
                await asyncio.sleep(600)

    try:
        print("初始化資料庫...")
        await init_db()
        
        tasks_to_run = []
        mode = sys.argv[1] if len(sys.argv) > 1 else "all"
        
        if mode in ["all", "discord"]:
            tasks_to_run.append(asyncio.create_task(start_discord_bot_task()))
        if mode in ["all", "web"]:
            tasks_to_run.append(asyncio.create_task(start_web_server_task()))

        # 只有在 discord bot 運行時才啟動更新檢查器
        if mode in ["all", "discord"]:
            update_checker_task = asyncio.create_task(start_github_update_checker_task())
            tasks_to_run.append(update_checker_task)

        print(f"\n啟動 AI戀人系統 (模式: {mode})...")
        
        # [v6.0 核心修正] 等待關閉事件
        if tasks_to_run:
            await shutdown_event.wait()
            print("收到關閉信號，正在優雅地終止所有任務...")
            # 取消所有正在運行的任務
            for task in tasks_to_run:
                task.cancel()
            await asyncio.gather(*tasks_to_run, return_exceptions=True)

    except Exception as e:
        print(f"\n主程式運行時發生未處理的錯誤: {str(e)}")
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
