# main.py 的中文註釋(v5.2 - 快取清理與自我修復)
# 更新紀錄:
# v5.2 (2025-09-02):
# 1. [根本性BUG修復] 在程式啟動的最前端增加了自動清理 __pycache__ 的功能。此修改將從根本上解決因 Python 加載舊的編譯快取而導致 Git 更新不生效的頑固問題。
# 2. [健壯性] 在 main 函式開頭增加了版本號打印，方便遠程診斷當前運行的程式碼版本。
# v5.1 (2025-09-02):
# 1. [健壯性] 修改了自動更新邏輯，改為使用與啟動器相同的 'git reset --hard'，確保更新的絕對性。

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

# [v5.2 新增] 在所有導入之前，先執行一次快取清理
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

# FastAPI 應用實例化
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
    MAIN_PY_VERSION = "v5.2"
    print(f"--- AI Lover 主程式 ({MAIN_PY_VERSION}) ---")
    
    _check_and_install_dependencies()

    async def start_discord_bot_task():
        from src.discord_bot import AILoverBot
        if not settings.DISCORD_BOT_TOKEN:
            print("錯誤：DISCORD_BOT_TOKEN 未在 config/.env 檔案中設定。")
            await asyncio.sleep(10)
            return
        try:
            bot = AILoverBot()
            async with bot:
                await bot.start(settings.DISCORD_BOT_TOKEN)
        except Exception as e:
            print(f"啟動 Discord Bot 時發生錯誤: {e}")

    async def start_web_server_task():
        config = uvicorn.Config(app, host="localhost", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    async def start_github_update_checker_task():
        await asyncio.sleep(10)
        print("✅ 背景任務：GitHub 自動更新檢查器已啟動。")
        def run_git_command(command: list) -> tuple[int, str, str]:
            process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False)
            return process.returncode, process.stdout, process.stderr
        while True:
            try:
                await asyncio.to_thread(run_git_command, ['git', 'fetch'])
                rc, stdout, _ = await asyncio.to_thread(run_git_command, ['git', 'status', '-uno'])
                if rc == 0 and ("Your branch is behind" in stdout or "您的分支落後" in stdout):
                    print("\n🔄 [自動更新] 偵測到遠端倉庫有新版本，正在更新...")
                    pull_rc, _, pull_stderr = await asyncio.to_thread(run_git_command, ['git', 'reset', '--hard', 'origin/main'])
                    if pull_rc == 0:
                        print("✅ [自動更新] 程式碼強制同步成功！")
                        print("🔄 應用程式將在 3 秒後自動重啟以應用變更...")
                        await asyncio.sleep(3)
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                    else:
                        print(f"🔥 [自動更新] 'git reset' 失敗: {pull_stderr}")
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
        update_checker_task = asyncio.create_task(start_github_update_checker_task())
        print("\n啟動 AI戀人系統...")
        mode = sys.argv[1] if len(sys.argv) > 1 else "all"
        tasks_to_run = []
        if mode == "web":
            print("模式: 只啟動 Web 伺服器")
            tasks_to_run.append(start_web_server_task())
        elif mode == "discord":
            print("模式: 只啟動 Discord Bot")
            tasks_to_run.append(start_discord_bot_task())
        else:
            print("模式: 同時啟動 Web 伺服器與 Discord Bot")
            tasks_to_run.append(start_discord_bot_task())
            tasks_to_run.append(start_web_server_task())
        if tasks_to_run:
            await asyncio.gather(*tasks_to_run)
        update_checker_task.cancel()
        try:
            await update_checker_task
        except asyncio.CancelledError:
            print("GitHub 自動更新檢查器已正常關閉。")
    except Exception as e:
        print(f"\n主程式運行時發生未處理的錯誤: {str(e)}")
        await asyncio.sleep(5)

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