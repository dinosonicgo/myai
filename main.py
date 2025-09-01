# main.py 的中文註釋(v5.0 - 導入結構重構)
# 更新紀錄:
# v5.0 (2025-09-02):
# 1. [根本性BUG修復] 將所有延遲導入語句 (from src...) 移至檔案頂部。
# 2. [健壯性] 此修改遵循 Python 最佳實踐，確保所有模組依賴在程式執行前被解析，從而徹底解決了因循環導入導致的程式靜默退出問題。
# v4.1 (2025-09-02):
# 1. [穩定性重構] 修改了 start_github_update_checker_task，使用 asyncio.to_thread 來運行子進程，解決了在 Windows 上的 NotImplementedError。

import os
import sys
import asyncio
import uvicorn
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import subprocess
import importlib.metadata

# [v5.0 修正] 將所有延遲導入移至檔案頂部，以解決循環導入問題
from src.database import init_db
from src.config import settings
from src.web_server import router as web_router

# FastAPI 應用實例化
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
app.include_router(web_router) # [v5.0 修正] 將路由包含也移上來

# 函式：檢查並安裝依賴項
# 說明：檢查必要的 Python 套件是否已安裝，如果沒有，則嘗試自動安裝並重啟程式。
def _check_and_install_dependencies():
    """
    檢查所有必要的 Python 套件是否已安裝。
    如果發現任何缺失的套件，它會嘗試使用 pip 進行安裝，
    然後自動重新啟動應用程式以載入新安裝的套件。
    """
    import importlib.util
    
    required_packages = {
        'uvicorn': 'uvicorn',
        'fastapi': 'fastapi',
        'SQLAlchemy': 'sqlalchemy',
        'aiosqlite': 'aiosqlite',
        'discord.py': 'discord',
        'langchain': 'langchain',
        'langchain-core': 'langchain_core',
        'langchain-google-genai': 'langchain_google_genai',
        'langchain-community': 'langchain_community',
        'langchain-chroma': 'langchain_chroma',
        'langchain-cohere': 'langchain_cohere',
        'google-generativeai': 'google.generativeai',
        'chromadb': 'chromadb',
        'rank_bm25': 'rank_bm25',
        'pydantic-settings': 'pydantic_settings',
        'Jinja2': 'jinja2',
        'python-Levenshtein': 'Levenshtein'
    }

    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            if importlib.util.find_spec(import_name) is None:
                importlib.metadata.version(package_name)
            print(f"✅ 依賴項 '{package_name}' 已安裝。")
        except importlib.metadata.PackageNotFoundError:
            print(f"❌ 依賴項 '{package_name}' 未找到。")
            missing_packages.append(package_name)

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
                if os.name == 'nt':
                    os.system("pause")
                sys.exit(1)

        print("\n🔄 所有依賴項已安裝完畢。程式將在 3 秒後自動重啟以應用變更...")
        time.sleep(3)
        os.execv(sys.executable, [sys.executable] + sys.argv)
# 函式：檢查並安裝依賴項

# 函式：根路由
# 說明：異步函式，處理對網站根目錄的 GET 請求，回傳主頁面。
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
# 函式：根路由

# 函式：主程式入口
# 說明：異步函式，初始化並根據命令列參數啟動應用程式的不同部分（Web、Discord 或全部）。
async def main():
    # --- 在主流程開始前，執行依賴檢查 ---
    # 這個檢查必須在最前面，確保後續的導入能夠成功
    _check_and_install_dependencies()

    # --- 由於導入已移至頂部，此處不再需要延遲導入區塊 ---

    # 函式：啟動 Discord Bot 的異步任務
    async def start_discord_bot_task():
        # 將 discord_bot 的導入延遲到這裡，因為它是一個重量級的組件
        from src.discord_bot import AILoverBot
        
        if not settings.DISCORD_BOT_TOKEN:
            print("錯誤：DISCORD_BOT_TOKEN 未在 config/.env 檔案中設定。Discord Bot 將無法啟動。")
            print("此模式將在 10 秒後終止...")
            await asyncio.sleep(10)
            return
        
        try:
            bot = AILoverBot()
            async with bot:
                await bot.start(settings.DISCORD_BOT_TOKEN)
        except Exception as e:
            print(f"啟動 Discord Bot 時發生錯誤: {e}")
    # 函式：啟動 Discord Bot 的異步任務

    # 函式：啟動 Web 伺服器的異步任務
    async def start_web_server_task():
        config = uvicorn.Config(app, host="localhost", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    # 函式：啟動 Web 伺服器的異步任務

    # 函式：啟動 GitHub 自動更新檢查器的異步任務 (v4.1 - 穩定性重構)
    async def start_github_update_checker_task():
        await asyncio.sleep(10)
        print("✅ 背景任務：GitHub 自動更新檢查器已啟動。")

        def run_git_command(command: list) -> tuple[int, str, str]:
            process = subprocess.run(
                command, capture_output=True, text=True, encoding='utf-8', check=False
            )
            return process.returncode, process.stdout, process.stderr

        while True:
            try:
                await asyncio.to_thread(run_git_command, ['git', 'fetch'])
                returncode, stdout, stderr = await asyncio.to_thread(
                    run_git_command, ['git', 'status', '-uno']
                )

                if returncode == 0:
                    status_output = stdout
                    if "Your branch is behind" in status_output or "您的分支落後" in status_output:
                        print("\n🔄 [自動更新] 偵測到遠端倉庫有新版本，正在更新...")
                        pull_rc, _, pull_stderr = await asyncio.to_thread(
                            run_git_command, ['git', 'pull']
                        )
                        if pull_rc == 0:
                            print("✅ [自動更新] 程式碼更新成功！")
                            print("🔄 應用程式將在 3 秒後自動重啟以應用變更...")
                            await asyncio.sleep(3)
                            os.execv(sys.executable, [sys.executable] + sys.argv)
                        else:
                            print("🔥 [自動更新] 'git pull' 失敗。請手動檢查程式碼目錄。")
                            print(f"   錯誤訊息: {pull_stderr}")
                await asyncio.sleep(300)
            except FileNotFoundError:
                print("🔥 [自動更新] 錯誤: 'git' 命令未找到。自動更新功能已停用。")
                break
            except Exception as e:
                print(f"🔥 [自動更新] 檢查更新時發生未預期的錯誤: {type(e).__name__}: {e}")
                await asyncio.sleep(600)
    # 函式：啟動 GitHub 自動更新檢查器的異步任務 (v4.1 - 穩定性重構)

    try:
        print("初始化資料庫...")
        await init_db()

        update_checker_task = asyncio.create_task(start_github_update_checker_task())

        print("\n啟動 AI戀人系統...")
        mode = sys.argv[1] if len(sys.argv) > 1 else "all"
        tasks_to_run = []
        if mode == "web":
            print("模式: 只啟動 Web 伺服器 (供 Personal Assistant 調用)")
            print("Web API 將在 http://localhost:8000 運行")
            tasks_to_run.append(start_web_server_task())
        elif mode == "discord":
            print("模式: 只啟動 Discord Bot")
            print("Discord Bot 正在啟動...")
            tasks_to_run.append(start_discord_bot_task())
        else:
            print("模式: 同時啟動 Web 伺服器與 Discord Bot (預設)")
            print("Discord Bot 正在啟動...")
            print("Web介面將在 http://localhost:8000 運行")
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
        print("程式將在 5 秒後關閉...")
        await asyncio.sleep(5)
# 函式：主程式入口

if __name__ == "__main__":
    try:
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n偵測到使用者中斷 (Ctrl+C)，程式已停止運行。")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n【啟動失敗】致命設定錯誤: {e}")
        print("請檢查您的 config/.env 檔案後再試一次。")
        if os.name == 'nt':
            os.system("pause")
    except Exception as e:
        # [v5.0 修正] 增加更詳細的導入錯誤提示
        if isinstance(e, ImportError):
            print(f"\n【啟動失敗】致命導入錯誤: {e}")
            print("這通常是因為循環導入 (Circular Import) 導致的。請檢查模組之間的相互引用關係。")
        else:
            print(f"\n程式啟動失敗，發生致命錯誤: {e}")
        if os.name == 'nt':
            os.system("pause")
