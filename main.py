# main.py 的中文註釋(v3.0 - 依賴自安裝)
# 更新紀錄:
# v3.0 (2025-08-12):
# 1. [重大功能新增] 新增了 `_check_and_install_dependencies` 函式，實現了依賴項的自動檢查與安裝。
# 2. [健壯性] 現在程式啟動時會自動檢測所有必要的函式庫，如果缺失則會嘗試使用 pip 安裝，並自動重新啟動以應用變更。這使得程式在全新環境下真正實現了「一鍵啟動」。
# 3. [架構改進] 將依賴項檢查放在主流程的最前端，確保後續所有導入和操作的環境完整性。
# v2.1 (2025-08-05):
# 1. [健壯性] 修改了 `start_discord_bot_task` 函式。不再直接呼叫 `bot.start()`，而是先創建 Bot 實例，然後使用 `async with bot:` 的上下文管理器來啟動。
# 2. [BUG修復] 這種新的啟動方式是 `discord.py` 2.0+ 的推薦做法，它能確保 Bot 的背景任務（如我們新增的健康檢查）被正確地加載、啟動和在關閉時被妥善清理，從而解決了之前背景任務無法運行的問題。
# v2.0 (2025-08-04):
# 1. [架構修改] 移除了在 web 模式下自動開啟瀏覽器的功能，因為現在它作為後端服務由主應用程式管理。

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

# FastAPI 應用實例化
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 函式：檢查並安裝依賴項
# 說明：檢查必要的 Python 套件是否已安裝，如果沒有，則嘗試自動安裝並重啟程式。
def _check_and_install_dependencies():
    """
    檢查所有必要的 Python 套件是否已安裝。
    如果發現任何缺失的套件，它會嘗試使用 pip 進行安裝，
    然後自動重新啟動應用程式以載入新安裝的套件。
    """
    # PyPI 套件名稱與其在程式中導入時的名稱的對應關係
    # 格式: 'pypi-package-name': 'import_name'
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
        'python-Levenshtein': 'Levenshtein' # v3.1 新增，用於工具備援方案
    }

    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            # 嘗試導入來檢查
            # 對於帶點的導入名稱，需要特殊處理
            if '.' in import_name:
                 __import__(import_name)
            else:
                importlib.metadata.version(package_name)
            print(f"✅ 依賴項 '{package_name}' 已安裝。")
        except (ImportError, importlib.metadata.PackageNotFoundError):
            print(f"❌ 依賴項 '{package_name}' 未找到。")
            missing_packages.append(package_name)

    if missing_packages:
        print("\n⏳ 正在自動安裝缺失的依賴項，請稍候...")
        for package in missing_packages:
            try:
                print(f"   -> 正在安裝 {package}...")
                # 使用 subprocess 呼叫 pip 來安裝套件
                # --quiet 選項可以減少不必要的輸出
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--quiet", package]
                )
                print(f"   ✅ {package} 安裝成功。")
            except subprocess.CalledProcessError:
                print(f"   🔥 {package} 安裝失敗！請手動執行 'pip install {package}' 後再試。")
                if os.name == 'nt':
                    os.system("pause")
                sys.exit(1) # 如果安裝失敗，則終止程式

        print("\n🔄 所有依賴項已安裝完畢。程式將在 3 秒後自動重啟以應用變更...")
        time.sleep(3)
        # 使用 os.execv 來用一個新進程替換當前進程，實現重啟
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
    _check_and_install_dependencies()

    # --- 延遲導入 ---
    from src.database import init_db
    from src.config import settings
    
    from src.web_server import router as web_router
    app.include_router(web_router)
    # --- 延遲導入結束 ---

    # 函式：啟動 Discord Bot 的異步任務
    async def start_discord_bot_task():
        from src.discord_bot import AILoverBot
        
        if not settings.DISCORD_BOT_TOKEN:
            print("錯誤：DISCORD_BOT_TOKEN 未在 config/.env 檔案中設定。Discord Bot 將無法啟動。")
            print("此模式將在 10 秒後終止...")
            await asyncio.sleep(10)
            return
        
        try:
            bot = AILoverBot()
            # [v2.1 修正] 使用 async with 來啟動 Bot，確保背景任務能被正確加載和管理
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

# 函式：啟動 GitHub 自動更新檢查器的異步任務 (v1.1 健壯性修正)
# 說明：一個背景任務，定期檢查遠端 GitHub 倉庫是否有更新。
#      如果有，則自動拉取最新程式碼並重啟應用。
async def start_github_update_checker_task():
    """
    每隔 5 分鐘檢查一次 GitHub 倉庫是否有新的提交。
    如果有，則自動執行 'git pull' 並重啟程式。
    """
    # 等待 10 秒，讓主程式完全啟動後再開始檢查
    await asyncio.sleep(10)
    print("✅ 背景任務：GitHub 自動更新檢查器已啟動。")

    while True:
        try:
            # 步驟 1: 從遠端獲取最新的分支資訊，但不合併
            git_fetch_process = await asyncio.create_subprocess_shell(
                'git fetch',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            await git_fetch_process.wait()

            # 步驟 2: 檢查本地分支是否落後於遠端分支
            # -uno 表示不顯示未追蹤的檔案，使輸出更乾淨
            git_status_process = await asyncio.create_subprocess_shell(
                'git status -uno',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await git_status_process.communicate()

            if git_status_process.returncode == 0:
                status_output = stdout.decode('utf-8')
                # 不同的 Git 版本和語言環境可能有不同的提示，這裡檢查最常見的一種
                if "Your branch is behind" in status_output or "您的分支落後" in status_output:
                    print("\n🔄 [自動更新] 偵測到遠端倉庫有新版本，正在更新...")
                    
                    # 步驟 3: 拉取最新的程式碼
                    git_pull_process = await asyncio.create_subprocess_shell(
                        'git pull',
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    pull_stdout, pull_stderr = await git_pull_process.communicate()

                    if git_pull_process.returncode == 0:
                        print("✅ [自動更新] 程式碼更新成功！")
                        print("🔄 應用程式將在 3 秒後自動重啟以應用變更...")
                        await asyncio.sleep(3)
                        
                        # 使用與依賴安裝相同的機制來重啟程式
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                    else:
                        print("🔥 [自動更新] 'git pull' 失敗。請手動檢查程式碼目錄。")
                        print(f"   錯誤訊息: {pull_stderr.decode('utf-8')}")

            # 每 300 秒（5分鐘）檢查一次
            await asyncio.sleep(300)

        except FileNotFoundError:
            print("🔥 [自動更新] 錯誤: 'git' 命令未找到。自動更新功能已停用。")
            print("   請確保您是透過 launcher.py 啟動，並且系統已安裝 Git。")
            break # 停止循環
        except Exception as e:
            # [v1.1 修正] 增加異常類型的輸出，確保能看到錯誤詳情
            print(f"🔥 [自動更新] 檢查更新時發生未預期的錯誤: {type(e).__name__}: {e}")
            # 發生錯誤後，等待更長的時間再重試，避免頻繁報錯
            await asyncio.sleep(600)
# 函式：啟動 GitHub 自動更新檢查器的異步任務 (v1.1 健壯性修正)


    try:
        print("初始化資料庫...")
        await init_db()

        # [新增] 在主流程中啟動背景更新檢查任務
        # 我們使用 asyncio.create_task 讓它在背景獨立運行，不阻塞主線程
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
        else: # 預設為 'all' 模式
            print("模式: 同時啟動 Web 伺服器與 Discord Bot (預設)")
            print("Discord Bot 正在啟動...")
            print("Web介面將在 http://localhost:8000 運行")
            tasks_to_run.append(start_discord_bot_task())
            tasks_to_run.append(start_web_server_task())

        if tasks_to_run:
            await asyncio.gather(*tasks_to_run)
        
        # 確保在程式主任務結束後，更新檢查器也能被正常取消
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
        print(f"\n程式啟動失敗，發生致命錯誤: {e}")
        if os.name == 'nt':
            os.system("pause")
