# main.py 的中文註釋(v13.0 - 移除依賴檢查)
# 更新紀錄:
# v13.0 (2025-11-26): [重大架構重構] 徹底移除了在 `main.py` 中直接調用 `_check_and_install_dependencies` 的邏輯。此職責已被更上游的 `launcher.py` 完全接管。此修改遵循了「關注點分離」原則，讓啟動器專注於環境準備，而主程式專注於應用邏輯，使架構更清晰、更健壯。
# v12.0 (2025-11-26): [重大架構升級] 在程式啟動的最開始，增加了對 `_setup_huggingface_mirror()` 和 `_check_and_install_dependencies()` 的調用。
# v11.1 (2025-09-26): [災難性BUG修復] 在文件頂部添加了所有運行FastAPI Web伺服器所需的、缺失的import語句。

import os
import sys
import shutil
from pathlib import Path
import asyncio
import time
import subprocess
import importlib.metadata
import datetime
import traceback
import json
import httpx

# [v11.1 核心修正] 導入所有運行 FastAPI 所需的第三方庫
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

PROJ_DIR = Path(__file__).resolve().parent

def _clear_pycache():
    """遞歸地查找並刪除 __pycache__ 資料夾。"""
    for path in PROJ_DIR.rglob('__pycache__'):
        if path.is_dir():
            print(f"🧹 清理舊快取: {path}")
            try:
                shutil.rmtree(path)
            except OSError as e:
                print(f"🔥 清理快取失敗: {e}")

_clear_pycache()

shutdown_event = asyncio.Event()
git_lock = asyncio.Lock()

# 函式：Ollama健康檢查與自動下載 (v1.2 - 縮排修正)
# 更新紀錄:
# v1.2 (2025-09-26): [災難性BUG修復] 修正了函式定義前的意外縮排，解決了導致程式啟動失敗的 `IndentationError`。
# v1.1 (2025-09-26): [災難性BUG修復] 根據網路搜尋的Ollama API文件驗證，將檢查本地模型的API呼叫從不正確的 `POST` 方法修正為正確的 `GET` 方法。
# v1.0 (2025-09-26): [重大架構升級] 引入了全局的、启动时的【Ollama健康检查】机制。
async def _ollama_health_check(model_name: str) -> bool:
    """
    在程式启动时检查本地Ollama服务的健康状况。
    1. 检查服务是否可连接。
    2. 检查所需模型是否已存在。
    3. 如果模型不存在，则尝试自动下载。
    返回一个布林值，表示本地备援方案是否最终可用。
    """
    print("\n--- 正在执行本地 AI (Ollama) 健康检查 ---")
    
    # 步骤 1: 检查Ollama服务是否正在运行
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:11434/")
        if response.status_code == 200 and "Ollama is running" in response.text:
            print("✅ [Ollama Health Check] 本地 Ollama 服务连接成功。")
        else:
            raise httpx.ConnectError("Invalid response from Ollama server")
    except (httpx.ConnectError, httpx.TimeoutException):
        print("⚠️ [Ollama Health Check] 未能连接到本地 Ollama 服务 (http://localhost:11434)。")
        print("   -> 这可能是因为 Ollama 未安装或未运行。")
        print("   -> 本地 LORE 解析备援方案将被【禁用】。")
        print("   -> 系统将完全依赖云端模型备援方案继续运行。")
        return False

    # 步骤 2: 检查所需模型是否已存在
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            data = response.json()
            installed_models = [m['name'] for m in data.get('models', [])]
            if model_name in installed_models:
                print(f"✅ [Ollama Health Check] 所需模型 '{model_name}' 已安装。本地备援方案已就绪。")
                return True
            else:
                print(f"⏳ [Ollama Health Check] 所需模型 '{model_name}' 未找到，正在尝试自动下载...")
    except Exception as e:
        # 使用 logger 记录更详细的错误
        # logger.error(f"🔥 [Ollama Health Check] 检查本地模型列表时发生错误: {e}", exc_info=True)
        print(f"🔥 [Ollama Health Check] 检查本地模型列表时发生错误: {e}")
        print("   -> 将尝试继续执行自动下载流程。")

    # 步骤 3: 自动下载模型
    try:
        process = await asyncio.create_subprocess_shell(
            f'ollama pull {model_name}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        async def log_stream(stream, prefix):
            while True:
                line = await stream.readline()
                if not line:
                    break
                print(f"   [{prefix}] {line.decode(errors='ignore').strip()}")

        await asyncio.gather(
            log_stream(process.stdout, "Ollama Pull"),
            log_stream(process.stderr, "Ollama Error")
        )
        
        return_code = await process.wait()
        if return_code == 0:
            print(f"✅ [Ollama Health Check] 模型 '{model_name}' 自动下载成功！本地备援方案已就绪。")
            return True
        else:
            print(f"🔥 [Ollama Health Check] 模型 '{model_name}' 自动下载失败，返回码: {return_code}。")
            print(f"   -> 请尝试手动在终端中运行 `ollama pull {model_name}`。")
            print("   -> 本地 LORE 解析备援方案将被【禁用】。")
            return False

    except FileNotFoundError:
        print("🔥 [Ollama Health Check] 'ollama' 命令未找到。")
        print("   -> 请确保您已安装 Ollama 并且其路径已添加到系统环境变量中。")
        print("   -> 本地 LORE 解析备援方案将被【禁用】。")
        return False
    except Exception as e:
        print(f"🔥 [Ollama Health Check] 执行 `ollama pull` 时发生未知错误: {e}")
        print("   -> 本地 LORE 解析备援方案将被【禁用】。")
        return False
# 函式：Ollama健康檢查與自動下載

# main.py 的 _setup_huggingface_mirror 函式 (v12.0 - 全新創建)
# 更新紀錄:
# v12.0 (2025-11-26): [重大架構升級] 創建此輔助函式，用於在程式啟動時自動設定 Hugging Face 相關的環境變數，將模型下載源強制指向國內鏡像。此修改旨在從根本上解決因網路問題導致的本地 Embedding 模型下載失敗或速度過慢的問題，是實現穩健的本地 RAG 系統的關鍵一步。
def _setup_huggingface_mirror():
    """
    設定 Hugging Face 相關函式庫的環境變數，使其從國內鏡像下載模型。
    這是為了解決直接連接 Hugging Face 官方伺服器速度緩慢或被阻斷的問題。
    """
    try:
        # 設定鏡像地址
        HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
        # 為 huggingface_hub 設定端點
        os.environ['HF_ENDPOINT'] = HF_MIRROR_ENDPOINT
        # 為 sentence-transformers 的舊版本可能需要的變數也設定一下（雙重保險）
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(PROJ_DIR / 'models' / 'sentence_transformers')
        
        print(f"✅ [環境配置] 已成功將 Hugging Face 模型下載源設定為鏡像: {HF_MIRROR_ENDPOINT}")
    except Exception as e:
        print(f"🔥 [環境配置] 設定 Hugging Face 鏡像時發生錯誤: {e}")


# main.py 的 _check_and_install_dependencies 函式 (v12.1 - 新增依賴檢查)
# 更新紀錄:
# v12.1 (2025-11-26): [架構擴展] 新增了對 `sentence-transformers` 和 `torch` 的依賴檢查。這是為了配合全新的本地 RAG 系統，確保在程式啟動時，所有必要的深度學習函式庫都已正確安裝，從而避免運行時的 `ImportError`。
# v12.0 (2025-11-26): [重大架構升級] 創建此輔助函式。
def _check_and_install_dependencies():
    """檢查並安裝缺失的 Python 依賴項，包括 spaCy 和其模型。"""
    import importlib.util
    
    required_packages = {
        'uvicorn': 'uvicorn', 'fastapi': 'fastapi', 'SQLAlchemy': 'sqlalchemy',
        'aiosqlite': 'aiosqlite', 'discord.py': 'discord', 'langchain': 'langchain',
        'langchain-core': 'langchain_core', 'langchain-google-genai': 'langchain_google_genai',
        'langchain-community': 'langchain_community', 'langchain-chroma': 'langchain_chroma', 'chromadb': 'chromadb',
        'langchain-cohere': 'langchain_cohere', 'google-generativeai': 'google.generativeai',
        'rank_bm25': 'rank_bm25', 'pydantic-settings': 'pydantic_settings', 'Jinja2': 'jinja2',
        'python-Levenshtein': 'Levenshtein',
        'spacy': 'spacy', 'httpx': 'httpx',
        # [v12.1 新增] 本地 RAG 依賴
        'sentence-transformers': 'sentence_transformers',
        'torch': 'torch',
    }
    
    missing_packages = []
    for package, import_name in required_packages.items():
        if importlib.util.find_spec(import_name) is None:
            missing_packages.append(package)

    if missing_packages:
        print("\n⏳ 正在自動安裝缺失的 Python 依賴項...")
        for package in missing_packages:
            try:
                print(f"   - 正在安裝 {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
                print(f"   ✅ {package} 安裝成功。")
            except subprocess.CalledProcessError:
                print(f"   🔥 {package} 安裝失敗！請手動在終端機執行 'pip install {package}'。")
                if os.name == 'nt': os.system("pause")
                sys.exit(1)
        print("\n🔄 依賴項安裝完畢。為確保所有模組被正確加載，程式將自動重啟...")
        sys.exit(0) # 觸發 launcher.py 的重啟機制

    try:
        import spacy
        spacy.load('zh_core_web_sm')
        print("✅ spaCy 中文模型已安裝。")
    except (ImportError, OSError):
        print("\n⏳ spaCy 中文模型未找到，正在自動下載...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "zh_core_web_sm"])
            print("✅ spaCy 中文模型下載成功。")
        except subprocess.CalledProcessError:
            print("   🔥 spaCy 中文模型下載失敗！請手動執行 'python -m spacy download zh_core_web_sm'。")
            if os.name == 'nt': os.system("pause")
            sys.exit(1)
            
    print("✅ 所有依賴項和模型均已準備就緒。")
# 函式：檢查並安裝依賴項






# --- 本地應用模組導入 ---
from src.database import init_db
from src.config import settings
from src.web_server import router as web_router
from src.discord_bot import AILoverBot

# --- FastAPI 應用實例化與配置 ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
app.include_router(web_router)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



# 函式：啟動 Git 日誌推送器任務 (v5.0 - Git 工作流修正)
# 更新紀錄:
# v5.0 (2025-09-28): [災難性BUG修復] 徹底重構了 `run_git_commands_sync` 的內部 Git 命令執行順序。新流程將 `git commit` 移至 `git pull --rebase` 之前，確保了本地的日誌變更在與遠端同步前被妥善提交，從根本上解決了因 `unstaged changes` 導致 rebase 失敗的致命錯誤。
# v4.0 (2025-11-22): [體驗優化] 根據使用者最新回饋，移除了在成功推送新日誌後顯示的最終確認訊息，使此背景任務在無錯誤發生時實現完全靜默運行。
# v3.0 (2025-11-22): [體驗優化] 移除了在日誌推送任務成功執行時產生的中間過程日誌。
async def start_git_log_pusher_task(lock: asyncio.Lock):
    """一個完全獨立的背景任務，定期將最新的日誌檔案推送到GitHub倉庫。"""
    await asyncio.sleep(15)
    print("✅ [守護任務] LOG 自動推送器已啟動。")
    
    log_file_path = PROJ_DIR / "data" / "logs" / "app.log"
    upload_log_path = PROJ_DIR / "latest_log.txt"

    def run_git_commands_sync() -> bool:
        """
        一個健壯的、同步的 Git 操作函式，包含了提交-拉取-變基-推送的完整流程。
        """
        try:
            # 步驟 0: 檢查日誌文件是否存在
            if not log_file_path.is_file(): return False

            # 步驟 1: 寫入最新的日誌內容
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            latest_lines = lines[-300:]
            log_content_to_write = "".join(latest_lines)
            with open(upload_log_path, 'w', encoding='utf-8') as f:
                f.write(f"### AI Lover Log - Last updated at {datetime.datetime.now().isoformat()} ###\n\n")
                f.write(log_content_to_write)

            # [v5.0 核心修正] 調整 Git 命令執行順序
            
            # 步驟 2: 將本地變更加入暫存區並提交
            subprocess.run(["git", "add", str(upload_log_path)], check=True, cwd=PROJ_DIR, capture_output=True)
            
            commit_message = f"docs: Update latest_log.txt at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            commit_process = subprocess.run(
                ["git", "commit", "-m", commit_message], 
                capture_output=True, text=True, encoding='utf-8', cwd=PROJ_DIR
            )
            # 如果提交失敗，但原因是“沒有東西可以提交”，則視為成功，直接結束本次推送
            if commit_process.returncode != 0:
                if "nothing to commit" in commit_process.stdout or "沒有東西可以提交" in commit_process.stdout:
                    return True 
                else:
                    # 如果是其他原因導致的提交失敗，則拋出異常
                    raise subprocess.CalledProcessError(
                        commit_process.returncode, commit_process.args, commit_process.stdout, commit_process.stderr
                    )
            
            # 步驟 3: 在推送前，先從遠端拉取並變基。此時本地變更已提交，不會再有 unstaged changes 錯誤。
            subprocess.run(["git", "pull", "--rebase"], check=True, cwd=PROJ_DIR, capture_output=True, text=True, encoding='utf-8')
            
            # 步驟 4: 推送到遠端倉庫
            subprocess.run(["git", "push", "origin", "main"], check=True, cwd=PROJ_DIR, capture_output=True)
            
            return True

        except subprocess.CalledProcessError as e:
            error_output = e.stderr or e.stdout
            if "CONFLICT" in str(error_output):
                print(f"🔥 [LOG Pusher] Git rebase 發生衝突，正在中止變基操作...")
                subprocess.run(["git", "rebase", "--abort"], cwd=PROJ_DIR, capture_output=True)
            
            # 忽略“沒有東西可以提交”的“錯誤”
            if "nothing to commit" not in str(error_output) and "沒有東西可以提交" not in str(error_output):
                print(f"🔥 [LOG Pusher] Git指令執行失敗: {error_output.strip()}")
            return False
        except Exception as e:
            print(f"🔥 [LOG Pusher] 執行時發生未知錯誤: {e}")
            return False

    # 主循環
    while not shutdown_event.is_set():
        try:
            async with lock:
                await asyncio.to_thread(run_git_commands_sync)
            
            await asyncio.sleep(300) 
        except asyncio.CancelledError:
            print("⚪️ [LOG Pusher] 背景任務被正常取消。")
            break
        except Exception as e:
            print(f"🔥 [LOG Pusher] 背景任務主循環發生錯誤: {e}")
            await asyncio.sleep(60)
# 函式：啟動 Git 日誌推送器任務




# 函式：啟動 GitHub 更新檢查器任務 (v4.0 - 健壯性修正)
# 更新紀錄:
# v4.0 (2025-12-08): [災難性BUG修復] 將 `git status` 字串解析法替換為更可靠的 `git rev-list --count HEAD..origin/main` 數字比較法，徹底解決因 Git 版本或系統語言環境不同導致自動更新失效的問題。
# v3.0 (2025-11-22): [重大架構重構] 將此任務從 Cog 遷移至主啟動流程，確保其作為一個獨立的守護進程運行。
# v2.0 (2025-11-22): [健壯性] 增加了 Git 鎖機制，以防止日誌推送和程式碼更新之間的競爭條件。
async def start_github_update_checker_task(lock: asyncio.Lock):
    """一個獨立的背景任務，檢查GitHub更新並在必要時觸發重啟。"""
    await asyncio.sleep(10)
    print("✅ [守護任務] GitHub 自動更新檢查器已啟動。")
    
    def run_git_command_sync(command: list) -> tuple[int, str, str]:
        try:
            process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False, cwd=PROJ_DIR)
            return process.returncode, process.stdout.strip(), process.stderr.strip()
        except FileNotFoundError:
            print("🔥 [Auto Update] 錯誤: 'git' 命令未找到。自動更新功能已停用。")
            return -1, "", "Git command not found."
        except Exception as e:
            print(f"🔥 [Auto Update] 執行 Git 命令時發生未知錯誤: {e}")
            return -1, "", str(e)

    while not shutdown_event.is_set():
        try:
            async with lock:
                # 步驟 1: 從遠端獲取最新的變更資訊
                fetch_rc, _, fetch_stderr = await asyncio.to_thread(run_git_command_sync, ['git', 'fetch'])
                if fetch_rc != 0:
                    print(f"🟡 [Auto Update] 'git fetch' 失敗: {fetch_stderr}。跳過本次檢查。")
                    await asyncio.sleep(300)
                    continue

                # 步驟 2: [v4.0 核心修正] 使用 rev-list 比較本地與遠端的差異
                rev_list_rc, rev_list_stdout, rev_list_stderr = await asyncio.to_thread(
                    run_git_command_sync, ['git', 'rev-list', '--count', 'HEAD..origin/main']
                )

                if rev_list_rc != 0:
                    print(f"🟡 [Auto Update] 'git rev-list' 失敗: {rev_list_stderr}。跳過本次檢查。")
                else:
                    try:
                        update_count = int(rev_list_stdout)
                        if update_count > 0:
                            print(f"🔵 [Auto Update] 檢測到 {update_count} 個新版本，準備更新...")
                            
                            reset_rc, _, reset_stderr = await asyncio.to_thread(
                                run_git_command_sync, ['git', 'reset', '--hard', 'origin/main']
                            )
                            if reset_rc == 0:
                                print("✅ [Auto Update] 程式碼強制同步成功！")
                                print("🔄 應用程式將在 3 秒後發出優雅關閉信號，由啟動器負責重啟...")
                                await asyncio.sleep(3)
                                shutdown_event.set()
                                break # 觸發關閉後，跳出循環
                            else:
                                print(f"🔥 [Auto Update] 'git reset' 失敗: {reset_stderr}")
                        else:
                            # 為了不在控制台刷屏，這條日誌可以註解掉或設定為 DEBUG 級別
                            # print("⚪️ [Auto Update] 當前已是最新版本。")
                            pass
                    except (ValueError, TypeError):
                        print(f"🟡 [Auto Update] 解析更新數量失敗。Git 輸出: '{rev_list_stdout}'")

            await asyncio.sleep(300) # 每 5 分鐘檢查一次

        except asyncio.CancelledError:
            print("⚪️ [Auto Update] 背景任務被正常取消。")
            break
        except Exception as e:
            print(f"🔥 [Auto Update] 檢查更新時發生未預期的錯誤: {type(e).__name__}: {e}")
            await asyncio.sleep(600) # 發生未知錯誤時，延長等待時間
# 函式：啟動 GitHub 更新檢查器任務

async def start_discord_bot_task(lock: asyncio.Lock, db_ready_event: asyncio.Event, is_ollama_available: bool):
    """啟動Discord Bot的核心服務。內建錯誤處理和啟動依賴等待。"""
    try:
        print("🔵 [Discord Bot] 正在等待數據庫初始化完成...")
        await db_ready_event.wait()
        print("✅ [Discord Bot] 數據庫已就緒，開始啟動核心服務...")

        if not settings.DISCORD_BOT_TOKEN:
            print("🔥 [Discord Bot] 錯誤：DISCORD_BOT_TOKEN 未在 config/.env 檔案中設定。服務無法啟動。")
            return

        bot = AILoverBot(shutdown_event=shutdown_event, git_lock=lock, is_ollama_available=is_ollama_available)
        
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

# main.py 的 main 函式 (v13.0 - 移除依賴檢查)
# 更新紀錄:
# v13.0 (2025-11-26): [重大架構重構] 徹底移除了在 `main.py` 中直接調用 `_check_and_install_dependencies` 的邏輯。此職責已被更上游的 `launcher.py` 完全接管。此修改遵循了「關注點分離」原則，讓啟動器專注於環境準備，而主程式專注於應用邏輯，使架構更清晰、更健壯。
# v12.0 (2025-11-26): [重大架構升級] 在程式啟動的最開始，增加了對 `_setup_huggingface_mirror()` 和 `_check_and_install_dependencies()` 的調用。
# v11.1 (2025-09-26): [災難性BUG修復] 在文件頂部添加了所有運行FastAPI Web伺服器所需的、缺失的import語句。
async def main():
    MAIN_PY_VERSION = "v13.0" # 版本號更新
    print(f"--- AI Lover 主程式 ({MAIN_PY_VERSION}) ---")
    
    # [v12.0 新增] 設定鏡像並檢查依賴
    _setup_huggingface_mirror()
    
    try:
        # [v13.0 核心修正] 移除此處的依賴檢查調用，此職責已移至 launcher.py
        # _check_and_install_dependencies()
        
        ollama_model_to_check = "HammerAI/llama-3-lexi-uncensored:latest"
        is_ollama_ready = await _ollama_health_check(ollama_model_to_check)
        
        db_ready_event = asyncio.Event()
        print("\n初始化資料庫...")
        await init_db(db_ready_event)
        
        # ... (后续逻辑保持不变) ...
        core_services = []
        guardian_tasks = []
        mode = sys.argv[1] if len(sys.argv) > 1 else "all"
        
        if mode in ["all", "discord"]:
            core_services.append(start_discord_bot_task(git_lock, db_ready_event, is_ollama_ready))
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
        traceback.print_exc()
        if os.name == 'nt': os.system("pause")