# main.py 的中文註釋(v11.1 - Import修正)
# 更新紀錄:
# v11.1 (2025-09-26): [災難性BUG修復] 在文件頂部添加了所有運行FastAPI Web伺服器所需的、缺失的import語句（`FastAPI`, `Request`, `HTMLResponse`, `StaticFiles`, `Jinja2Templates`），並對import塊進行了PEP 8標準化分組，徹底解決了NameError問題。
# v11.0 (2025-09-26): [重大架構升級] 引入了全局的、启动时的【Ollama健康检查】机制。
# v10.1 (2025-09-26): [災難性BUG修復] 將 PROJ_DIR 和快取清理邏輯提升到所有 src 導入之前。

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

    # 函式：Ollama健康檢查與自動下載 (v1.1 - HTTP方法修正)
    # 更新紀錄:
    # v1.1 (2025-09-26): [災難性BUG修復] 根據網路搜尋的Ollama API文件驗證，將檢查本地模型的API呼叫從不正確的 `POST` 方法修正為正確的 `GET` 方法，徹底解決了 `405 Method Not Allowed` 錯誤。
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
                # [v1.1 核心修正] 使用 GET 方法，並移除不必要的 json 參數
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
            logger.error(f"🔥 [Ollama Health Check] 检查本地模型列表时发生错误: {e}", exc_info=True)
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

def _check_and_install_dependencies():
    """檢查並安裝缺失的 Python 依賴項，包括 spaCy 和其模型。"""
    import importlib.util
    
    required_packages = {
        'uvicorn': 'uvicorn', 'fastapi': 'fastapi', 'SQLAlchemy': 'sqlalchemy',
        'aiosqlite': 'aiosqlite', 'discord.py': 'discord', 'langchain': 'langchain',
        'langchain-core': 'langchain_core', 'langchain-google-genai': 'langchain_google_genai',
        'langchain-community': 'langchain_community', 'langchain-chroma': 'langchain_chroma',
        'langchain-cohere': 'langchain_cohere', 'google-generativeai': 'google.generativeai',
        'chromadb': 'chromadb', 'rank_bm25': 'rank_bm25',
        'pydantic-settings': 'pydantic_settings', 'Jinja2': 'jinja2',
        'python-Levenshtein': 'Levenshtein',
        'spacy': 'spacy', 'httpx': 'httpx'
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            if importlib.util.find_spec(import_name) is None:
                raise ImportError
            if package_name == 'spacy':
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
        print("\n🔄 依賴項安裝完畢。需要重啟以加載新模組。")
        sys.exit(0)

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

# --- 異步守護任務與核心服務 ---
async def start_git_log_pusher_task(lock: asyncio.Lock):
    """一個完全獨立的背景任務，定期將最新的日誌檔案推送到GitHub倉庫。"""
    await asyncio.sleep(15)
    print("✅ [守護任務] LOG 自動推送器已啟動。")
    
    log_file_path = PROJ_DIR / "data" / "logs" / "app.log"
    upload_log_path = PROJ_DIR / "latest_log.txt"

    def run_git_commands_sync() -> bool:
        try:
            if not log_file_path.is_file(): return False
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
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
                await asyncio.to_thread(run_git_commands_sync)
            
            await asyncio.sleep(300) 
        except asyncio.CancelledError:
            print("⚪️ [LOG Pusher] 背景任務被正常取消。")
            break
        except Exception as e:
            print(f"🔥 [LOG Pusher] 背景任務主循環發生錯誤: {e}")
            await asyncio.sleep(60)

async def start_github_update_checker_task(lock: asyncio.Lock):
    """一個獨立的背景任務，檢查GitHub更新並在必要時觸發重啟。"""
    await asyncio.sleep(10)
    print("✅ [守護任務] GitHub 自動更新檢查器已啟動。")
    
    def run_git_command_sync(command: list) -> tuple[int, str, str]:
        process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False, cwd=PROJ_DIR)
        return process.returncode, process.stdout, process.stderr
        
    while not shutdown_event.is_set():
        try:
            async with lock:
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
            
            await asyncio.sleep(300)

        except asyncio.CancelledError:
            print("⚪️ [自動更新] 背景任務被正常取消。")
            break
        except Exception as e:
            print(f"🔥 [自動更新] 檢查更新時發生未預期的錯誤: {type(e).__name__}: {e}")
            await asyncio.sleep(600)

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

async def main():
    MAIN_PY_VERSION = "v11.1" # 版本號更新
    print(f"--- AI Lover 主程式 ({MAIN_PY_VERSION}) ---")
    
    try:
        _check_and_install_dependencies()
        
        ollama_model_to_check = "HammerAI/llama-3-lexi-uncensored:latest"
        is_ollama_ready = await _ollama_health_check(ollama_model_to_check)
        
        db_ready_event = asyncio.Event()
        print("\n初始化資料庫...")
        await init_db(db_ready_event)
        
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
