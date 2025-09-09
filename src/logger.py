# src/logger.py
import logging
import sys
from pathlib import Path

# 創建 data/logs 目錄如果它不存在
log_dir = Path(__file__).resolve().parent.parent / "data" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / "app.log"

# [核心修改] 確保 .gitignore 中 *沒有* 忽略日誌檔案
gitignore_path = Path(__file__).resolve().parent.parent / ".gitignore"
log_entry_to_remove = "data/logs/"
if gitignore_path.is_file():
    with open(gitignore_path, "r") as f:
        lines = f.readlines()
    
    # 如果忽略規則存在，則將其移除
    if any(log_entry_to_remove in line for line in lines):
        print(f"🔧 正在從 .gitignore 中移除 '{log_entry_to_remove}' 以便追蹤LOG...")
        with open(gitignore_path, "w") as f:
            for line in lines:
                if log_entry_to_remove not in line:
                    f.write(line)

logger = logging.getLogger("AILoverApp")
logger.setLevel(logging.INFO)

# 定義統一的格式
formatter = logging.Formatter(
    '%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 檢查是否已經有 handlers，防止重複添加
if not logger.handlers:
    # 輸出到終端機的 handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # 輸出到檔案的 handler
    # 使用 'a' 模式表示附加，'utf-8' 編碼以支援中文
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
