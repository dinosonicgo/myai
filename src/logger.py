# src/logger.py 的中文註釋(v1.0)
# 更新紀錄:
# v1.0 (2050-08-05):
# 1. [全新創建] 創建了中央日誌系統，使用 Python 標準的 logging 模組。
# 2. [品質提升] 定義了統一的日誌格式，包含時間戳、級別和訊息，使後台監控更清晰。

import logging
import sys
from pathlib import Path

# 定義路徑
PROJ_DIR = Path(__file__).resolve().parent.parent
log_dir = PROJ_DIR / "data" / "logs"
log_file_path = log_dir / "app.log"

# 確保日誌目錄存在
log_dir.mkdir(parents=True, exist_ok=True)

# [核心修改] 確保 .gitignore 中 *沒有* 忽略日誌檔案，以便Git可以追蹤它
gitignore_path = PROJ_DIR / ".gitignore"
log_entry_to_remove = "data/logs/" # 我們要確保這一行不在 .gitignore 中

if gitignore_path.is_file():
    try:
        with open(gitignore_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
        
        # 檢查是否需要修改
        if any(log_entry_to_remove in line for line in lines):
            print(f"🔧 正在從 .gitignore 中移除 '{log_entry_to_remove}' 以便追蹤LOG...")
            # 過濾掉包含目標路徑的行
            new_lines = [line for line in lines if log_entry_to_remove not in line]
            with open(gitignore_path, "w", encoding='utf-8') as f:
                f.writelines(new_lines)
    except Exception as e:
        print(f"🔥 修改 .gitignore 時發生錯誤: {e}")


# 設置 logger
logger = logging.getLogger("AILoverApp")
logger.setLevel(logging.INFO)

# 定義日誌格式
formatter = logging.Formatter(
    '%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 防止重複添加 handler
if not logger.handlers:
    # 終端機 handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # 檔案 handler
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
