# src/logger.py 的中文註釋(v1.0)
# 更新紀錄:
# v1.0 (2050-08-05):
# 1. [全新創建] 創建了中央日誌系統，使用 Python 標準的 logging 模組。
# 2. [品質提升] 定義了統一的日誌格式，包含時間戳、級別和訊息，使後台監控更清晰。

import logging
import sys

# 創建一個名為 "AILoverApp" 的 logger 實例
# 使用命名空間可以避免與其他庫的 logger 發生衝突
logger = logging.getLogger("AILoverApp")

# 設置 logger 的最低處理級別為 INFO
# 這意味著 INFO, WARNING, ERROR, CRITICAL 等級的日誌都會被處理
logger.setLevel(logging.INFO)

# 創建一個 handler，用於將日誌訊息輸出到標準輸出（終端機）
handler = logging.StreamHandler(sys.stdout)

# 定義日誌訊息的格式
# asctime: 訊息時間
# levelname: 日誌級別 (e.g., INFO, WARNING)
# message: 日誌訊息內容
formatter = logging.Formatter(
    '%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 將格式應用於 handler
handler.setFormatter(formatter)

# 將 handler 添加到 logger 中
# 如果 logger 中沒有 handler，日誌訊息將不會被輸出
if not logger.handlers:
    logger.addHandler(handler)