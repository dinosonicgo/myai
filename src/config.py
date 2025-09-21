# src/config.py 的中文註釋(v1.7 - 完整性修復)
# 更新紀錄:
# v1.7 (2025-11-17): [完整性修復] 提供了完整的檔案內容，以確保 TEST_GUILD_ID 屬性被正確加載，解決因快取導致的 AttributeError。
# v1.6 (2025-11-17): [功能擴展] 新增了 TEST_GUILD_ID 變數，用於快速同步指令。
# v1.5 (2050-08-01): [健壯性] 重構了錯誤處理，改為拋出具體異常。

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = ROOT_DIR / "config" / ".env"

# 應用程式的統一設定管理類別。
class Settings(BaseSettings):
    """
    應用程式的統一設定管理類別。
    使用 Pydantic-settings，自動從 .env 檔案讀取設定並進行類型驗證。
    """
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH, 
        env_file_encoding='utf-8', 
        extra='ignore',
        case_sensitive=True
    )

    DATABASE_URL: str = "sqlite+aiosqlite:///./data/ai_lover.db"
    
    DISCORD_BOT_TOKEN: Optional[str] = None
    ADMIN_USER_ID: Optional[str] = None
    
    # [v1.6 新增] 用於快速同步指令的測試伺服器ID
    TEST_GUILD_ID: Optional[str] = None
    
    # 明確定義每一個可能的 API Key 變數，Pydantic-settings 會自動尋找並賦值
    GOOGLE_API_KEYS_1: Optional[str] = None
    GOOGLE_API_KEYS_2: Optional[str] = None
    GOOGLE_API_KEYS_3: Optional[str] = None
    GOOGLE_API_KEYS_4: Optional[str] = None
    GOOGLE_API_KEYS_5: Optional[str] = None
    GOOGLE_API_KEYS_6: Optional[str] = None
    
    # 我們將手動構建這個列表，而不是讓 Pydantic 直接讀取
    GOOGLE_API_KEYS_LIST: List[str] = []

    COHERE_KEY: Optional[str] = None

    # 函式：在 pydantic-settings 讀取完所有變數後，手動構建 API 金鑰列表。
    @model_validator(mode='after')
    def build_api_keys_list(self) -> 'Settings':
        """在 pydantic-settings 讀取完所有變數後，手動構建 API 金鑰列表。"""
        keys = []
        # 遍歷所有已定義的單獨金鑰欄位
        for i in range(1, 7):
            key = getattr(self, f"GOOGLE_API_KEYS_{i}")
            if key:
                keys.append(key)
        
        self.GOOGLE_API_KEYS_LIST = keys
        return self
    # 函式：在 pydantic-settings 讀取完所有變數後，手動構建 API 金鑰列表。
# 應用程式的統一設定管理類別結束

# 創建一個全域唯一的設定實例
try:
    if not ENV_FILE_PATH.is_file():
        # [修正] 不再直接退出，而是拋出一個清晰的 FileNotFoundError。
        # 這個異常將由 main.py 捕獲和處理。
        raise FileNotFoundError(f"錯誤：設定檔未找到！請確保在以下路徑中存在 .env 檔案: {ENV_FILE_PATH}")
        
    settings = Settings()

except ValidationError as e:
    # [修正] 如果 .env 檔案內容不符合 Pydantic 模型（例如，類型錯誤），
    # 拋出一個包含詳細驗證錯誤的 ValueError。
    raise ValueError(f"錯誤：設定檔 (config/.env) 格式不正確或缺少必要項。\n{e}")

# [修正] 將對 API 金鑰列表的檢查移到 try 區塊外。
# 確保在 settings 實例成功創建後，再檢查業務邏輯上的必要條件。
if not settings.GOOGLE_API_KEYS_LIST:
    raise ValueError(f"錯誤：在 {ENV_FILE_PATH} 中至少需要設定一個 GOOGLE_API_KEYS_1！請檢查變數名稱是否正確。")
