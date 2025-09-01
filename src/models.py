# models.py 的中文註釋(v16.0 - 依賴鏈重構)
# 更新紀錄:
# v16.0 (2025-08-12):
# 1. [重大架構重構] 移除了所有基礎 LORE 模型 (CharacterProfile, LocationInfo 等) 的本地定義，改為從新的 `schemas.py` 檔案中導入。
# 2. [BUG修復] 使 models.py 成為依賴於 schemas.py 的上層模型定義檔案，徹底解決了循環導入問題，並建立了清晰的單向依賴鏈。
# v15.0 (2025-08-12):
# 1. [架構重構] 移除了本地的 CharacterAction 和 TurnPlan 模型定義。

import json
import re
from typing import Optional, Dict, List, Any, Literal
from pydantic import BaseModel, Field, field_validator

# [v16.0 修正] 從 schemas 導入所有基礎 LORE 模型和驗證器
from .schemas import (
    CharacterProfile, Quest, LocationInfo, ItemInfo, CreatureInfo, WorldLore,
    _validate_string_to_list
)

# --- 頂層數據模型 ---

class GameState(BaseModel):
    money: int = 100
    location_path: List[str] = Field(default_factory=lambda: ["時空奇點"], description="表示當前地點的層級路徑，例如 ['艾爾文森林', '閃金鎮', '獅王之傲旅店']。")
    inventory: List[str] = Field(default_factory=list, description="團隊共用的儲存空間，存放【未被穿戴】的物品。")
    
    @field_validator('inventory', 'location_path', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

class UserProfile(BaseModel):
    user_id: str
    user_profile: CharacterProfile
    ai_profile: CharacterProfile
    affinity: int = Field(default=0, description="AI 戀人對使用者的好感度，範圍從 -1000 (憎恨) 到 1000 (愛戀)。")
    world_settings: Optional[str] = None
    one_instruction: Optional[str] = None
    response_style_prompt: Optional[str] = None
    game_state: GameState = Field(default_factory=GameState)
    
UserProfile.model_rebuild()

class Memory(BaseModel):
    content: str
    timestamp: float
    importance: int = 1
    
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: float

class PersonalMemoryEntry(BaseModel):
    should_save: bool = Field(description="判斷剛剛的對話是否包含了對 AI 自身有意義的、值得記住的成長、感悟或決定。如果是，則為 true，否則為 false。")
    thought: str = Field(description="如果 should_save 為 true，則在此以 AI 的第一人稱，簡短地記錄下這次的感悟或決定。")