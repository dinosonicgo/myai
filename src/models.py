# models.py 的中文註釋(v17.0 - 新增導演視角)
# 更新紀錄:
# v17.0 (2025-09-06): [災難性BUG修復] 在 GameState 模型中增加了 viewing_mode 和 remote_target_path 兩個關鍵欄位。這創建了一個持久化的“導演視角”狀態機，旨在從根本上解決 AI 在生成遠程場景後，下一輪互動錯誤地將使用者角色重新拉回場景的“上下文洩漏”問題。
# v16.0 (2025-08-12): [重大架構重構] 移除了所有基礎 LORE 模型 (CharacterProfile, LocationInfo 等) 的本地定義，改為從新的 `schemas.py` 檔案中導入。
# v15.0 (2025-08-12): [架構重構] 移除了本地的 CharacterAction 和 TurnPlan 模型定義。

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
    location_path: List[str] = Field(default_factory=lambda: ["時空奇點"], description="表示使用者角色【當前的真實物理位置】的層級路徑。")
    inventory: List[str] = Field(default_factory=list, description="團隊共用的儲存空間，存放【未被穿戴】的物品。")
    # [v17.0 新增] 導演視角狀態機
    viewing_mode: Literal['local', 'remote'] = Field(default='local', description="當前玩家的視角模式。'local'表示正在與身邊環境互動，'remote'表示正在觀察遠程地點。")
    remote_target_path: Optional[List[str]] = Field(default=None, description="如果 viewing_mode 為 'remote'，這裡儲存遠程觀察的目標路徑。")
    
    @field_validator('inventory', 'location_path', 'remote_target_path', mode='before')
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
