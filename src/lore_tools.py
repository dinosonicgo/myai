# src/lore_tools.py 的中文註釋(v1.2 - 職責明確化)
# 更新紀錄:
# v1.2 (2025-09-02): [健壯性] 為了從源頭上防止 LORE 重複生成，將 `add_or_update_npc_profile` 重命名為 `create_new_npc_profile`，並修改其描述，明確其職責【只用於創建全新NPC】。這將引導 LLM 在生成計畫時做出更精確的決策，減少對實體解析鏈的依賴。
# v1.1 (2025-09-02): [重大架構重構] 移除了本地的上下文管理，改為從中央 `tool_context` 導入共享實例。
# v1.0 (2025-08-27): [全新創建] 將所有 LORE 相關的工具從 `tools.py` 遷移至此，實現了核心動作工具與世界知識工具的職責分離。

import time
import re
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field, ConfigDict, field_validator, AliasChoices
from langchain_core.tools import tool, Tool

from .logger import logger
from .database import Lore
from .lore_book import add_or_update_lore
from .tool_context import tool_context

# --- Pydantic 模型與配置 ---

def remove_title_from_schema(schema: Dict[str, Any], model: Type['BaseModel']) -> None:
    if 'title' in schema:
        del schema['title']
    for prop in schema.get('properties', {}).values():
        if 'title' in prop:
            del prop['title']

class BaseToolArgs(BaseModel):
    model_config = ConfigDict(
        json_schema_extra=remove_title_from_schema
    )

# --- NPC 相關工具 ---

# 類別：創建新 NPC 檔案參數 (v1.2 - 重命名)
class CreateNewNpcProfileArgs(BaseToolArgs):
    lore_key: str = Field(description="系統內部使用的唯一標識符，由實體解析鏈生成。")
    standardized_name: str = Field(description="由實體解析鏈生成的、用於內部索引的標準化名稱。")
    original_name: str = Field(description="LLM 在計畫中生成的原始 NPC 名稱。")
    description: str = Field(description="對 NPC 的詳細描述，包括其職業、性格、背景故事等。")
    location_path: List[str] = Field(description="該 NPC 所在的完整地點路徑列表。")
    status: Optional[str] = Field(default="閒置", description="NPC 當前的簡短狀態，例如：'正在巡邏'、'受傷'。")
    equipment: Optional[List[str]] = Field(default_factory=list, description="NPC 當前穿戴的裝備列表。")

# 工具：創建新 NPC 檔案 (v1.2 - 重命名與職責明確化)
@tool(args_schema=CreateNewNpcProfileArgs)
async def create_new_npc_profile(
    lore_key: str, 
    standardized_name: str, 
    original_name: str, 
    description: str, 
    location_path: List[str],
    status: Optional[str] = "閒置", 
    equipment: Optional[List[str]] = None
) -> str:
    """【只在】你需要創造一個【全新的、不存在的】NPC 時使用此工具。它會在世界中永久記錄一個新的角色檔案。"""
    user_id = tool_context.get_user_id()
    profile_data = {
        "name": standardized_name,
        "description": description,
        "location_path": location_path,
        "status": status,
        "equipment": equipment or [],
        "aliases": [original_name] if original_name.lower() != standardized_name.lower() else []
    }
    await add_or_update_lore(user_id, 'npc_profile', lore_key, profile_data)
    return f"已成功為新 NPC '{standardized_name}' 創建了檔案，主鍵為: '{lore_key}'"

# 類別：更新 NPC 檔案參數
class UpdateNpcProfileArgs(BaseToolArgs):
    lore_key: str = Field(description="要更新的 NPC 的【精確】主鍵（lore_key）。必須通過 `search_knowledge_base` 確認其存在。")
    updates: Dict[str, Any] = Field(description="一個包含要更新欄位和新值的字典。例如：{'status': '受傷', 'description': '他看起來很痛苦。'}")

# 工具：更新 NPC 檔案
@tool(args_schema=UpdateNpcProfileArgs)
async def update_npc_profile(lore_key: str, updates: Dict[str, Any]) -> str:
    """當你需要更新一個【已存在】NPC 的狀態或資訊時使用此工具。你必須提供其精確的 `lore_key`。"""
    user_id = tool_context.get_user_id()
    await add_or_update_lore(user_id, 'npc_profile', lore_key, updates, merge=True)
    npc_name = lore_key.split(' > ')[-1]
    return f"已成功更新 NPC '{npc_name}' 的檔案。"

# --- 地點相關工具 ---

# 類別：新增或更新地點資訊參數
class AddOrUpdateLocationInfoArgs(BaseToolArgs):
    lore_key: str = Field(description="系統內部使用的唯一標識符，由實體解析鏈生成。")
    standardized_name: str = Field(description="由實體解析鏈生成的、用於內部索引的標準化地點名稱。")
    original_name: str = Field(description="LLM 在計畫中生成的原始地點名稱。")
    description: str = Field(description="對該地點的詳細描述，包括其氛圍、建築風格、環境特徵等。")

# 工具：新增或更新地點資訊
@tool(args_schema=AddOrUpdateLocationInfoArgs)
async def add_or_update_location_info(lore_key: str, standardized_name: str, original_name: str, description: str) -> str:
    """用於創建一個新的地點條目，或用全新的描述覆蓋一個已有的地點條目。"""
    user_id = tool_context.get_user_id()
    location_data = {
        "name": standardized_name,
        "description": description,
        "aliases": [original_name] if original_name.lower() != standardized_name.lower() else []
    }
    await add_or_update_lore(user_id, 'location_info', lore_key, location_data)
    return f"已成功為新地點 '{standardized_name}' 記錄了資訊。"

# --- 物品相關工具 ---

# 類別：新增或更新物品資訊參數 (v2.0 - 強化外觀描述)
# 更新紀錄:
# v2.0 (2025-09-08): [功能擴展] 根據 schemas.py 的更新，新增了 `visual_description` 參數，並強化了其描述，強制要求 AI 提供詳細的外觀資訊。
# v1.2 (2025-09-02): [健壯性] 將 `add_or_update_npc_profile` 重命名為 `create_new_npc_profile`。
# v1.1 (2025-09-02): [重大架構重構] 移除了本地的上下文管理，改為從中央 `tool_context` 導入共享實例。
class AddOrUpdateItemInfoArgs(BaseToolArgs):
    lore_key: str = Field(description="系統內部使用的唯一標識符，由實體解析鏈生成。")
    standardized_name: str = Field(description="由實體解析鏈生成的、用於內部索引的標準化物品名稱。")
    original_name: str = Field(description="LLM 在計畫中生成的原始物品名稱。")
    description: str = Field(description="對物品的詳細描述，包括其材質、歷史等。")
    effect: Optional[str] = Field(default=None, description="物品的效果，例如：'恢復少量生命值'。")
    visual_description: Optional[str] = Field(default=None, description="對物品外觀的詳細、生動的描寫。")

# 工具：新增或更新物品資訊 (v2.0 - 適配新參數)
@tool(args_schema=AddOrUpdateItemInfoArgs)
async def add_or_update_item_info(lore_key: str, standardized_name: str, original_name: str, description: str, effect: Optional[str] = None, visual_description: Optional[str] = None) -> str:
    """用於創建一個新的物品條目，或用全新的描述覆蓋一個已有的物品條目。"""
    user_id = tool_context.get_user_id()
    item_data = {
        "name": standardized_name,
        "description": description,
        "effect": effect,
        "visual_description": visual_description,
        "aliases": [original_name] if original_name.lower() != standardized_name.lower() else []
    }
    await add_or_update_lore(user_id, 'item_info', lore_key, item_data)
    return f"已成功為新物品 '{standardized_name}' 記錄了詳細資訊。"
# 工具：新增或更新物品資訊 (v2.0 - 適配新參數)

# --- 生物相關工具 ---

# 類別：定義生物類型參數 (v2.0 - 強化外觀描述)
# 更新紀錄:
# v2.0 (2025-09-08): [功能擴展] 強化了 `description` 參數的描述，強制要求 AI 在描述生物時必須包含詳細的外觀資訊。
# v1.2 (2025-09-02): [健壯性] 將 `add_or_update_npc_profile` 重命名為 `create_new_npc_profile`。
# v1.1 (2025-09-02): [重大架構重構] 移除了本地的上下文管理，改為從中央 `tool_context` 導入共享實例。
class DefineCreatureTypeArgs(BaseToolArgs):
    lore_key: str = Field(description="系統內部使用的唯一標識符，由實體解析鏈生成。")
    standardized_name: str = Field(description="由實體解析鏈生成的、用於內部索引的標準化生物名稱。")
    original_name: str = Field(description="LLM 在計畫中生成的原始生物名稱。")
    description: str = Field(description="對該生物/物種的詳細描述，【必須包含】其詳細的外貌、習性、能力、棲息地等資訊。")

# 工具：定義生物類型 (v2.0 - 無功能變更)
@tool(args_schema=DefineCreatureTypeArgs)
async def define_creature_type(lore_key: str, standardized_name: str, original_name: str, description: str) -> str:
    """用於在世界百科全書中創建一個全新的生物/物種詞條。"""
    user_id = tool_context.get_user_id()
    creature_data = {
        "name": standardized_name,
        "description": description,
        "aliases": [original_name] if original_name.lower() != standardized_name.lower() else []
    }
    await add_or_update_lore(user_id, 'creature_info', lore_key, creature_data)
    return f"已成功為新物種 '{standardized_name}' 創建了百科詞條。"
# 工具：定義生物類型 (v2.0 - 無功能變更)

# --- 任務與世界傳說相關工具 ---

# 类别：新增或更新任务传说参数 (v2.0 - 增加参数别名)
# 更新纪录:
# v2.0 (2025-09-03): [健壮性] 根据背景任务的 ValidationError 日志，为 `description` 字段增加了 `content` 作为别名。此修改使得该工具能够兼容上游 LLM 可能生成的不同参数名称，从根本上解决了因参数名不匹配而导致的工具调用失败问题。
class AddOrUpdateQuestLoreArgs(BaseToolArgs):
    lore_key: str = Field(description="系统内部使用的唯一标识符，由实体解析链生成。")
    standardized_name: str = Field(description="由实体解析链生成的、用于内部索引的标准化任务标题。")
    original_name: str = Field(description="LLM 在计画中生成的原始任务标题。")
    description: str = Field(
        description="任务的详细描述，包括背景、目标、奖励等。",
        validation_alias=AliasChoices('description', 'content', 'quest_description')
    )
    location_path: List[str] = Field(description="触发或与该任务相关的地点路径。")
    status: str = Field(default="可用", description="任务的当前状态，例如：'可用'、'进行中'、'已完成'。")

# 工具：新增或更新任务传说
@tool(args_schema=AddOrUpdateQuestLoreArgs)
async def add_or_update_quest_lore(lore_key: str, standardized_name: str, original_name: str, description: str, location_path: List[str], status: str = "可用") -> str:
    """用于创建一个新的任务，或用全新的描述覆盖一个已有的任务。"""
    user_id = tool_context.get_user_id()
    quest_data = {
        "title": standardized_name,
        "description": description,
        "location_path": location_path,
        "status": status,
        "aliases": [original_name] if original_name.lower() != standardized_name.lower() else []
    }
    await add_or_update_lore(user_id, 'quest', lore_key, quest_data)
    return f"已成功为任务 '{standardized_name}' 创建或更新了记录。"

# 类别：新增或更新世界传说参数 (v2.0 - 增加参数别名)
# 更新纪录:
# v2.0 (2025-09-03): [健壮性] 为 `content` 字段增加了 `description` 作为别名。此修改使得该工具能够兼容上游 LLM 可能生成的不同参数名称，提高了背景 LORE 生成任务的成功率。
class AddOrUpdateWorldLoreArgs(BaseToolArgs):
    lore_key: str = Field(description="系统内部使用的唯一标识符，由实体解析链生成。")
    standardized_name: str = Field(description="由实体解析链生成的、用于内部索引的标准化传说标题。")
    original_name: str = Field(description="LLM 在计画中生成的原始传说标题。")
    content: str = Field(
        description="传说或背景故事的详细内容。",
        validation_alias=AliasChoices('content', 'description', 'lore_content')
    )

# 工具：新增或更新世界传说
@tool(args_schema=AddOrUpdateWorldLoreArgs)
async def add_or_update_world_lore(lore_key: str, standardized_name: str, original_name: str, content: str) -> str:
    """用于在世界历史或传说中记录一个新的故事、事件或背景设定。"""
    user_id = tool_context.get_user_id()
    lore_data = {
        "title": standardized_name,
        "content": content,
        "aliases": [original_name] if original_name.lower() != standardized_name.lower() else []
    }
    await add_or_update_lore(user_id, 'world_lore', lore_key, lore_data)
    return f"已成功将 '{standardized_name}' 识别为现有传说 '{lore_key}' 并更新了其内容。"

# --- 工具列表導出 ---

# 函式：獲取所有 LORE 工具 (v1.2 - 更新列表)
def get_lore_tools() -> List[Tool]:
    """返回一個列表，包含所有用於管理世界 LORE 的工具。"""
    return [
        create_new_npc_profile,
        update_npc_profile,
        add_or_update_location_info,
        add_or_update_item_info,
        define_creature_type,
        add_or_update_quest_lore,
        add_or_update_world_lore,
    ]
# 函式：獲取所有 LORE 工具 (v1.2 - 更新列表)

