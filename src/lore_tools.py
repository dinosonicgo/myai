# src/lore_tools.py 的中文註釋(v3.4 - RAG增量更新)
# 更新紀錄:
# v3.4 (2025-09-23): [根本性重構] 根據“RAG增量更新”架構，在每個成功修改資料庫的工具末尾，都增加了對 `ai_core._update_rag_for_single_lore` 的調用。此修改將RAG索引更新的觸發器從一個集中的、全量的位置，下放到了各個具體的、增量的工具中，是實現高效RAG的關鍵一步。
# v3.3 (2025-11-22): [根本性重構] 根據「持久化 RAG」架構，移除了此檔案中所有對 ai_core.add_lore_to_rag 的異步任務調用。
# v3.2 (2025-11-22): [災難性BUG修復] 對所有 Pydantic 參數模型進行了全面的健壯性升級。

import time
import re
import asyncio
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

class CreateNewNpcProfileArgs(BaseToolArgs):
    lore_key: str = Field(description="系統內部使用的唯一標識符，由實體解析鏈生成。")
    standardized_name: Optional[str] = Field(default=None, validation_alias=AliasChoices('standardized_name', 'name'), description="標準化名稱。")
    original_name: Optional[str] = Field(default=None, description="原始 NPC 名稱。")
    description: str = Field(description="對 NPC 的詳細描述。")
    location_path: Optional[List[str]] = Field(default_factory=list, description="該 NPC 所在的完整地點路徑列表。如果未知，則留空。")
    status: Optional[str] = Field(default="閒置", description="NPC 當前的簡短狀態。")
    equipment: Optional[List[str]] = Field(default_factory=list, description="NPC 當前穿戴的裝備列表。")

    @field_validator('standardized_name', 'original_name', mode='before')
    @classmethod
    def default_names(cls, v, info):
        if not v: return info.data.get('lore_key', '').split(' > ')[-1]
        return v

    @field_validator('location_path', mode='before')
    @classmethod
    def validate_location_path(cls, v):
        if isinstance(v, str):
            if v.strip(): return [v.strip()]
            return []
        return v

@tool(args_schema=CreateNewNpcProfileArgs)
async def create_new_npc_profile(lore_key: str, standardized_name: str, original_name: str, description: str, location_path: Optional[List[str]] = None, status: Optional[str] = "閒置", equipment: Optional[List[str]] = None) -> str:
    """【只在】你需要創造一個【全新的、不存在的】NPC 時使用此工具。它會在世界中永久記錄一個新的角色檔案。"""
    user_id = tool_context.get_user_id()
    ai_core = tool_context.get_ai_core()
    final_location_path = location_path if location_path is not None else []
    profile_data = {
        "name": standardized_name, "description": description, "location_path": final_location_path, 
        "status": status, "equipment": equipment or [], 
        "aliases": [original_name] if original_name and original_name.lower() != standardized_name.lower() else []
    }
    lore_entry = await add_or_update_lore(user_id, 'npc_profile', lore_key, profile_data)
    # [v3.4 核心修正] 觸發RAG增量更新
    if ai_core:
        await ai_core._update_rag_for_single_lore(lore_entry)
    location_str = " > ".join(final_location_path) if final_location_path else "未知地點"
    return f"已成功為新 NPC '{standardized_name}' 創建了檔案 (主鍵: '{lore_key}')，地點: '{location_str}'。"

class UpdateNpcProfileArgs(BaseToolArgs):
    lore_key: str = Field(description="要更新的 NPC 的【精確】主鍵（lore_key）。")
    updates: Dict[str, Any] = Field(description="一個包含要更新欄位和新值的字典。")

@tool(args_schema=UpdateNpcProfileArgs)
async def update_npc_profile(lore_key: str, updates: Dict[str, Any]) -> str:
    """當你需要更新一個【已存在】NPC 的狀態或資訊時使用此工具。你必須提供其精確的 `lore_key`。"""
    user_id = tool_context.get_user_id()
    ai_core = tool_context.get_ai_core()
    updated_lore = await add_or_update_lore(user_id, 'npc_profile', lore_key, updates, merge=True)
    # [v3.4 核心修正] 觸發RAG增量更新
    if ai_core:
        await ai_core._update_rag_for_single_lore(updated_lore)
    npc_name = updated_lore.content.get('name', lore_key.split(' > ')[-1])
    return f"已成功更新 NPC '{npc_name}' 的檔案。"

# --- 地點相關工具 ---

class AddOrUpdateLocationInfoArgs(BaseToolArgs):
    lore_key: str = Field(description="系統內部使用的唯一標識符。")
    standardized_name: Optional[str] = Field(default=None, validation_alias=AliasChoices('standardized_name', 'name'), description="標準化地點名稱。")
    original_name: Optional[str] = Field(default=None, description="原始地點名稱。")
    description: str = Field(description="對該地點的詳細描述。")

    @field_validator('standardized_name', 'original_name', mode='before')
    @classmethod
    def default_names(cls, v, info):
        if not v: return info.data.get('lore_key', '').split(' > ')[-1]
        return v

@tool(args_schema=AddOrUpdateLocationInfoArgs)
async def add_or_update_location_info(lore_key: str, standardized_name: str, original_name: str, description: str) -> str:
    """用於創建一個新的地點條目，或用全新的描述覆蓋一個已有的地點條目。"""
    user_id = tool_context.get_user_id()
    ai_core = tool_context.get_ai_core()
    location_data = {"name": standardized_name, "description": description, "aliases": [original_name] if original_name and original_name.lower() != standardized_name.lower() else []}
    lore_entry = await add_or_update_lore(user_id, 'location_info', lore_key, location_data)
    # [v3.4 核心修正] 觸發RAG增量更新
    if ai_core:
        await ai_core._update_rag_for_single_lore(lore_entry)
    return f"已成功為地點 '{standardized_name}' 記錄了資訊。"

# --- 物品相關工具 ---

class AddOrUpdateItemInfoArgs(BaseToolArgs):
    lore_key: str = Field(description="系統內部使用的唯一標識符。")
    standardized_name: Optional[str] = Field(default=None, validation_alias=AliasChoices('standardized_name', 'name'), description="標準化物品名稱。")
    original_name: Optional[str] = Field(default=None, description="原始物品名稱。")
    description: str = Field(description="對物品的詳細描述。")
    effect: Optional[str] = Field(default=None, description="物品的效果。")
    visual_description: Optional[str] = Field(default=None, description="對物品外觀的詳細描寫。")

    @field_validator('standardized_name', 'original_name', mode='before')
    @classmethod
    def default_names(cls, v, info):
        if not v: return info.data.get('lore_key', '').split(' > ')[-1]
        return v

@tool(args_schema=AddOrUpdateItemInfoArgs)
async def add_or_update_item_info(lore_key: str, standardized_name: str, original_name: str, description: str, effect: Optional[str] = None, visual_description: Optional[str] = None) -> str:
    """用於創建一個新的物品條目，或用全新的描述覆蓋一個已有的物品條目。"""
    user_id = tool_context.get_user_id()
    ai_core = tool_context.get_ai_core()
    item_data = {"name": standardized_name, "description": description, "effect": effect, "visual_description": visual_description, "aliases": [original_name] if original_name and original_name.lower() != standardized_name.lower() else []}
    lore_entry = await add_or_update_lore(user_id, 'item_info', lore_key, item_data)
    # [v3.4 核心修正] 觸發RAG增量更新
    if ai_core:
        await ai_core._update_rag_for_single_lore(lore_entry)
    return f"已成功為物品 '{standardized_name}' 記錄了詳細資訊。"

# --- 生物相關工具 ---

class DefineCreatureTypeArgs(BaseToolArgs):
    lore_key: str = Field(description="系統內部使用的唯一標識符。")
    standardized_name: Optional[str] = Field(default=None, validation_alias=AliasChoices('standardized_name', 'name'), description="標準化生物名稱。")
    original_name: Optional[str] = Field(default=None, description="原始生物名稱。")
    description: str = Field(description="對該生物/物種的詳細描述。")

    @field_validator('standardized_name', 'original_name', mode='before')
    @classmethod
    def default_names(cls, v, info):
        if not v: return info.data.get('lore_key', '').split(' > ')[-1]
        return v

@tool(args_schema=DefineCreatureTypeArgs)
async def define_creature_type(lore_key: str, standardized_name: str, original_name: str, description: str) -> str:
    """用於在世界百科全書中創建一個全新的生物/物種詞條。"""
    user_id = tool_context.get_user_id()
    ai_core = tool_context.get_ai_core()
    creature_data = {"name": standardized_name, "description": description, "aliases": [original_name] if original_name and original_name.lower() != standardized_name.lower() else []}
    lore_entry = await add_or_update_lore(user_id, 'creature_info', lore_key, creature_data)
    # [v3.4 核心修正] 觸發RAG增量更新
    if ai_core:
        await ai_core._update_rag_for_single_lore(lore_entry)
    return f"已成功為物種 '{standardized_name}' 創建了百科詞條。"

# --- 任務與世界傳說相關工具 ---

class AddOrUpdateQuestLoreArgs(BaseToolArgs):
    lore_key: str = Field(description="系統內部使用的唯一標識符。")
    standardized_name: Optional[str] = Field(default=None, validation_alias=AliasChoices('standardized_name', 'title', 'name'), description="標準化任務標題。")
    original_name: Optional[str] = Field(default=None, description="原始任務標題。")
    description: str = Field(validation_alias=AliasChoices('description', 'content', 'quest_description'), description="任務的詳細描述。")
    location_path: List[str] = Field(description="觸發或與該任務相關的地點路徑。")
    status: str = Field(default="可用", description="任務的當前狀態。")

    @field_validator('standardized_name', 'original_name', mode='before')
    @classmethod
    def default_names(cls, v, info):
        if not v: return info.data.get('lore_key', '').split(' > ')[-1]
        return v

@tool(args_schema=AddOrUpdateQuestLoreArgs)
async def add_or_update_quest_lore(lore_key: str, standardized_name: str, original_name: str, description: str, location_path: List[str], status: str = "可用") -> str:
    """用於創建一個新的任務，或用全新的描述覆蓋一個已有的任務。"""
    user_id = tool_context.get_user_id()
    ai_core = tool_context.get_ai_core()
    quest_data = {"title": standardized_name, "description": description, "location_path": location_path, "status": status, "aliases": [original_name] if original_name and original_name.lower() != standardized_name.lower() else []}
    lore_entry = await add_or_update_lore(user_id, 'quest', lore_key, quest_data)
    # [v3.4 核心修正] 觸發RAG增量更新
    if ai_core:
        await ai_core._update_rag_for_single_lore(lore_entry)
    return f"已成功為任務 '{standardized_name}' 創建或更新了記錄。"

class AddOrUpdateWorldLoreArgs(BaseToolArgs):
    lore_key: str = Field(description="系統內部使用的唯一標識符。")
    standardized_name: Optional[str] = Field(default=None, validation_alias=AliasChoices('standardized_name', 'title', 'name'), description="標準化傳說標題。")
    original_name: Optional[str] = Field(default=None, description="原始傳說標題。")
    content: str = Field(validation_alias=AliasChoices('content', 'description', 'lore_content'), description="傳說或背景故事的詳細內容。")

    @field_validator('standardized_name', 'original_name', mode='before')
    @classmethod
    def default_names(cls, v, info):
        if not v: return info.data.get('lore_key', '').split(' > ')[-1]
        return v

@tool(args_schema=AddOrUpdateWorldLoreArgs)
async def add_or_update_world_lore(lore_key: str, standardized_name: str, original_name: str, content: str) -> str:
    """用於在世界歷史或傳說中記錄一個新的故事、事件或背景設定。"""
    user_id = tool_context.get_user_id()
    ai_core = tool_context.get_ai_core()
    lore_data = {"title": standardized_name, "content": content, "aliases": [original_name] if original_name and original_name.lower() != standardized_name.lower() else []}
    lore_entry = await add_or_update_lore(user_id, 'world_lore', lore_key, lore_data)
    # [v3.4 核心修正] 觸發RAG增量更新
    if ai_core:
        await ai_core._update_rag_for_single_lore(lore_entry)
    return f"已成功將 '{standardized_name}' 記錄為傳說。"

# --- 工具列表導出 ---

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
    ]        add_or_update_world_lore,
    ]

