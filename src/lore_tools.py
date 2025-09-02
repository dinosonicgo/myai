# src/lore_tools.py 的中文註釋(v24.0 - 上下文統一)
# 更新紀錄:
# v24.0 (2025-09-02): [重大架構重構] 移除了本地的 `ToolContext` 類和實例的定義。現在，此模組從新創建的中央 `tool_context.py` 導入共享的 `tool_context` 實例。此修改徹底解決了 LORE 工具因無法獲取正確上下文而執行失敗的嚴重問題。
# v23.1.1 (2025-08-31): [災難性BUG修復] 恢復了因先前提交中意外省略而丟失的所有 Pydantic 參數模型。
# v23.1 (2025-08-31): [災難性BUG修復] 在 `update_npc_profile` 工具中增加了【核心角色身份保護】檢查。

import json
import time
import re
from typing import Dict, Any, Type, List, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator, AliasChoices
from langchain_core.tools import tool

from .logger import logger
from .lore_book import add_or_update_lore as db_add_or_update_lore, get_lore as db_get_lore, get_lores_by_category_and_filter
from .database import AsyncSessionLocal, UserData
from .schemas import CharacterProfile, LocationInfo, ItemInfo, WorldLore, Quest, CreatureInfo
# [v24.0 新增] 導入共享的工具上下文
from .tool_context import tool_context

# [v24.0 移除] 刪除了本地的 ToolContext 類和實例的定義

# --- 通用驗證器 ---
def _validate_string_to_list(value: Any) -> Any:
    if isinstance(value, str):
        items = re.split(r'[，,、;\n]', value)
        return [item.strip() for item in items if item and item.strip()]
    if isinstance(value, list):
        processed_list = []
        for item in value:
            if isinstance(item, dict) and 'name' in item:
                processed_list.append(item['name'])
            elif isinstance(item, str):
                processed_list.append(item)
        return processed_list
    return value

def _validate_string_to_dict(value: Any) -> Any:
    if isinstance(value, str):
        if value.strip().lower() in ["無", "未知", "", "none", "null"]:
            return {}
        try:
            return json.loads(value.replace("'", '"'))
        except json.JSONDecodeError:
            return {"summary": value}
    return value

# --- Pydantic 基礎模型 ---
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

# --- 針對特定 Lore 類別的結構化新增/更新工具 ---

class AddNpcLoreArgs(BaseToolArgs):
    lore_key: str = Field(description="由實體解析鏈生成的、此 NPC 的絕對唯一資料庫主鍵。")
    standardized_name: str = Field(
        description="由實體解析鏈生成的、此 NPC 的標準化名稱。",
        validation_alias=AliasChoices('standardized_name', 'name', 'npc_name')
    )
    original_name: str = Field(description="LLM 最初生成的原始名稱，將作為別名被記錄。")
    location_path: List[str] = Field(description="【必需】NPC 所在的層級式地點路徑。")
    gender: str = Field(default="未知", description="NPC 的性別。")
    race: str = Field(default="未知", description="NPC 的種族。")
    age: str = Field(default="未知", description="NPC 的年齡或年齡段。")
    appearance: str = Field(default="", description="NPC 的外貌特徵的總體描述。")
    appearance_details: Dict[str, str] = Field(default_factory=dict, description="NPC 的具體外貌細節。")
    description: str = Field(default="", description="NPC 的性格、背景故事、職責等。")
    affinity: int = Field(default=0, description="此 NPC 對使用者的好感度。")
    relationships: Dict[str, str] = Field(default_factory=dict, description="記錄此 NPC 與其他角色的關係。")
    equipment: List[str] = Field(default_factory=list, description="NPC 當前穿戴或持有的裝備列表。")
    skills: List[str] = Field(default_factory=list, description="NPC 掌握的技能列表。")
    status: str = Field(default="健康", description="NPC 的當前健康或狀態。")
    
    @field_validator('location_path', 'equipment', 'skills', mode='before')
    @classmethod
    def _string_to_list(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

    @field_validator('appearance_details', 'relationships', mode='before')
    @classmethod
    def _string_to_dict(cls, value: Any) -> Any:
        return _validate_string_to_dict(value)

    @field_validator('age', mode='before')
    @classmethod
    def _validate_age(cls, value: Any) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        return value

@tool(args_schema=AddNpcLoreArgs)
async def add_or_update_npc_profile(
    lore_key: str,
    standardized_name: str,
    original_name: str,
    location_path: List[str],
    **kwargs: Any
) -> str:
    """
    接收預先解析過的數據，創建或更新 NPC 的詳細檔案。
    """
    user_id = tool_context.get_user_id()
    
    if 'npc_profile' in kwargs and isinstance(kwargs['npc_profile'], dict):
        profile_data = kwargs['npc_profile']
        standardized_name = profile_data.get('name', standardized_name)
        location_path = profile_data.get('location_path', location_path)
        kwargs = profile_data
    
    existing_lore = await db_get_lore(user_id, 'npc_profile', lore_key)

    if existing_lore:
        content_model = CharacterProfile.model_validate(existing_lore.content)
        update_data = {k: v for k, v in kwargs.items() if v is not None}
        updated_model = content_model.model_copy(update=update_data)
        if original_name and original_name not in updated_model.aliases and original_name != updated_model.name:
            updated_model.aliases.append(original_name)
        content_model = updated_model
        message = f"已成功將 '{original_name}' 識別為現有 NPC '{content_model.name}' 並更新了其檔案。"
    else:
        content_model = CharacterProfile(
            name=standardized_name,
            aliases=[original_name] if original_name and original_name != standardized_name else [],
            location=location_path[-1] if location_path else "未知",
            **kwargs
        )
        message = f"已成功為新 NPC '{standardized_name}' 創建了檔案。"

    content = content_model.model_dump()
    content['location_path'] = location_path

    await db_add_or_update_lore(user_id, 'npc_profile', lore_key, content)
    logger.info(f"[{user_id}] 已新增/更新 NPC Lore，主鍵為: '{lore_key}'")
    return message

class UpdateNpcArgs(BaseToolArgs):
    lore_key: str = Field(description="要更新的 NPC 的絕對唯一資料庫主鍵。")
    description: Optional[str] = Field(default=None, description="NPC 的新版性格、背景故事或職責描述。")
    status: Optional[str] = Field(default=None, description="NPC 的新健康或狀態，例如：'重傷', '死亡'。")
    relationships: Optional[Dict[str, str]] = Field(default=None, description="要新增或更新的關係字典，例如：{'另一個角色': '盟友'}。")
    affinity: Optional[int] = Field(default=None, description="NPC 對使用者的新好感度數值。")
    current_action: Optional[str] = Field(default=None, description="NPC 當前正在進行的、持續性的動作或所處的姿態。")

@tool(args_schema=UpdateNpcArgs)
async def update_npc_profile(
    lore_key: str, 
    **kwargs: Any
) -> str:
    """[背景/核心專用] 更新一個已存在的NPC的檔案，用於記錄狀態變化（如健康狀況、關係、當前動作）。"""
    user_id = tool_context.get_user_id()
    ai_core = tool_context.get_ai_core()

    if ai_core.profile:
        npc_name_from_key = lore_key.split(' > ')[-1]
        user_name = ai_core.profile.user_profile.name
        ai_name = ai_core.profile.ai_profile.name
        if npc_name_from_key.lower() in [user_name.lower(), ai_name.lower()]:
            logger.warning(f"[{user_id}] 【身份保護觸發】: 檢測到一個試圖使用 NPC 更新工具來修改核心角色 '{npc_name_from_key}' 的無效請求。操作已被阻止。")
            return f"錯誤：拒絕操作。'{npc_name_from_key}' 是一個核心主角，不能被當作 NPC 進行修改。"

    existing_lore = await db_get_lore(user_id, 'npc_profile', lore_key)

    if not existing_lore:
        all_npcs = await get_lores_by_category_and_filter(user_id, 'npc_profile')
        found = False
        for npc in all_npcs:
            if lore_key in npc.key:
                existing_lore = npc
                lore_key = npc.key
                logger.warning(f"[{user_id}] 在 update_npc_profile 中找不到精確的 key，但成功模糊匹配到: '{lore_key}'")
                found = True
                break
        if not found:
            return f"錯誤：在資料庫中找不到主鍵為 '{lore_key}' 的 NPC 可供更新。"

    try:
        content_model = CharacterProfile.model_validate(existing_lore.content)
        
        updates = {key: value for key, value in kwargs.items() if value is not None}
        
        if not updates:
            return f"提醒：為 NPC '{content_model.name}' 呼叫了更新工具，但沒有提供任何要更新的欄位。"

        for key, value in updates.items():
            if hasattr(content_model, key):
                current_value = getattr(content_model, key)
                if isinstance(current_value, dict) and isinstance(value, dict):
                    current_value.update(value)
                elif isinstance(current_value, list) and isinstance(value, list):
                    current_value.extend(item for item in value if item not in current_value)
                else:
                    setattr(content_model, key, value)

        await db_add_or_update_lore(user_id, 'npc_profile', lore_key, content_model.model_dump())
        message = f"已成功更新 NPC '{content_model.name}' 的檔案，更新的欄位: {', '.join(updates.keys())}。"
        logger.info(f"[{user_id}] {message}")
        return message
    except Exception as e:
        logger.error(f"[{user_id}] 更新NPC '{lore_key}' 時發生錯誤: {e}", exc_info=True)
        return f"錯誤：更新NPC '{lore_key}' 時發生內部錯誤。"

class AddLocationLoreArgs(BaseToolArgs):
    lore_key: str = Field(description="由實體解析鏈生成的、此地點的絕對唯一資料庫主鍵。")
    standardized_name: str = Field(
        description="由實體解析鏈生成的、此地點的標準化名稱。",
        validation_alias=AliasChoices('standardized_name', 'name', 'location_name')
    )
    original_name: str = Field(description="LLM 最初生成的原始名稱，將作為別名被記錄。")
    description: str = Field(default="", description="對該地點的詳細描述。")
    notable_features: List[str] = Field(default_factory=list, description="該地點的顯著特徵或地標列表。")
    known_npcs: List[str] = Field(default_factory=list, description="已知居住或出現在此地點的 NPC 名字列表。")

    @field_validator('notable_features', 'known_npcs', mode='before')
    @classmethod
    def _string_to_list(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

@tool(args_schema=AddLocationLoreArgs)
async def add_or_update_location_info(
    lore_key: str,
    standardized_name: str,
    original_name: str,
    **kwargs: Any
) -> str:
    """接收預先解析過的數據，創建或更新地點的資訊。此工具應由系統協調器呼叫。"""
    user_id = tool_context.get_user_id()
    existing_lore = await db_get_lore(user_id, 'location_info', lore_key)

    if existing_lore:
        content_model = LocationInfo.model_validate(existing_lore.content)
        if original_name not in content_model.aliases and original_name != content_model.name:
            content_model.aliases.append(original_name)
        if kwargs.get('description'):
            content_model.description = kwargs['description']
        message = f"已成功將 '{original_name}' 識別為現有地點 '{content_model.name}' 並更新了其資訊。"
    else:
        content_model = LocationInfo(
            name=standardized_name,
            aliases=[original_name] if original_name != standardized_name else [],
            **kwargs
        )
        message = f"已成功為新地點 '{standardized_name}' 記錄了資訊。"

    await db_add_or_update_lore(user_id, 'location_info', lore_key, content_model.model_dump())
    logger.info(f"[{user_id}] 已新增/更新 Location Lore，主鍵為: '{lore_key}'")
    return message

class AddItemLoreArgs(BaseToolArgs):
    lore_key: str = Field(description="由實體解析鏈生成的、此物品的絕對唯一資料庫主鍵。")
    standardized_name: str = Field(
        description="由實體解析鏈生成的、此物品的標準化名稱。",
        validation_alias=AliasChoices('standardized_name', 'name', 'item_name')
    )
    original_name: str = Field(description="LLM 最初生成的原始名稱，將作為別名被記錄。")
    description: str = Field(default="", description="對該道具的詳細描述。")
    item_type: str = Field(default="未知", description="道具的類型。")
    effect: str = Field(default="無", description="道具的效果描述。")
    rarity: str = Field(default="普通", description="道具的稀有度。")
    visual_description: Optional[str] = Field(default="", description="對道具外觀的詳細描寫。")
    origin: Optional[str] = Field(default="", description="關於該道具來源的簡短傳說。")

@tool(args_schema=AddItemLoreArgs)
async def add_or_update_item_info(
    lore_key: str,
    standardized_name: str,
    original_name: str,
    **kwargs: Any
) -> str:
    """接收預先解析過的數據，創建或更新物品的詳細資訊。此工具應由系統協調器呼叫。"""
    user_id = tool_context.get_user_id()
    existing_lore = await db_get_lore(user_id, 'item_info', lore_key)

    if existing_lore:
        content_model = ItemInfo.model_validate(existing_lore.content)
        if original_name not in content_model.aliases and original_name != content_model.name:
            content_model.aliases.append(original_name)
        if kwargs.get('description'):
            content_model.description = kwargs['description']
        message = f"已成功將 '{original_name}' 識別為現有物品 '{content_model.name}' 並更新了其資訊。"
    else:
        kwargs['visual_description'] = kwargs.get('visual_description') or ""
        kwargs['origin'] = kwargs.get('origin') or ""
        content_model = ItemInfo(
            name=standardized_name,
            aliases=[original_name] if original_name != standardized_name else [],
            **kwargs
        )
        message = f"已成功為新物品 '{standardized_name}' 記錄了詳細資訊。"

    await db_add_or_update_lore(user_id, 'item_info', lore_key, content_model.model_dump())
    logger.info(f"[{user_id}] 已新增/更新 Item Lore: '{lore_key}'")
    return message

class DefineCreatureTypeArgs(BaseToolArgs):
    lore_key: str = Field(description="由實體解析鏈生成的、此生物的絕對唯一資料庫主鍵。")
    standardized_name: str = Field(
        description="由實體解析鏈生成的、此生物的標準化名稱。",
        validation_alias=AliasChoices('standardized_name', 'name', 'creature_name')
    )
    original_name: str = Field(description="LLM 最初生成的原始名稱，將作為別名被記錄。")
    description: str = Field(default="", description="對該生物/魔物的詳細描述。")
    abilities: List[str] = Field(default_factory=list, description="該生物/魔物的特殊能力列表。")
    habitat: List[str] = Field(default_factory=list, description="該生物/魔物的主要棲息地列表。")

    @field_validator('abilities', 'habitat', mode='before')
    @classmethod
    def _string_to_list(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

@tool(args_schema=DefineCreatureTypeArgs)
async def define_creature_type(
    lore_key: str,
    standardized_name: str,
    original_name: str,
    **kwargs: Any
) -> str:
    """接收預先解析過的數據，為一個生物、魔物或物種創建一個權威的「物種詞條」。此工具應由系統協調器呼叫。"""
    user_id = tool_context.get_user_id()
    existing_lore = await db_get_lore(user_id, 'creature_info', lore_key)

    if existing_lore:
        content_model = CreatureInfo.model_validate(existing_lore.content)
        if original_name not in content_model.aliases and original_name != content_model.name:
            content_model.aliases.append(original_name)
        if kwargs.get('description'):
            content_model.description = kwargs['description']
        message = f"已成功將 '{original_name}' 識別為現有物種 '{content_model.name}' 並更新了其百科詞條。"
    else:
        content_model = CreatureInfo(
            name=standardized_name,
            aliases=[original_name] if original_name != standardized_name else [],
            **kwargs
        )
        message = f"已成功為新物種 '{standardized_name}' 創建了百科詞條。"

    await db_add_or_update_lore(user_id, 'creature_info', lore_key, content_model.model_dump())
    logger.info(f"[{user_id}] 已定義新的生物類型 Lore: '{lore_key}'")
    return message

class AddQuestLoreArgs(BaseToolArgs):
    lore_key: str = Field(description="由實體解析鏈生成的、此任務的絕對唯一資料庫主鍵。")
    standardized_name: str = Field(
        description="由實體解析鏈生成的、此任務的標準化名稱。",
        validation_alias=AliasChoices('standardized_name', 'name', 'quest_name')
    )
    original_name: str = Field(description="LLM 最初生成的原始名稱，將作為別名被記錄。")
    location_path: List[str] = Field(description="【必需】任務發布地點的層級式路徑。")
    description: str = Field(default="", description="任務的詳細描述和目標。")
    status: str = Field(default="active", description="任務的當前狀態。")
    quest_giver: Optional[str] = Field(default=None, description="此任務的發布者（NPC名字）。")
    suggested_level: Optional[str] = Field(default=None, description="建議執行此任務的角色等級。")
    rewards: Dict[str, Any] = Field(default_factory=dict, description="完成任務的獎勵。")

    @field_validator('location_path', mode='before')
    @classmethod
    def _string_to_list(cls, value: Any) -> Any:
        return _validate_string_to_list(value)
        
    @field_validator('rewards', mode='before')
    @classmethod
    def _string_to_dict(cls, value: Any) -> Any:
        return _validate_string_to_dict(value)

@tool(args_schema=AddQuestLoreArgs)
async def add_or_update_quest_lore(
    lore_key: str,
    standardized_name: str,
    original_name: str,
    **kwargs: Any
) -> str:
    """接收預先解析過的數據，記錄或更新任務資訊。此工具應由系統協調器呼叫。"""
    user_id = tool_context.get_user_id()
    existing_lore = await db_get_lore(user_id, 'quest', lore_key)

    if existing_lore:
        content_model = Quest.model_validate(existing_lore.content)
        if original_name not in content_model.aliases and original_name != content_model.name:
            content_model.aliases.append(original_name)
        if kwargs.get('description'):
            content_model.description = kwargs['description']
        message = f"已成功將 '{original_name}' 識別為現有任務 '{content_model.name}' 並更新了其資訊。"
    else:
        content_model = Quest(
            name=standardized_name,
            aliases=[original_name] if original_name != standardized_name else [],
            **kwargs
        )
        message = f"已成功為新任務 '{standardized_name}' 記錄了詳細資訊。"

    await db_add_or_update_lore(user_id, 'quest', lore_key, content_model.model_dump())
    logger.info(f"[{user_id}] 已新增/更新 Quest Lore，主鍵為: '{lore_key}'")
    return message

class AddWorldLoreArgs(BaseToolArgs):
    lore_key: str = Field(description="由實體解析鏈生成的、此傳說的絕對唯一資料庫主鍵。")
    standardized_name: str = Field(
        description="由實體解析鏈生成的、此傳說的標準化標題。",
        validation_alias=AliasChoices('standardized_name', 'title', 'name', 'lore_name')
    )
    original_name: str = Field(description="LLM 最初生成的原始名稱，將作為別名被記錄。")
    content: str = Field(default="", description="詳細的內容描述。")
    category: str = Field(default="未知", description="Lore 的分類。")
    key_elements: List[str] = Field(default_factory=list, description="與此 Lore 相關的關鍵詞列表。")
    related_entities: List[str] = Field(default_factory=list, description="與此 Lore 相關的實體名稱列表。")

    @field_validator('key_elements', 'related_entities', mode='before')
    @classmethod
    def _string_to_list(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

@tool(args_schema=AddWorldLoreArgs)
async def add_or_update_world_lore(
    lore_key: str,
    standardized_name: str,
    original_name: str,
    **kwargs: Any
) -> str:
    """接收預先解析過的數據，記錄世界背景、神話傳說等。此工具應由系統協調器呼叫。"""
    user_id = tool_context.get_user_id()
    existing_lore = await db_get_lore(user_id, 'world_lore', lore_key)

    if existing_lore:
        content_model = WorldLore.model_validate(existing_lore.content)
        if original_name not in content_model.aliases and original_name != content_model.title:
            content_model.aliases.append(original_name)
        if kwargs.get('content'):
            content_model.content = kwargs['content']
        message = f"已成功將 '{original_name}' 識別為現有傳說 '{content_model.title}' 並更新了其內容。"
    else:
        content_model = WorldLore(
            title=standardized_name,
            aliases=[original_name] if original_name != standardized_name else [],
            **kwargs
        )
        message = f"已成功記錄了關於 '{standardized_name}' 的世界傳說。"

    await db_add_or_update_lore(user_id, 'world_lore', lore_key, content_model.model_dump())
    logger.info(f"[{user_id}] 已新增/更新 World Lore: '{lore_key}'")
    return message

class UpdateQuestStatusArgs(BaseToolArgs):
    lore_key: str = Field(description="由實體解析鏈生成的、此任務的絕對唯一資料庫主鍵。")
    new_status: str = Field(description="要更新到的新狀態，例如 'completed', 'failed', 或描述下一個目標的文字。")

@tool(args_schema=UpdateQuestStatusArgs)
async def update_quest_status(
    lore_key: str,
    new_status: str,
) -> str:
    """更新現有任務的狀態。"""
    user_id = tool_context.get_user_id()
    existing_lore = await db_get_lore(user_id, 'quest', lore_key)

    if not existing_lore:
        return f"錯誤：找不到主鍵為 '{lore_key}' 的任務。"

    content_model = Quest.model_validate(existing_lore.content)
    old_status = content_model.status
    content_model.status = new_status
    
    await db_add_or_update_lore(user_id, 'quest', lore_key, content_model.model_dump())
    
    message = f"已成功將任務 '{content_model.name}' 的狀態從 '{old_status}' 更新為 '{new_status}'。"
    logger.info(f"[{user_id}] {message}")
    return message

# 函式：獲取所有 LORE 工具
def get_lore_tools():
    """返回一個包含此檔案中所有用於「寫入」的 LoreBook 相關工具的列表。"""
    return [
        add_or_update_npc_profile,
        update_npc_profile,
        add_or_update_location_info,
        add_or_update_item_info,
        define_creature_type,
        add_or_update_quest_lore,
        add_or_update_world_lore,
        update_quest_status,
    ]
# 函式：獲取所有 LORE 工具
