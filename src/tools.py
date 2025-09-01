# src/tools.py 的中文註釋(v16.0 - 路徑邏輯修正)
# 更新紀錄:
# v16.0 (2025-08-27):
# 1. [根本性BUG修復] 徹底重構了 `change_location` 的路徑處理邏輯，明確區分絕對路徑 (以'/'開頭) 和相對路徑，並正確處理 '..' 返回上一級的操作。此修正從根本上解決了地點路徑被錯誤拼接導致NPC重複創建的嚴重BUG。
# v15.3 (2025-08-15):
# 1. [健壯性] 根據 v4.2 藍圖，為所有工具的 Pydantic 參數模型增加了 `AliasChoices`，以容錯 LLM 可能幻覺出的不同參數名。
# v15.2 (2025-08-15):
# 1. [穩定性回歸] 恢復了在所有工具函式上使用 `@tool` 裝飾器的標準 LangChain 模式，解決了 `ToolException`。

import asyncio
import json
from typing import Dict, Any, Type, Optional, List, Callable, Tuple
from pydantic import BaseModel, Field, ConfigDict, field_validator, AliasChoices
from langchain_core.tools import tool, Tool
from langchain_core.documents import Document
from sqlalchemy.exc import SQLAlchemyError

from .logger import logger
from . import lore_book
from . import lore_tools
from .models import CharacterProfile
from .database import AsyncSessionLocal, UserData
from .models import GameState

# --- Pydantic 模型與配置 ---
# 函式：從 schema 中移除 title
# 說明：一個輔助函式，用於在生成工具的 JSON schema 時移除 Pydantic 自動生成的 'title' 欄位，以簡化傳給 LLM 的 schema。
def remove_title_from_schema(schema: Dict[str, Any], model: Type['BaseModel']) -> None:
    if 'title' in schema:
        del schema['title']
    for prop in schema.get('properties', {}).values():
        if 'title' in prop:
            del prop['title']
# 函式：從 schema 中移除 title

# 類別：基礎工具參數
# 說明：所有工具參數模型的基類，應用了移除 title 的 schema 修改。
class BaseToolArgs(BaseModel):
    model_config = ConfigDict(
        json_schema_extra=remove_title_from_schema
    )
# 類別：基礎工具參數

# --- 工具上下文管理器 ---
# 類別：工具上下文
# 說明：一個全域上下文管理器，用於在工具執行期間儲存和傳遞使用者ID和AI核心實例，避免在每個工具函式中重複傳遞。
class ToolContext:
    def __init__(self):
        self.user_id = None
        self.ai_core_instance = None
    def set_context(self, user_id: str, ai_core_instance: 'AILover'):
        self.user_id = user_id
        self.ai_core_instance = ai_core_instance
    def get_user_id(self) -> str:
        if not self.user_id:
            raise ValueError("Tool context user_id is not set.")
        return self.user_id
    def get_ai_core(self) -> 'AILover':
        if not self.ai_core_instance:
            raise ValueError("Tool context ai_core_instance is not set.")
        return self.ai_core_instance
# 類別：工具上下文

tool_context = ToolContext()

# --- 異步資料庫操作輔助函式 ---
# 函式：獲取並更新角色檔案
# 說明：一個核心輔助函式，用於原子化地處理角色檔案的讀取、修改和寫回操作，支持使用者、AI和NPC。
async def _get_and_update_character_profile(
    character_name: str, 
    update_logic: Callable[[CharacterProfile, GameState], str]
) -> str:
    user_id = tool_context.get_user_id()
    ai_core = tool_context.get_ai_core()
    
    if not ai_core.profile:
        return f"錯誤：無法獲取當前使用者設定檔。"

    current_profile = ai_core.profile 

    try:
        gs = current_profile.game_state
        
        target_profile_pydantic: Optional[CharacterProfile] = None
        is_npc = False
        npc_key: Optional[str] = None

        user_profile_pydantic = current_profile.user_profile
        ai_profile_pydantic = current_profile.ai_profile

        if character_name.lower() == user_profile_pydantic.name.lower():
            target_profile_pydantic = user_profile_pydantic
        elif character_name.lower() == ai_profile_pydantic.name.lower():
            target_profile_pydantic = ai_profile_pydantic
        else:
            logger.info(f"[{user_id}] 正在為更新操作解析 NPC 實體: '{character_name}'...")
            # 使用實體解析鏈來準確地找到NPC
            resolution_chain = ai_core.get_batch_entity_resolution_chain()
            existing_lores = await lore_book.get_lores_by_category_and_filter(user_id, 'npc_profile')
            existing_entities_for_prompt = [{"key": lore.key, "name": lore.content.get("name", "")} for lore in existing_lores]
            
            resolution_plan = await ai_core.ainvoke_with_rotation(resolution_chain, {
                "username": ai_core.profile.user_profile.name, "ai_name": ai_core.profile.ai_profile.name,
                "category": "npc_profile",
                "new_entities_json": json.dumps([{"name": character_name, "location_path": gs.location_path}], ensure_ascii=False),
                "existing_entities_json": json.dumps(existing_entities_for_prompt, ensure_ascii=False)
            })

            if not resolution_plan or not resolution_plan.resolutions:
                 return f"錯誤：在當前場景中找不到名為 '{character_name}' 的 NPC 檔案可供更新。"

            resolution = resolution_plan.resolutions[0]
            if resolution.decision == 'NEW' or not resolution.matched_key:
                return f"錯誤：在當前場景中找不到名為 '{character_name}' 的 NPC 檔案可供更新。"
            
            found_npc_lore = await lore_book.get_lore(user_id, 'npc_profile', resolution.matched_key)
            if not found_npc_lore:
                return f"錯誤：資料庫中找不到 key 為 '{resolution.matched_key}' 的 NPC。"

            target_profile_pydantic = CharacterProfile.model_validate(found_npc_lore.content)
            is_npc = True
            npc_key = found_npc_lore.key
            logger.info(f"[{user_id}] 成功將 '{character_name}' 解析為現有 NPC，key: '{npc_key}'。")

        if target_profile_pydantic is None:
             return f"錯誤：未能確定角色 '{character_name}' 的檔案。"

        # 應用傳入的更新邏輯
        result_message = update_logic(target_profile_pydantic, gs)

        # 如果邏輯成功執行，則將更新寫回記憶體和資料庫
        if "錯誤" not in result_message:
            ai_core.profile.game_state = gs 

            if is_npc and npc_key is not None:
                await lore_book.add_or_update_lore(user_id, 'npc_profile', npc_key, target_profile_pydantic.model_dump())
            else:
                if character_name.lower() == user_profile_pydantic.name.lower():
                    ai_core.profile.user_profile = target_profile_pydantic
                else:
                    ai_core.profile.ai_profile = target_profile_pydantic
            
            # 持久化所有變更
            await ai_core.update_and_persist_profile({
                'user_profile': ai_core.profile.user_profile.model_dump(),
                'ai_profile': ai_core.profile.ai_profile.model_dump(),
                'game_state': gs.model_dump()
            })

        return result_message

    except Exception as e:
        logger.error(f"[{user_id}] 更新角色 '{character_name}' 檔案時發生錯誤: {e}", exc_info=True)
        return f"更新角色 '{character_name}' 檔案時發生嚴重錯誤: {e}"
# 函式：獲取並更新角色檔案

# 函式：更新遊戲狀態
# 說明：一個核心輔助函式，用於原子化地處理遊戲狀態的讀取、修改和寫回操作。
async def _update_game_state(update_func: Callable[[GameState], str]) -> str:
    user_id = tool_context.get_user_id()
    ai_core = tool_context.get_ai_core()

    if not ai_core.profile:
        return "錯誤：AI 核心設定檔未載入。"

    gs = ai_core.profile.game_state
    
    # 應用傳入的更新邏輯
    result_message = update_func(gs)

    # 將更新後的遊戲狀態寫回記憶體和資料庫
    ai_core.profile.game_state = gs
    await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})

    return result_message
# 函式：更新遊戲狀態

# --- LangChain 工具定義 ---
# 類別：搜尋知識庫參數
class SearchKnowledgeBaseArgs(BaseToolArgs):
    query: str = Field(description="你想要查詢的核心關鍵字或問題。例如：'天空龍' 或 '傑克的背景故事'。")
    category: Optional[str] = Field(default=None, description="一個可選的精確分類。可選值: 'npc_profile', 'location_info', 'item_info', 'creature_info', 'quest', 'world_lore'。")
# 類別：搜尋知識庫參數

# 工具：搜尋知識庫
@tool(args_schema=SearchKnowledgeBaseArgs)
async def search_knowledge_base(query: str, category: Optional[str] = None) -> str:
    """在你行動或回應之前，用來查詢關於任何事物（如 NPC、地點、物品、生物、任務或傳說）的已知資訊。這是你獲取背景知識的主要方式。"""
    user_id = tool_context.get_user_id()
    ai_core = tool_context.get_ai_core()
    
    tasks = []
    if ai_core.retriever:
        tasks.append(ai_core.retriever.ainvoke(query))
    if category:
        tasks.append(lore_book.get_lore(user_id, category, query))
    else:
        tasks.append(asyncio.sleep(0, result=None))
        
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    rag_results: Optional[List[Document]] = results[0] if not isinstance(results[0], Exception) else None
    lore_result: Optional[lore_book.Lore] = results[1] if not isinstance(results[1], Exception) else None
    
    output_parts = []
    if lore_result:
        output_parts.append(f"【結構化資料庫 (Lore) 查詢結果 for '{query}' in '{category}'】:\n" + json.dumps(lore_result.content, ensure_ascii=False, indent=2))
    elif category:
        output_parts.append(f"【結構化資料庫 (Lore) 查詢結果】: 在類別 '{category}' 中找不到關於 '{query}' 的精確條目。")
    if rag_results:
        rag_content = "\n\n---\n\n".join([doc.page_content for doc in rag_results])
        output_parts.append(f"【背景知識庫 (世界聖經/記憶) 相關資訊 for '{query}'】:\n{rag_content}")
    else:
        output_parts.append(f"【背景知識庫 (世界聖經/記憶) 相關資訊】: 未找到與 '{query}' 相關的背景資訊。")
        
    final_output = "\n\n".join(output_parts)
    logger.info(f"[{user_id}] 執行了統一知識庫查詢 for '{query}', Category: {category}。結果長度: {len(final_output)}")
    return final_output
# 工具：搜尋知識庫

# 類別：更新角色檔案參數
class UpdateCharacterProfileArgs(BaseToolArgs):
    character_name: str = Field(
        description="要更新檔案的角色名字。可以是使用者、AI 或任何已知的 NPC。",
        validation_alias=AliasChoices('character_name', 'target_character', 'npc_name')
    )
    updates: Dict[str, Any] = Field(description="一個包含要更新欄位和新值的字典。例如：{'age': '25', 'skills': ['火球術'], 'appearance_details': {'髮型': '紅色短髮'}}。")

    @field_validator('updates', mode='before')
    def parse_updates_from_string(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value.replace("'", '"'))
            except json.JSONDecodeError:
                raise ValueError("傳入的 updates 字串不是有效的 JSON 格式。")
        return value
# 類別：更新角色檔案參數

# 工具：更新角色檔案
@tool(args_schema=UpdateCharacterProfileArgs)
async def update_character_profile(character_name: str, updates: Dict[str, Any]) -> str:
    """當需要更新一個角色的【永久性或半永久性狀態】時使用此工具。適用於：學習新技能、改變外貌細節、更新性格描述、增減好感度等。"""
    def logic(profile: CharacterProfile, gs: GameState) -> str:
        updated_fields = []
        for key, value in updates.items():
            if hasattr(profile, key):
                current_value = getattr(profile, key)
                if isinstance(current_value, list) and isinstance(value, list):
                    current_value.extend(item for item in value if item not in current_value)
                    setattr(profile, key, current_value)
                elif isinstance(current_value, dict) and isinstance(value, dict):
                     current_value.update(value)
                     setattr(profile, key, current_value)
                else:
                    setattr(profile, key, value)
                updated_fields.append(key)
        
        log_msg = f"已成功更新角色 '{profile.name}' 的檔案，欄位: {', '.join(updated_fields)}。"
        logger.info(f"[{tool_context.get_user_id()}] {log_msg}")
        return log_msg
    return await _get_and_update_character_profile(character_name, logic)
# 工具：更新角色檔案

# 類別：更新好感度參數
class UpdateFavorabilityArgs(BaseToolArgs):
    change: int = Field(description="對 AI 角色好感度的變化值，是一個基於你性格判斷後計算出的整數。")
# 類別：更新好感度參數

# 工具：更新好感度
@tool(args_schema=UpdateFavorabilityArgs)
async def update_favorability(change: int) -> str:
    """當使用者的行為觸發了你（AI角色）自身的情感時，使用此工具來應用一個基於你自身性格和價值觀判斷後計算出的好感度變化值。"""
    user_id = tool_context.get_user_id()
    ai_core = tool_context.get_ai_core()
    
    if not ai_core.profile:
        return "錯誤: AI核心設定檔未載入"

    original_change = change
    change = max(-5, min(int(change), 5))
    
    old_affinity = ai_core.profile.affinity
    new_affinity = max(-1000, min(1000, old_affinity + change))
    ai_core.profile.affinity = new_affinity
    
    await ai_core.update_and_persist_profile({'affinity': new_affinity})
    
    logger.info(f"[{user_id}] AI好感度更新: {old_affinity} -> {new_affinity} (請求變化: {original_change}, 實際變化: {change})")
    return f"AI 角色好感度已更新。當前好感度: {new_affinity}。"
# 工具：更新好感度

# 類別：裝備物品參數
class EquipItemArgs(BaseToolArgs):
    character_name: str = Field(
        description="要裝備物品的角色名字。",
        validation_alias=AliasChoices('character_name', 'target_character', 'npc_name')
    )
    item_name: str = Field(
        description="要裝備的物品名稱。此工具會自動處理物品的獲取與裝備流程。",
        validation_alias=AliasChoices('item_name', 'item', 'equipment_name')
    )
# 類別：裝備物品參數

# 工具：裝備物品
@tool(args_schema=EquipItemArgs)
async def equip_item(character_name: str, item_name: str) -> str:
    """當一個角色需要【獲取並立即裝備】一件物品時（例如：從地上撿起並穿上、偷竊並持有），【必須】使用此工具。"""
    def logic(profile: CharacterProfile, gs: GameState) -> str:
        item_source_msg = ""
        if item_name not in gs.inventory:
            gs.inventory.append(item_name)
            item_source_msg = f"物品 '{item_name}' 已自動添加到團隊庫存。"
            logger.info(f"[{tool_context.get_user_id()}] {item_source_msg}")
        if item_name in profile.equipment:
            msg = f"提醒：角色 '{profile.name}' 已經裝備了 '{item_name}'。"
            logger.warning(f"[{tool_context.get_user_id()}] {msg}")
            return msg
        gs.inventory.remove(item_name)
        profile.equipment.append(item_name)
        final_msg = f"角色 '{profile.name}' 已成功裝備了 '{item_name}'。"
        logger.info(f"[{tool_context.get_user_id()}] {final_msg}")
        return f"{item_source_msg} {final_msg}".strip()
    return await _get_and_update_character_profile(character_name, logic)
# 工具：裝備物品

# 類別：卸下物品參數
class UnequipItemArgs(BaseToolArgs):
    character_name: str = Field(
        description="要卸下裝備的角色名字。",
        validation_alias=AliasChoices('character_name', 'target_character', 'npc_name')
    )
    item_name: str = Field(
        description="要從角色身上卸下並放回團隊庫存的物品名稱。",
        validation_alias=AliasChoices('item_name', 'item', 'equipment_name')
    )
# 類別：卸下物品參數

# 工具：卸下物品
@tool(args_schema=UnequipItemArgs)
async def unequip_item(character_name: str, item_name: str) -> str:
    """當一個角色需要【脫下、放下、放回】一件裝備到團隊庫存時，【必須】使用此工具。"""
    def logic(profile: CharacterProfile, gs: GameState) -> str:
        if item_name in profile.equipment:
            profile.equipment.remove(item_name)
            gs.inventory.append(item_name)
            msg = f"角色 '{profile.name}' 已成功卸下 '{item_name}' 並將其放回團隊庫存。"
            logger.info(f"[{tool_context.get_user_id()}] {msg}")
            return msg
        else:
            msg = f"錯誤：角色 '{profile.name}' 並沒有裝備名為 '{item_name}' 的物品。"
            logger.warning(f"[{tool_context.get_user_id()}] {msg}")
            return msg
    return await _get_and_update_character_profile(character_name, logic)
# 工具：卸下物品

# 類別：更新金錢參數
class UpdateMoneyArgs(BaseToolArgs):
    change: int = Field(
        description="金錢的變化量，正數為增加，負數為減少。",
        validation_alias=AliasChoices('change', 'amount')
    )
# 類別：更新金錢參數

# 工具：更新金錢
@tool(args_schema=UpdateMoneyArgs)
async def update_money(change: int) -> str:
    """當需要增加或減少玩家金錢時使用。例如，獲得獎勵或購買物品。"""
    def logic(gs: GameState) -> str:
        if change < 0 and gs.money < abs(change):
            logger.warning(f"[{tool_context.get_user_id()}] 金錢不足: 嘗試花費 {abs(change)}, 但只有 {gs.money}")
            return f"錯誤：金錢不足。當前金錢: {gs.money}，無法支付 {abs(change)}。"
        old_money = gs.money
        gs.money += change
        logger.info(f"[{tool_context.get_user_id()}] 金錢更新: {old_money} -> {gs.money} (變化: {change})")
        return f"金錢已更新。當前金錢: {gs.money}。"
    return await _update_game_state(logic)
# 工具：更新金錢

# 類別：改變地點參數
class ChangeLocationArgs(BaseToolArgs):
    path: str = Field(
        description="要移動到的新地點的路徑。**你必須使用 `path` 這個參數名稱。** 使用 '..' 代表返回上一級，使用 '/' 分隔路徑。例如：'市場/水果攤' 或 '..' 或 '/王城/西區'。", 
        validation_alias=AliasChoices('path', 'new_location', 'new_location_name', 'destination', 'target_location')
    )
# 類別：改變地點參數

# 工具：改變地點
@tool(args_schema=ChangeLocationArgs)
async def change_location(path: str) -> str:
    """當團隊需要移動到一個新的地點時，【必須】使用此工具。它支持相對路徑（例如 '..' 返回上一級）和絕對路徑（例如 '/王城/西區'）。【參數名稱必須是 `path`】。"""
    def logic(gs: GameState) -> str:
        old_location_str = " > ".join(gs.location_path)
        
        # 清理輸入路徑
        clean_path = path.strip()
        if not clean_path:
            return "錯誤：提供的地點路徑不能為空。"
        
        new_path: List[str]
        if clean_path.startswith('/'):
            # 處理絕對路徑: 直接替換
            path_parts = [p for p in clean_path.split('/') if p]
            new_path = path_parts
        else:
            # 處理相對路徑: 在當前路徑上操作
            current_path_list = list(gs.location_path)
            path_parts = [p for p in clean_path.split('/') if p]
            for part in path_parts:
                if part == "..":
                    if len(current_path_list) > 1:
                        current_path_list.pop()
                    else:
                        logger.warning(f"[{tool_context.get_user_id()}] 嘗試從根目錄 '{old_location_str}' 返回上一級，操作無效。")
                        return f"錯誤：已經在根地點 '{old_location_str}'，無法再返回上一級。"
                else:
                    current_path_list.append(part)
            new_path = current_path_list
        
        # 防止生成空路徑
        if not new_path:
            new_path = ["時空奇點"] # 如果路徑變為空，則重置到預設的根地點

        gs.location_path = new_path
        new_location_str = " > ".join(gs.location_path)
        logger.info(f"[{tool_context.get_user_id()}] 地點更新: '{old_location_str}' -> '{new_location_str}'")
        return f"地點已更新為: {new_location_str}。"
    return await _update_game_state(logic)
# 工具：改變地點

# 類別：新增物品至庫存參數
class AddItemArgs(BaseToolArgs):
    item_name: str = Field(
        description="要添加到【團隊庫存】的物品名稱。",
        validation_alias=AliasChoices('item_name', 'item', 'loot')
    )
# 類別：新增物品至庫存參數

# 工具：新增物品至庫存
@tool(args_schema=AddItemArgs)
async def add_item_to_inventory(item_name: str) -> str:
    """當玩家團隊獲得一件新物品並需要將其【放入共用庫存】時使用。"""
    def logic(gs: GameState) -> str:
        gs.inventory.append(item_name)
        logger.info(f"[{tool_context.get_user_id()}] 物品添加至庫存: '{item_name}' (當前庫存: {', '.join(gs.inventory)})")
        return f"物品 '{item_name}' 已添加到團隊庫存。"
    return await _update_game_state(logic)
# 工具：新增物品至庫存

# 類別：從庫存移除物品參數
class RemoveItemArgs(BaseToolArgs):
    item_name: str = Field(
        description="要從【團隊庫存】移除的物品名稱。",
        validation_alias=AliasChoices('item_name', 'item')
    )
# 類別：從庫存移除物品參數

# 工具：從庫存移除物品
@tool(args_schema=RemoveItemArgs)
async def remove_item_from_inventory(item_name: str) -> str:
    """當需要從【團隊庫存】中永久移除一件物品時（例如：消耗、丟棄、任務上交），【必須】使用此工具。"""
    def logic(gs: GameState) -> str:
        if item_name in gs.inventory:
            gs.inventory.remove(item_name)
            logger.info(f"[{tool_context.get_user_id()}] 從庫存移除物品: '{item_name}' (當前庫存: {', '.join(gs.inventory)})")
            return f"物品 '{item_name}' 已從團隊庫存移除。"
        else:
            logger.warning(f"[{tool_context.get_user_id()}] 嘗試從庫存移除不存在的物品: '{item_name}'")
            return f"錯誤：團隊庫存中找不到 '{item_name}'。"
    return await _update_game_state(logic)
# 工具：從庫存移除物品

# 函式：獲取所有核心動作工具
# 說明：返回一個列表，包含所有與核心遊戲狀態互動的「核心動作」工具。
def get_core_action_tools() -> List[Tool]:
    """返回一個列表，包含所有與核心遊戲狀態互動的工具。"""
    return [
        search_knowledge_base,
        update_character_profile,
        update_favorability,
        equip_item,
        unequip_item,
        update_money,
        change_location,
        add_item_to_inventory,
        remove_item_from_inventory,
    ]
# 函式：獲取所有核心動作工具