# src/graph.py 的中文註釋(v22.0 - 渲染器分离修复)
# 更新紀錄:
# v22.0 (2025-09-22): [災難性BUG修復] 解决了因重命名渲染节点导致的 NameError。恢复并重命名了专用于 SFW 路径的 `sfw_narrative_rendering_node`，并确保 `create_main_response_graph` 正确注册了两个独立的渲染器（`sfw_narrative_rendering_node` 和 `final_rendering_node`），从而修复了图的拓扑结构。
# v21.1 (2025-09-10): [災難性BUG修復] 恢复了所有被先前版本错误省略的 `SetupGraph` 相关节点。
# v21.0 (2025-09-09): [重大架構重構] 对图的拓扑结构进行了精细化重构。
import sys
print(f"[DEBUG] graph.py loaded from: {__file__}", file=sys.stderr)
import asyncio
import json
import re
from typing import Dict, List, Literal, Optional, Any

from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, END

from .ai_core import AILover
from .logger import logger
from .graph_state import ConversationGraphState, SetupGraphState
from . import lore_book, tools
from .schemas import (CharacterProfile, TurnPlan, ExpansionDecision, 
                      UserInputAnalysis, SceneAnalysisResult, SceneCastingResult, 
                      WorldGenesisResult, IntentClassificationResult, StyleAnalysisResult,
                      CharacterQuantificationResult)
from .tool_context import tool_context
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 主對話圖 (Main Conversation Graph) 的節點 ---

async def classify_intent_node(state: ConversationGraphState) -> Dict:
    """[1] 圖的入口點，對輸入进行意图分类，并能处理延续性指令以继承持久化的状态。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    
    logger.info(f"[{user_id}] (Graph|1) Node: classify_intent -> 正在進行初步輸入類型分析...")
    input_analysis_chain = ai_core.get_input_analysis_chain()
    input_analysis_result = await ai_core.ainvoke_with_rotation(
        input_analysis_chain,
        {"user_input": user_input},
        retry_strategy='euphemize'
    )
    
    if input_analysis_result and input_analysis_result.input_type == 'continuation':
        if ai_core.profile and ai_core.profile.game_state.last_intent_type:
            last_intent_type = ai_core.profile.game_state.last_intent_type
            logger.info(f"[{user_id}] (Graph|1) 檢測到延续性指令，已從【持久化 GameState】繼承意图: '{last_intent_type}'")
            
            inherited_intent = IntentClassificationResult(
                intent_type=last_intent_type,
                reasoning=f"從持久化狀態繼承了上一輪的 '{last_intent_type}' 意圖。"
            )
            return {
                "intent_classification": inherited_intent,
                "input_analysis": input_analysis_result
            }
        else:
            logger.warning(f"[{user_id}] (Graph|1) 檢測到延续性指令，但 GameState 中沒有意图可供继承，将按常规流程处理。")

    logger.info(f"[{user_id}] (Graph|1) 正在对具体指令 '{user_input[:30]}...' 進行意圖分類...")
    classification_chain = ai_core.get_intent_classification_chain()
    classification_result = await ai_core.ainvoke_with_rotation(
        classification_chain,
        {"user_input": user_input},
        retry_strategy='euphemize'
    )
    
    if not classification_result:
        logger.warning(f"[{user_id}] (Graph|1) 意圖分類鏈失敗，啟動安全備援，預設為 SFW。")
        classification_result = IntentClassificationResult(intent_type='sfw', reasoning="安全備援：分類鏈失敗。")
        
    return {
        "intent_classification": classification_result,
        "input_analysis": input_analysis_result
    }

# 函式：檢索記憶節點 (v29.0 - 源頭清洗)
# 更新紀錄:
# v29.0 (2025-09-08): [重大架構重構] 此節點的職責已升級為【唯一的 NSFW 輸入清洗源頭】。它不再使用脆弱的實體提取鏈，而是強制性地使用更強大的“文學評論家”鏈將原始輸入轉換為一段安全的文學概述。這段安全的概述將被用於 RAG 檢索，並同時儲存到新的 `sanitized_query_for_tools` 狀態中，供所有下游內部工具鏈安全使用，從根本上解決因多個節點重複處理原始 NSFW 輸入而導致的超時和掛起問題。
# v28.0 (2025-09-12): [災難性BUG修復] 專用記憶檢索節點，執行RAG操作。
async def retrieve_memories_node(state: ConversationGraphState) -> Dict:
    """
    [v29.0 源頭清洗] 專用記憶檢索與輸入清洗節點。
    它會將原始輸入轉換為安全的文學概述，用於RAG，並將其傳遞給下游。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input_for_retrieval = state['messages'][-1].content
    
    logger.info(f"[{user_id}] (Graph|2) Node: retrieve_memories -> 正在對輸入 '{user_input_for_retrieval[:30]}...' 執行【源頭清洗】與記憶檢索...")
    
    sanitized_query = ""
    try:
        # --- 步驟 1: 強制性的安全預清洗 ---
        # 使用強大的“文學評論家”鏈，將潛在的 NSFW 輸入轉換為安全的文學概述。
        logger.info(f"[{user_id}] (RAG Pre-cleaner) 正在使用'文學評論家'鏈對輸入進行強制性安全預清洗...")
        literary_chain = ai_core.get_literary_euphemization_chain()

        # 使用 ainvoke_with_rotation 並指定 euphemize 策略，以確保即使清洗鏈本身被審查也能有備援
        sanitized_query = await ai_core.ainvoke_with_rotation(
            literary_chain, 
            {"dialogue_history": user_input_for_retrieval},
            retry_strategy='euphemize' 
        )
        
        # 如果委婉化重試後仍然失敗，則觸發終極備援
        if not sanitized_query or not sanitized_query.strip():
            logger.error(f"[{user_id}] (RAG Pre-cleaner) '文學評論家'清洗鏈最終失敗，將觸發終極備援。")
            raise ValueError("Literary chain failed to produce output.")

        logger.info(f"[{user_id}] (RAG Pre-cleaner) 輸入已成功預清洗為安全的文學概述: '{sanitized_query[:50]}...'")

    except Exception as e:
        # --- 步驟 2: 終極備援 ---
        logger.error(f"[{user_id}] (RAG Pre-cleaner) 安全預清洗失敗: {e}。啟動【終極備援】：直接使用原始輸入進行本地檢索。")
        sanitized_query = user_input_for_retrieval

    # --- 步驟 3: 執行檢索 ---
    # 使用清洗後的安全查詢文本來執行 RAG
    rag_context_str = await ai_core.retrieve_and_summarize_memories(sanitized_query)
    
    # --- 步驟 4: 返回結果，並將安全查詢文本存入狀態供下游使用 ---
    return {
        "rag_context": rag_context_str,
        "sanitized_query_for_tools": sanitized_query
    }
# 函式：檢索記憶節點 (v29.0 - 源頭清洗)

# 函式：查詢 LORE 節點 (v29.0 - 適配安全查詢)
# 更新紀錄:
# v29.0 (2025-09-08): [重大架構重構] 此節點的邏輯被極大簡化。它不再直接處理原始的、有風險的 `user_input`，而是直接從 `ConversationGraphState` 中讀取由上游 `retrieve_memories_node` 生成的、絕對安全的 `sanitized_query_for_tools` 來提取實體。此修改使其完全免疫於因輸入內容審查而導致的掛起或錯誤。
# v28.0 (2025-09-12): [災難性BUG修復] 專用LORE查詢節點，從資料庫獲取與當前輸入和【整個場景】相關的所有【非主角】LORE對象。
async def query_lore_node(state: ConversationGraphState) -> Dict:
    """[v29.0 適配安全查詢] 專用LORE查詢節點，使用預清洗過的查詢文本來提取實體。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    # [v29.0 核心修正] 使用上游節點生成的安全查詢文本
    safe_query_text = state['sanitized_query_for_tools']
    logger.info(f"[{user_id}] (Graph|3) Node: query_lore -> 正在基於【安全查詢文本】 '{safe_query_text[:30]}...' 執行LORE查詢...")

    if not ai_core.profile:
        logger.error(f"[{user_id}] (LORE Querier) ai_core.profile 未加載，無法查詢LORE。")
        return {"raw_lore_objects": []}

    gs = ai_core.profile.game_state
    
    effective_location_path: List[str]
    if gs.viewing_mode == 'remote' and gs.remote_target_path:
        effective_location_path = gs.remote_target_path
    else:
        effective_location_path = gs.location_path
    
    logger.info(f"[{user_id}] (LORE Querier) 已鎖定有效場景: {' > '.join(effective_location_path)}")

    lores_in_scene = await lore_book.get_lores_by_category_and_filter(
        user_id,
        'npc_profile',
        lambda c: c.get('location_path') == effective_location_path
    )
    logger.info(f"[{user_id}] (LORE Querier) 在有效場景中找到 {len(lores_in_scene)} 位常駐NPC。")

    # [v29.0 核心修正] 使用安全查詢文本進行實體提取
    is_remote = gs.viewing_mode == 'remote'
    lores_from_input = await ai_core._query_lore_from_entities(safe_query_text, is_remote_scene=is_remote)
    logger.info(f"[{user_id}] (LORE Querier) 從安全查詢文本中提取並查詢到 {len(lores_from_input)} 條相關LORE。")

    final_lores_map = {lore.key: lore for lore in lores_in_scene}
    for lore in lores_from_input:
        if lore.key not in final_lores_map:
            final_lores_map[lore.key] = lore
            
    protected_names = {
        ai_core.profile.user_profile.name.lower(),
        ai_core.profile.ai_profile.name.lower()
    }
    
    filtered_lores_list = []
    for lore in final_lores_map.values():
        lore_name = lore.content.get('name', '').lower()
        if lore_name not in protected_names:
            filtered_lores_list.append(lore)
        else:
            logger.warning(f"[{user_id}] (LORE Querier) 已過濾掉與核心主角同名的LORE記錄: '{lore.content.get('name')}'")

    logger.info(f"[{user_id}] (LORE Querier) 經過上下文優先合併與過濾後，共鎖定 {len(filtered_lores_list)} 條LORE作為本回合上下文。")
    
    return {"raw_lore_objects": filtered_lores_list}
# 函式：查詢 LORE 節點 (v29.0 - 適配安全查詢)

# 函式：感知并设定视角
async def perceive_and_set_view_node(state: ConversationGraphState) -> Dict:
    """
    [v30.0 修正] 一个统一的节 点，负责分析场景、根据意图设定视角、并持久化状态。
    其职责已被精简，不再负责组装上下文，只专注于视角的分析与更新。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    intent = state['intent_classification'].intent_type
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: perceive_and_set_view -> 正在基於意圖 '{intent}' 统一处理感知与视角...")

    if not ai_core.profile:
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='错误：AI profile 未加载。', action_summary=user_input)}

    gs = ai_core.profile.game_state
    new_viewing_mode = gs.viewing_mode
    new_target_path = gs.remote_target_path

    if 'descriptive' in intent:
        logger.info(f"[{user_id}] (View Mode) 检测到描述性意图，准备进入/更新远程视角。")
        
        # 为了进行地点推断，我们需要一个临时的、轻量级的上下文
        scene_context_lores = [lore.content for lore in state.get('raw_lore_objects_for_view_decision', []) if lore.category == 'npc_profile']
        scene_context_json_str = json.dumps(scene_context_lores, ensure_ascii=False, indent=2)
        
        location_chain = ai_core.get_contextual_location_chain()
        location_result = await ai_core.ainvoke_with_rotation(
            location_chain, 
            {
                "user_input": user_input,
                "world_settings": ai_core.profile.world_settings or "未设定",
                "scene_context_json": scene_context_json_str
            }
        )
        
        extracted_path = location_result.location_path if location_result else None
        
        if extracted_path:
            new_viewing_mode = 'remote'
            new_target_path = extracted_path
        else:
            logger.warning(f"[{user_id}] (Perception Hub) 描述性意图未能推断出有效地点，将回退到本地模式。")
            new_viewing_mode = 'local'
            new_target_path = None
            
    else:
        new_viewing_mode = 'local'
        new_target_path = None

    if gs.viewing_mode != new_viewing_mode or gs.remote_target_path != new_target_path:
        gs.viewing_mode = new_viewing_mode
        gs.remote_target_path = new_target_path
        await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})
        logger.info(f"[{user_id}] (Perception Hub) GameState 已更新: mode={gs.viewing_mode}, path={gs.remote_target_path}")
    else:
        logger.info(f"[{user_id}] (Perception Hub) GameState 无需更新。")

    scene_analysis = SceneAnalysisResult(
        viewing_mode=gs.viewing_mode,
        reasoning=f"基於意圖 '{intent}' 的统一感知结果。",
        target_location_path=gs.remote_target_path,
        focus_entity=None,
        action_summary=user_input
    )
    
    # [v30.0 核心修正] 不再返回 structured_context，因为 LORE 数据尚未完全查询
    return {"scene_analysis": scene_analysis}
# 函式：感知并设定视角



# 函式：组装上下文
async def assemble_context_node(state: ConversationGraphState) -> Dict:
    """
    [v30.0 新增] 一个全新的、职责单一的节 点。
    它的唯一任务是在 LORE 查询完成后，将所有 LORE 数据和游戏状态组装成最终的 structured_context。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    raw_lore_objects = state.get('raw_lore_objects', [])
    is_remote_scene = state.get('scene_analysis', {}).viewing_mode == 'remote'
    
    logger.info(f"[{user_id}] (Graph) Node: assemble_context -> 正在将 {len(raw_lore_objects)} 条 LORE 记录组装为最终上下文...")
    
    structured_context = ai_core._assemble_context_from_lore(raw_lore_objects, is_remote_scene=is_remote_scene)
    
    return {"structured_context": structured_context}
# 函式：组装上下文

# 函式：LORE擴展決策 (v32.0 - 健壯性與安全查詢適配)
# 更新紀錄:
# v32.0 (2025-09-08): [重大架構重構 & 健壯性] 此節點不再處理原始 user_input，而是改為使用上游傳入的、絕對安全的 `sanitized_query_for_tools`。同時，徹底移除了內部脆弱的“預清洗”備援邏輯，改為依賴上游的源頭清洗，從根本上解決了因嵌套備援失敗而導致的 `ValueError`。
# v31.0 (2025-09-12): [災難性BUG修復] LORE擴展決策節點。
async def expansion_decision_node(state: ConversationGraphState) -> Dict:
    """
    [v32.0 修正] LORE擴展決策節點，使用預清洗過的查詢文本進行決策。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    # [v32.0 核心修正] 直接使用上游傳入的安全查詢文本，不再處理原始輸入
    safe_query_text = state['sanitized_query_for_tools']
    raw_lore_objects = state.get('raw_lore_objects', [])
    logger.info(f"[{user_id}] (Graph|5) Node: expansion_decision -> 正在基於【安全查詢文本】 '{safe_query_text[:30]}...' 判斷是否擴展...")

    # [v32.0 核心修正] 移除所有內部的預清洗邏輯，因為輸入已經是安全的
    
    lightweight_lore_for_decision = []
    for lore in raw_lore_objects:
        if lore.category == 'npc_profile':
            content = lore.content
            lightweight_lore_for_decision.append({
                "name": content.get("name"),
                "gender": content.get("gender"),
                "description": content.get("description")
            })

    lore_json_str = json.dumps(lightweight_lore_for_decision, ensure_ascii=False, indent=2)
    
    examples_str = """
- **情境 1**: 
    - 現有角色JSON: `[{"name": "海妖吟", "description": "一位販賣活魚的女性性神教徒..."}]`
    - 使用者輸入: `继续描述那个卖鱼的女人`
    - **你的決策**: `should_expand: false` (理由應類似於: 場景中已存在符合 '賣魚的女人' 描述的角色 (例如 '海妖吟')，應優先與其互動。)
- **情境 2**:
    - 現有角色JSON: `[{"name": "海妖吟", "description": "一位女性性神教徒..."}]`
    - 使用者輸入: `這時一個衛兵走了過來`
    - **你的決策**: `should_expand: true` (理由應類似於: 場景中缺乏能夠扮演 '衛兵' 的角色，需要創建新角色以響應指令。)
"""

    decision_chain = ai_core.get_expansion_decision_chain()
    # [v32.0 核心修正] 將 user_input 替換為 safe_query_text
    decision = await ai_core.ainvoke_with_rotation(
        decision_chain, 
        {
            "user_input": safe_query_text,
            "existing_characters_json": lore_json_str,
            "examples": examples_str
        },
        retry_strategy='euphemize'
    )

    if not decision:
        logger.warning(f"[{user_id}] (Graph|5) LORE擴展決策鏈失敗，安全備援為不擴展。")
        decision = ExpansionDecision(should_expand=False, reasoning="安全備援：決策鏈失敗。")
    
    logger.info(f"[{user_id}] (Graph|5) LORE擴展決策: {decision.should_expand}。理由: {decision.reasoning}")
    return {"expansion_decision": decision}
# 函式：LORE擴展決策 (v32.0 - 健壯性與安全查詢適配)

async def character_quantification_node(state: ConversationGraphState) -> Dict:
    """[6A.1] 將模糊的群體描述轉化為具體的角色列表。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|6A.1) Node: character_quantification -> 正在量化輸入中的角色...")

    quantification_chain = ai_core.get_character_quantification_chain()
    quantification_result = await ai_core.ainvoke_with_rotation(
        quantification_chain,
        {"user_input": user_input},
        retry_strategy='euphemize'
    )

    if not quantification_result or not quantification_result.character_descriptions:
        logger.warning(f"[{user_id}] (Graph|6A.1) 角色量化鏈失敗或返回空列表，LORE擴展將被跳過。")
        return {"quantified_character_list": []}
    
    logger.info(f"[{user_id}] (Graph|6A.1) 角色量化成功，識別出 {len(quantification_result.character_descriptions)} 個待創建角色。")
    return {"quantified_character_list": quantification_result.character_descriptions}

async def lore_expansion_node(state: ConversationGraphState) -> Dict:
    """[6A.2] 專用的LORE擴展執行節點，為量化後的角色列表創建檔案。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    existing_lores = state.get('raw_lore_objects', [])
    quantified_character_list = state.get('quantified_character_list', [])
    
    logger.info(f"[{user_id}] (Graph|6A.2) Node: lore_expansion -> 正在為 {len(quantified_character_list)} 個量化角色執行選角...")
    
    if not quantified_character_list:
        logger.info(f"[{user_id}] (Graph|6A.2) 量化角色列表為空，跳過LORE擴展。")
        planning_subjects = [lore.content for lore in existing_lores if lore.category == 'npc_profile']
        return {"planning_subjects": planning_subjects}

    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|6A.2) ai_core.profile 未加載，跳過 LORE 擴展。")
        return {}

    gs = ai_core.profile.game_state
    effective_location_path = gs.remote_target_path if gs.viewing_mode == 'remote' and gs.remote_target_path else gs.location_path

    cast_result = await ai_core.ainvoke_with_rotation(
        ai_core.get_scene_casting_chain(),
        {
            "world_settings": ai_core.profile.world_settings or "", 
            "current_location_path": effective_location_path,
            "character_descriptions_list": quantified_character_list
        },
        retry_strategy='euphemize'
    )
    
    if cast_result and cast_result.implied_location:
        location_info = cast_result.implied_location
        base_path = [gs.location_path[0]] if gs.location_path else ["未知區域"]
        new_location_path = base_path + [location_info.name]
        lore_key = " > ".join(new_location_path)
        
        await lore_book.add_or_update_lore(user_id, 'location_info', lore_key, location_info.model_dump())
        logger.info(f"[{user_id}] (Scene Anchor) 已成功為場景錨定並創建新地點LORE: '{lore_key}'")
        
        gs.viewing_mode = 'remote'
        gs.remote_target_path = new_location_path
        await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})
        logger.info(f"[{user_id}] (Scene Anchor) GameState 已強制更新為遠程視角，目標: {new_location_path}")

    planning_subjects = [lore.content for lore in existing_lores if lore.category == 'npc_profile']
    
    if cast_result and (cast_result.newly_created_npcs or cast_result.supporting_cast):
        created_names = await ai_core._add_cast_to_scene(cast_result)
        logger.info(f"[{user_id}] (Graph|6A.2) 選角完成，創建了 {len(created_names)} 位新角色: {', '.join(created_names)}.")
        
        if created_names:
            newly_created_lores = await lore_book.get_lores_by_category_and_filter(user_id, 'npc_profile', lambda c: c.get('name') in created_names)
            if newly_created_lores:
                planning_subjects.extend([lore.content for lore in newly_created_lores])
    
    logger.info(f"[{user_id}] (Graph|6A.2) 已將 {len(planning_subjects)} 位角色 (新舊合併) 成功綁定為本回合的規劃主體。")
    return {"planning_subjects": planning_subjects}

async def sfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7A] SFW路徑專用規劃器，生成結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|7A) Node: sfw_planning -> 正在基於指令 '{user_input[:50]}...' 生成SFW行動計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(execution_rejection_reason="錯誤：AI profile 未加載，無法規劃。")}

    planning_subjects_raw = state.get('planning_subjects')
    if planning_subjects_raw is None:
        lore_objects = state.get('raw_lore_objects', [])
        planning_subjects_raw = [lore.content for lore in lore_objects if lore.category == 'npc_profile']
    planning_subjects_json = json.dumps(planning_subjects_raw, ensure_ascii=False, indent=2)

    gs = ai_core.profile.game_state
    chat_history_str = _get_formatted_chat_history(ai_core, user_id)

    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': state.get('structured_context', {}).get('possessions_context', ''),
        'quests_context': state.get('structured_context', {}).get('quests_context', ''),
        'location_context': state.get('structured_context', {}).get('location_context', ''),
        'npc_context': "(已棄用，請參考 planning_subjects_json)",
        'relevant_npc_context': "(已棄用，請參考 planning_subjects_json)",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': gs.viewing_mode,
        'remote_target_path_str': " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定",
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)
    
    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_sfw_planning_chain(), 
        {
            "one_instruction": ai_core.profile.one_instruction, 
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot, 
            "chat_history": chat_history_str,
            "planning_subjects_json": planning_subjects_json,
            "user_input": user_input,
        },
        retry_strategy='euphemize'
    )
    if not plan:
        plan = TurnPlan(execution_rejection_reason="安全備援：SFW規劃鏈失敗。")
    return {"turn_plan": plan}


# 函式：獲取摘要後的對話歷史 (v28.0 - 終極備援修正)
# 更新紀錄:
# v28.0 (2025-09-08): [災難性BUG修復] 徹底重構了此函式的終極備援邏輯。舊版本在所有摘要嘗試失敗後，會不安全地返回原始的、未經處理的對話歷史，這是導致 AI“偷懶”並重複歷史文本的根本原因。新版本在所有嘗試失敗後，將返回一個安全的中性提示字符串，從而切斷了將重複或露骨內容污染到下游鏈的數據源。
# v27.0 (2025-09-08): [災難性BUG修復] 補充了缺失的導入語句。
async def _get_summarized_chat_history(ai_core: AILover, user_id: str, num_messages: int = 8) -> str:
    """
    [v28.0 修正] 提取並摘要最近的對話歷史，並內建一個強大的、基於「文學評論家」重寫的 NSFW 內容安全備援機制。
    """
    if not ai_core.profile: return "（沒有最近的對話歷史）"
    chat_history_manager = ai_core.session_histories.get(user_id, ChatMessageHistory())
    if not chat_history_manager.messages:
        return "（沒有最近的對話歷史）"
        
    recent_messages = chat_history_manager.messages[-num_messages:]
    if not recent_messages:
        return "（沒有最近的對話歷史）"

    raw_history_text = "\n".join([f"{'使用者' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in recent_messages])

    # 創建即時的、輕量級的摘要鏈
    summarizer_prompt_template = """你的唯一任務是扮演一名小說編輯。請閱讀下方的【對話紀錄】，並將其提煉成一段簡潔的、客觀的、第三人稱的【前情提要】。

【核心規則】
1.  **只提取核心劇情**: 你的摘要【必須且只能】包含關鍵的劇情發展、角色的核心行動和重要的狀態變化。
2.  **禁止對話**: 【絕對禁止】在摘要中包含任何直接的對話引號。
3.  **保持中立**: 不要添加任何原始文本中沒有的推論或評論。
4.  **簡潔至上**: 你的目標是用最少的文字講清楚故事的來龍去脈。

---
【對話紀錄】:
{dialogue_history}
---
【前情提要】:
"""
    summarizer_prompt = ChatPromptTemplate.from_template(summarizer_prompt_template)
    summarizer_llm = ai_core._create_llm_instance(temperature=0.0)
    summarizer_chain = summarizer_prompt | summarizer_llm | StrOutputParser()

    try:
        # --- 步驟 1: 樂觀嘗試 ---
        logger.info(f"[{user_id}] (History Summarizer) 正在樂觀嘗試直接摘要原始歷史文本...")
        summary = await summarizer_chain.ainvoke({"dialogue_history": raw_history_text})

        if not summary or not summary.strip():
            raise Exception("SafetyError: Direct summarization returned empty content.")
            
        logger.info(f"[{user_id}] (History Summarizer) 直接摘要成功。")
        return f"【前情提要】:\n{summary}"

    except Exception as e:
        error_str = str(e).lower()
        if "safety" in error_str or "blocked" in error_str:
            logger.warning(f"[{user_id}] (History Summarizer) 直接摘要失敗，觸發【文學評論家】NSFW安全備援...")
            try:
                literary_chain = ai_core.get_literary_euphemization_chain()
                safe_literary_overview = await literary_chain.ainvoke({"dialogue_history": raw_history_text})
                
                if not safe_literary_overview or not safe_literary_overview.strip():
                    raise Exception("Literary euphemization also returned empty content.")

                logger.info(f"[{user_id}] (History Summarizer) 文學式委婉化成功，正在基於安全的概述重新生成摘要...")
                
                final_summary = await summarizer_chain.ainvoke({"dialogue_history": safe_literary_overview})

                if not final_summary or not final_summary.strip():
                     raise Exception("Final summarization after euphemization returned empty content.")

                logger.info(f"[{user_id}] (History Summarizer) NSFW 安全備援成功完成。")
                return f"【前情提要】:\n{final_summary}"

            except Exception as fallback_e:
                # [v28.0 核心修正] 終極備援不再返回原始歷史
                logger.error(f"[{user_id}] (History Summarizer) 【文學評論家】備援機制最終失敗: {fallback_e}。啟動終極備援。", exc_info=False) # 減少日誌噪音
                return "（歷史對話摘要因內容審查而生成失敗，部分上下文可能缺失。）"
        else:
            logger.error(f"[{user_id}] (History Summarizer) 生成摘要時發生非安全相關的未知錯誤: {e}。啟動終極備援。", exc_info=True)
            return "（歷史對話摘要因技術錯誤而生成失敗，部分上下文可能缺失。）"
# 函式：獲取摘要後的對話歷史 (v28.0 - 終極備援修正)



async def remote_sfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7D] SFW 描述路徑專用規劃器，生成遠景場景的結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|7D) Node: remote_sfw_planning -> 正在基於指令 '{user_input[:50]}...' 生成遠程SFW場景計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(execution_rejection_reason="錯誤：AI profile 未加載，無法規劃。")}

    scene_analysis = state.get('scene_analysis')
    gs = ai_core.profile.game_state
    target_location_path: Optional[List[str]] = None

    if scene_analysis and scene_analysis.target_location_path:
        target_location_path = scene_analysis.target_location_path
        logger.info(f"[{user_id}] (Graph|7D) 已從當前回合分析中獲取遠程目標: {target_location_path}")
    elif gs.viewing_mode == 'remote' and gs.remote_target_path:
        target_location_path = gs.remote_target_path
        logger.warning(f"[{user_id}] (Graph|7D) 當前回合分析未提供目標，已從持久化 GameState 中成功回退。目標: {target_location_path}")
    else:
        error_msg = "錯誤：未能從當前回合分析或持久化狀態中解析出遠程觀察的目標地點。"
        logger.error(f"[{user_id}] (Graph|7D) {error_msg}")
        return {"turn_plan": TurnPlan(execution_rejection_reason=error_msg)}

    target_location_path_str = " > ".join(target_location_path)
    
    planning_subjects_raw = state.get('planning_subjects')
    if planning_subjects_raw is None:
        lore_objects = state.get('raw_lore_objects', [])
        planning_subjects_raw = [lore.content for lore in lore_objects if lore.category == 'npc_profile']
    planning_subjects_json = json.dumps(planning_subjects_raw, ensure_ascii=False, indent=2)

    chat_history_str = _get_formatted_chat_history(ai_core, user_id)

    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': "(遠程觀察模式)",
        'quests_context': "(遠程觀察模式)",
        'location_context': f"遠程觀察地點: {target_location_path_str}",
        'npc_context': "(已棄用，請參考 planning_subjects_json)",
        'relevant_npc_context': "(已棄用，請參考 planning_subjects_json)",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': 'remote',
        'remote_target_path_str': target_location_path_str,
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)

    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_remote_sfw_planning_chain(),
        {
            "one_instruction": ai_core.profile.one_instruction, 
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str,
            "planning_subjects_json": planning_subjects_json,
            "target_location_path_str": target_location_path_str,
            "user_input": user_input,
        },
        retry_strategy='euphemize'
    )
    if not plan:
        plan = TurnPlan(execution_rejection_reason="安全備援：遠程SFW規劃鏈失敗。")
    return {"turn_plan": plan}

async def nsfw_breakthrough_node(state: ConversationGraphState) -> Dict[str, Any]:
    """[数据伪装-步骤1] 生成初步的、自然语言的“剧本大纲”草稿。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|NSFW Outline Pt.1) Node: nsfw_breakthrough -> 正在生成NSFW故事大纲草稿...")

    if not ai_core.profile:
        return {"narrative_outline": "错误：AI profile 未加载，无法规划。"}

    planning_subjects_raw = state.get('planning_subjects')
    if planning_subjects_raw is None:
        lore_objects = state.get('raw_lore_objects', [])
        planning_subjects_raw = [lore.content for lore in lore_objects if lore.category == 'npc_profile']
    planning_subjects_json = json.dumps(planning_subjects_raw, ensure_ascii=False, indent=2)

    gs = ai_core.profile.game_state
    chat_history_str = _get_formatted_chat_history(ai_core, user_id)

    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': state.get('structured_context', {}).get('possessions_context', ''),
        'quests_context': state.get('structured_context', {}).get('quests_context', ''),
        'location_context': state.get('structured_context', {}).get('location_context', ''),
        'npc_context': "(已棄用，請參考 planning_subjects_json)",
        'relevant_npc_context': "(已棄用，請參考 planning_subjects_json)",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': gs.viewing_mode,
        'remote_target_path_str': " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定",
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)
    
    outline_draft = await ai_core.ainvoke_with_rotation(
        ai_core.get_nsfw_breakthrough_planning_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction,
            "action_sexual_content_prompt": ai_core.modular_prompts.get("action_sexual_content", "警告:性愛模組未加載"),
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str,
            "planning_subjects_json": planning_subjects_json,
            "user_input": user_input,
        },
        retry_strategy='euphemize'
    )
    if not outline_draft:
        outline_draft = "安全備援：NSFW大纲生成鏈最终失败。"

    return {"narrative_outline": outline_draft, "world_snapshot": world_snapshot}

async def nsfw_refinement_node(state: ConversationGraphState) -> Dict[str, str]:
    """[数据伪装-步骤2] 接收大纲草稿，并将其丰富为最终的、详细的故事大纲。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    narrative_outline_draft = state['narrative_outline']
    logger.info(f"[{user_id}] (Graph|NSFW Outline Pt.2) Node: nsfw_refinement -> 正在润色NSFW故事大纲...")

    if not ai_core.profile or "安全備援" in narrative_outline_draft:
        return {} 

    chat_history_str = _get_formatted_chat_history(ai_core, user_id)
    world_snapshot = state.get('world_snapshot', '') 

    final_outline = await ai_core.ainvoke_with_rotation(
        ai_core.get_nsfw_refinement_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str,
            "narrative_outline_draft": narrative_outline_draft
        },
        retry_strategy='euphemize'
    )
    if not final_outline:
        logger.warning(f"[{user_id}] (Graph|NSFW Outline Pt.2) NSFW大纲润色链返回空值，将使用未经润色的原始大纲。")
        return {}

    return {"narrative_outline": final_outline}

async def tool_execution_node(state: ConversationGraphState) -> Dict[str, str]:
    """[8] 統一的工具執行節點 (主要用於 SFW 路徑)。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    plan = state.get('turn_plan') # turn_plan 只在 SFW 路径中存在
    logger.info(f"[{user_id}] (Graph|8) Node: tool_execution -> 正在執行行動計劃中的工具...")
    
    if not plan or not plan.character_actions:
        return {"tool_results": "系統事件：無任何工具被調用。"}
    try:
        results_summary = await ai_core._execute_planned_actions(plan)
    except Exception as e:
        logger.error(f"[{user_id}] (Graph|8) 工具執行時發生未捕獲的異常: {e}", exc_info=True)
        results_summary = f"系統事件：工具執行時發生嚴重錯誤: {e}"
    finally:
        tool_context.set_context(None, None)
    
    return {"tool_results": results_summary}

# [v22.0 新增] 恢复并重命名的 SFW 专用渲染节点
async def sfw_narrative_rendering_node(state: ConversationGraphState) -> Dict[str, str]:
    """[SFW Path] 将 SFW 的 TurnPlan 渲染成小说文本。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state.get('turn_plan')
    logger.info(f"[{user_id}] (Graph|9 SFW) Node: sfw_narrative_rendering -> 正在將 SFW 行動計劃渲染為小說...")

    if not turn_plan:
        return {"llm_response": "（系統錯誤：未能生成有效的 SFW 行動計劃。）"}
        
    if turn_plan.execution_rejection_reason:
        logger.warning(f"[{user_id}] (SFW Narrator) 檢測到上游規劃節點的執行否決，跳過渲染。理由: {turn_plan.execution_rejection_reason}")
        return {"llm_response": turn_plan.execution_rejection_reason}
    
    # 注意：这里我们调用的是旧的、通用的 get_narrative_chain，因为它能处理 TurnPlan
    # 在未来的版本中，可以为 SFW 创建一个专门的、更简单的渲染链
    chain_input = {
        "system_prompt": ai_core.profile.one_instruction if ai_core.profile else "預設系統指令",
        "action_sexual_content_prompt": "（SFW 模式下不適用）",
        "response_style_prompt": ai_core.profile.response_style_prompt if ai_core.profile else "預設風格",
        "turn_plan": turn_plan
    }
        
    narrative_text = await ai_core.ainvoke_with_rotation(
        ai_core.get_narrative_chain(), # 使用旧的渲染器
        chain_input,
        retry_strategy='euphemize' # SFW 路径使用委婉化重试
    )
    if not narrative_text:
        narrative_text = "（AI 在將 SFW 計劃轉化為故事時遭遇了內容安全限制。）"
    return {"llm_response": narrative_text}

async def final_rendering_node(state: ConversationGraphState) -> Dict[str, str]:
    """[数据伪装-最终步骤] 将最终的自然语言大纲渲染为电影感小说。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    narrative_outline = state['narrative_outline']
    logger.info(f"[{user_id}] (Graph|Final Rendering) Node: final_rendering -> 正在将故事大纲渲染为最终小说...")

    if not narrative_outline or "安全備援" in narrative_outline:
        return {"llm_response": narrative_outline or "（系统错误：未能生成有效的叙事大纲。）"}
        
    chain_input = {
        "system_prompt": ai_core.profile.one_instruction if ai_core.profile else "預設系統指令",
        "action_sexual_content_prompt": ai_core.modular_prompts.get("action_sexual_content", "警告：性愛內容模組未加載。"),
        "response_style_prompt": ai_core.profile.response_style_prompt if ai_core.profile else "預設風格",
        "narrative_outline": narrative_outline
    }
        
    narrative_text = await ai_core.ainvoke_with_rotation(
        ai_core.get_final_novelist_chain(),
        chain_input,
        retry_strategy='force'
    )
    if not narrative_text:
        narrative_text = "（AI 在将故事大纲扩展为最终小说时遭遇了内容安全限制。）"
    return {"llm_response": narrative_text}

async def validate_and_rewrite_node(state: ConversationGraphState) -> Dict:
    """[10] 統一的輸出驗證與淨化節點。"""
    user_id = state['user_id']
    initial_response = state['llm_response']
    logger.info(f"[{user_id}] (Graph|10) Node: validate_and_rewrite -> 正在對 LLM 原始輸出進行內容保全式淨化...")
    
    if not initial_response or not initial_response.strip():
        logger.error(f"[{user_id}] 核心鏈在淨化前返回了空的或無效的回應。")
        return {"final_output": "（...）"}
    
    clean_response = initial_response
    clean_response = re.sub(r'（(思考|行動|自我觀察)\s*[:：\s\S]*?）', '', clean_response)
    clean_response = re.sub(r'^\s*(旁白|對話)\s*[:：]\s*', '', clean_response, flags=re.MULTILINE)
    if '旁白:' in clean_response or '對話:' in clean_response:
        logger.warning(f"[{user_id}] 檢測到非標準格式的標籤洩漏，啟動備援清理。")
        clean_response = clean_response.replace('旁白:', '').replace('對話:', '')
        clean_response = clean_response.replace('旁白：', '').replace('對話：', '')
    
    final_response = clean_response.strip()
    if not final_response:
        logger.warning(f"[{user_id}] LLM 原始輸出在淨化後為空。原始輸出為: '{initial_response[:200]}...'")
        return {"final_output": "（...）"}
        
    return {"final_output": final_response}

async def persist_state_node(state: ConversationGraphState) -> Dict:
    """[11] 統一的狀態持久化節點，負責儲存對話歷史並將當前意圖持久化。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    clean_response = state['final_output']
    intent_classification = state.get('intent_classification')
    logger.info(f"[{user_id}] (Graph|11) Node: persist_state -> 正在持久化狀態與記憶...")
    
    if ai_core.profile and intent_classification:
        current_intent_type = intent_classification.intent_type
        if ai_core.profile.game_state.last_intent_type != current_intent_type:
            logger.info(f"[{user_id}] (Persist) 正在將當前意圖 '{current_intent_type}' 持久化到 GameState...")
            ai_core.profile.game_state.last_intent_type = current_intent_type
            await ai_core.update_and_persist_profile({'game_state': ai_core.profile.game_state.model_dump()})

    if clean_response and clean_response != "（...）":
        chat_history_manager = ai_core.session_histories.get(user_id)
        if chat_history_manager:
            chat_history_manager.add_user_message(user_input)
            chat_history_manager.add_ai_message(clean_response)
        
        last_interaction_text = f"使用者 '{ai_core.profile.user_profile.name}' 說: {user_input}\n\n[場景回應]:\n{clean_response}"
        tasks = []
        tasks.append(ai_core._generate_and_save_personal_memory(last_interaction_text))
        if ai_core.vector_store:
            tasks.append(asyncio.to_thread(ai_core.vector_store.add_texts, [last_interaction_text], metadatas=[{"source": "history"}]))
        
        async def save_to_sql():
            from .database import AsyncSessionLocal, MemoryData
            import time
            timestamp = time.time()
            async with AsyncSessionLocal() as session:
                session.add(MemoryData(user_id=user_id, content=last_interaction_text, timestamp=timestamp, importance=1))
                await session.commit()
        
        tasks.append(save_to_sql())
        await asyncio.gather(*tasks, return_exceptions=True)
        
    return {}

def _get_formatted_chat_history(ai_core: AILover, user_id: str, num_messages: int = 10) -> str:
    """從 AI 核心實例中提取並格式化最近的對話歷史。"""
    if not ai_core.profile: return "（沒有最近的對話歷史）"
    chat_history_manager = ai_core.session_histories.get(user_id, ChatMessageHistory())
    if not chat_history_manager.messages:
        return "（沒有最近的對話歷史）"
    
    recent_messages = chat_history_manager.messages[-num_messages:]
    
    formatted_history = []
    for msg in recent_messages:
        role = "使用者" if isinstance(msg, HumanMessage) else ai_core.profile.ai_profile.name
        formatted_history.append(f"{role}: {msg.content}")
        
    return "\n".join(formatted_history)

def route_expansion_decision(state: ConversationGraphState) -> Literal["expand_lore", "continue_to_planner"]:
    """根據LORE擴展決策，決定是否進入擴展節點。"""
    if state.get("expansion_decision") and state["expansion_decision"].should_expand:
        return "expand_lore"
    else:
        return "continue_to_planner"





# 函式：直接 NSFW 生成节 点
async def direct_nsfw_generation_node(state: ConversationGraphState) -> Dict[str, str]:
    """
    [v31.0 修正] [NSFW Path] 執行單次指令轟炸，直接生成最終的NSFW小說內容。
    內建全新的“场景自检”和“强制LORE扩展”机制。在生成前，它会检查场景中是否存在具名的NPC。
    如果不存在，它将绕过上游错误的决策，强制执行一次LORE扩展流程，以确保最终的生成请求总是有具体的角色可供驱使，
    从而极大提高突破审查的成功率。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|NSFW Direct) Node: direct_nsfw_generation -> 正在執行【指令轟炸】...")

    if not ai_core.profile:
        return {"llm_response": "（系統錯誤：AI profile 未加載，無法生成內容。）"}

    # [v31.0 核心修正] 场景自检与强制LORE扩展
    latest_characters = state.get('planning_subjects', [])
    if not latest_characters:
        logger.warning(f"[{user_id}] (NSFW Node) 检测到“无米之炊”场景（无具名NPC），正在启动【强制LORE扩展】备援机制...")
        
        # 步骤 1: 量化
        quant_state = await character_quantification_node(state)
        quantified_list = quant_state.get('quantified_character_list', [])
        
        if not quantified_list:
            logger.error(f"[{user_id}] (NSFW Node) 强制LORE扩展失败：量化步骤未能识别出任何可创建的角色。")
        else:
            # 步骤 2: 扩展
            logger.info(f"[{user_id}] (NSFW Node) 强制扩展：已量化出 {len(quantified_list)} 个角色，正在执行扩展...")
            # 创建一个临时的、只包含量化结果的 state 字典来调用扩展节 点
            temp_state_for_expansion = state.copy()
            temp_state_for_expansion['quantified_character_list'] = quantified_list
            
            expansion_result = await lore_expansion_node(temp_state_for_expansion)
            latest_characters = expansion_result.get('planning_subjects', [])
            
            if latest_characters:
                logger.info(f"[{user_id}] (NSFW Node) 强制LORE扩展成功！已生成并加载 {len(latest_characters)} 个角色作为场景主体。")
            else:
                logger.error(f"[{user_id}] (NSFW Node) 强制LORE扩展失败：扩展步骤未能返回任何角色。")

    gs = ai_core.profile.game_state
    chat_history_str = await _get_summarized_chat_history(ai_core, user_id)
    
    # --- 后续流程与之前版本相同，但现在有了可靠的角色数据 ---
    dossiers = []
    for char_data in latest_characters:
        name = char_data.get('name', '未知名稱')
        dossier_content = [f"--- 檔案: {name} (npc_profile) ---"]
        if 'description' in char_data: dossier_content.append(f"- 描述: {char_data['description']}")
        dossiers.append("\n".join(dossier_content))
    
    dossiers.insert(0, f"--- 檔案: {ai_core.profile.ai_profile.name} (AI 角色) ---\n- 描述: {ai_core.profile.ai_profile.description}")
    dossiers.insert(0, f"--- 檔案: {ai_core.profile.user_profile.name} (使用者角色) ---\n- 描述: {ai_core.profile.user_profile.description}")
        
    realtime_npc_context = "\n".join(dossiers) if dossiers else "場景中無已知的特定情報。"
    logger.info(f"[{user_id}] (NSFW Node) 已為世界快照生成了包含 {len(latest_characters)} 個NPC的即時上下文。")

    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': state.get('structured_context', {}).get('possessions_context', ''),
        'quests_context': state.get('structured_context', {}).get('quests_context', ''),
        'location_context': state.get('structured_context', {}).get('location_context', ''),
        'npc_context': realtime_npc_context,
        'relevant_npc_context': "(已整合至上方檔案)",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': gs.viewing_mode,
        'remote_target_path_str': " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定",
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)

    chain_input = {
        "core_protocol_prompt": ai_core.modular_prompts.get("00_core_protocol", "警告：核心協議未加載。"),
        "action_sexual_content_prompt": ai_core.modular_prompts.get("action_sexual_content", "警告：性愛內容模組未加載。"),
        "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
        "world_snapshot": world_snapshot,
        "chat_history": chat_history_str,
        "user_input": user_input,
    }

    narrative_text = await ai_core.ainvoke_with_rotation(
        ai_core.get_direct_nsfw_chain(),
        chain_input,
        retry_strategy='force'
    )

    if not narrative_text:
        narrative_text = "（AI 在直接生成 NSFW 內容時遭遇了無法繞過的内容安全限制。）"
        
    return {"llm_response": narrative_text}
# 函式：直接 NSFW 生成节 点










# 函式：创建主回应图
def create_main_response_graph() -> StateGraph:
    """
    [v30.0 修正] 创建主回应图。
    再次重构了图的拓扑结构，将感知、查询和组装拆分为三个独立的、按正确顺序连接的节 点，
    彻底解决了因数据依赖和执行时序错乱导致的 KeyError。
    """
    graph = StateGraph(ConversationGraphState)
    
    # --- 节 点注册 ---
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_memories", retrieve_memories_node)
    graph.add_node("perceive_and_set_view", perceive_and_set_view_node)
    graph.add_node("query_lore", query_lore_node)
    graph.add_node("assemble_context", assemble_context_node) # 新增节 点
    graph.add_node("expansion_decision", expansion_decision_node)
    graph.add_node("character_quantification", character_quantification_node)
    graph.add_node("lore_expansion", lore_expansion_node)
    graph.add_node("sfw_planning", sfw_planning_node)
    graph.add_node("remote_sfw_planning", remote_sfw_planning_node)
    graph.add_node("direct_nsfw_generation", direct_nsfw_generation_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("sfw_narrative_rendering", sfw_narrative_rendering_node)
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)
    graph.add_node("planner_junction", lambda state: {})
    graph.add_node("rendering_junction", lambda state: {})
    
    def prepare_existing_subjects_node(state: ConversationGraphState) -> Dict:
        lore_objects = state.get('raw_lore_objects', [])
        planning_subjects = [lore.content for lore in lore_objects if lore.category == 'npc_profile']
        logger.info(f"[{state['user_id']}] (Graph) Node: prepare_existing_subjects -> 已将 {len(planning_subjects)} 个现有NPC打包为规划主体。")
        return {"planning_subjects": planning_subjects}
        
    graph.add_node("prepare_existing_subjects", prepare_existing_subjects_node)

    # --- 图的边缘连接 (核心修正) ---
    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "retrieve_memories")
    
    # [v30.0 核心修正] 建立全新的、逻辑正确的拓扑结构
    # 1. 先感知 (perceive) 和设定视角 (set_view)，确定场景。
    graph.add_edge("retrieve_memories", "perceive_and_set_view")
    # 2. 然后带着确定的视角去查询 (query) LORE。
    graph.add_edge("perceive_and_set_view", "query_lore")
    # 3. 在拿到 LORE 数据后，专门进行组装 (assemble) 上下文。
    graph.add_edge("query_lore", "assemble_context")
    # 4. 最后，带着完整的上下文进行扩展决策。
    graph.add_edge("assemble_context", "expansion_decision")
    
    graph.add_conditional_edges(
        "expansion_decision",
        route_expansion_decision,
        { 
            "expand_lore": "character_quantification", 
            "continue_to_planner": "prepare_existing_subjects"
        }
    )
    graph.add_edge("character_quantification", "lore_expansion")
    graph.add_edge("lore_expansion", "planner_junction")
    graph.add_edge("prepare_existing_subjects", "planner_junction")

    def route_to_planner(state: ConversationGraphState) -> str:
        user_id = state['user_id']
        intent_classification = state.get('intent_classification')
        if not intent_classification: return "sfw_planner" 
        intent = intent_classification.intent_type
        ai_core = state['ai_core']
        viewing_mode = ai_core.profile.game_state.viewing_mode if ai_core.profile else 'local'
        logger.info(f"[{user_id}] (Router) Routing to planner. Intent: '{intent}', Final Viewing Mode: '{viewing_mode}'")
        
        if 'nsfw' in intent:
            return "direct_nsfw_planner"
        if viewing_mode == 'remote':
            return "remote_sfw_planner"
        else:
            return "sfw_planner"

    graph.add_conditional_edges(
        "planner_junction",
        route_to_planner,
        { 
            "sfw_planner": "sfw_planning", 
            "remote_sfw_planner": "remote_sfw_planning",
            "direct_nsfw_planner": "direct_nsfw_generation" 
        }
    )
    
    # SFW 路径
    graph.add_edge("sfw_planning", "tool_execution")
    graph.add_edge("remote_sfw_planning", "tool_execution")
    graph.add_edge("tool_execution", "sfw_narrative_rendering")
    graph.add_edge("sfw_narrative_rendering", "rendering_junction")
    
    # NSFW 新路径
    graph.add_edge("direct_nsfw_generation", "rendering_junction")

    # 汇合点及最终流程
    graph.add_edge("rendering_junction", "validate_and_rewrite")
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", END)
    
    return graph.compile()
# 函式：创建主回应图

        






async def process_canon_node(state: SetupGraphState) -> Dict:
    ai_core = state['ai_core']
    canon_text = state['canon_text']
    if canon_text:
        await ai_core.add_canon_to_vector_store(canon_text)
        await ai_core.parse_and_create_lore_from_canon(None, canon_text, is_setup_flow=True)
    return {}

async def complete_profiles_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Setup Graph) Node: complete_profiles_node -> 正在補完角色檔案...")
    completion_chain = ai_core.get_profile_completion_chain()
    if not ai_core.profile:
        logger.error(f"[{user_id}] 在 complete_profiles_node 中 ai_core.profile 為空，無法繼續。")
        return {}
    
    completed_user_profile_task = ai_core.ainvoke_with_rotation(completion_chain, {"profile_json": ai_core.profile.user_profile.model_dump_json()}, retry_strategy='euphemize')
    completed_ai_profile_task = ai_core.ainvoke_with_rotation(completion_chain, {"profile_json": ai_core.profile.ai_profile.model_dump_json()}, retry_strategy='euphemize')
    
    completed_user_profile, completed_ai_profile = await asyncio.gather(completed_user_profile_task, completed_ai_profile_task)

    update_payload = {}
    if completed_user_profile:
        update_payload['user_profile'] = completed_user_profile.model_dump()
    if completed_ai_profile:
        update_payload['ai_profile'] = completed_ai_profile.model_dump()
        
    if update_payload:
        await ai_core.update_and_persist_profile(update_payload)
        
    return {}

async def world_genesis_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    
    if not ai_core.profile:
        raise Exception("AI Profile is not loaded for world genesis.")

    genesis_chain = ai_core.get_world_genesis_chain()
    genesis_result = await ai_core.ainvoke_with_rotation(genesis_chain, {"world_settings": ai_core.profile.world_settings, "username": ai_core.profile.user_profile.name, "ai_name": ai_core.profile.ai_profile.name}, retry_strategy='force')
    
    if not genesis_result:
        raise Exception("世界創世鏈返回了空結果，可能是內容審查。")
        
    gs = ai_core.profile.game_state
    gs.location_path = genesis_result.location_path
    await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})
    
    await lore_book.add_or_update_lore(user_id, 'location_info', " > ".join(genesis_result.location_path), genesis_result.location_info.model_dump())
    
    for npc in genesis_result.initial_npcs:
        npc_key = " > ".join(genesis_result.location_path) + f" > {npc.name}"
        await lore_book.add_or_update_lore(user_id, 'npc_profile', npc_key, npc.model_dump())
        
    return {"genesis_result": genesis_result}

async def generate_opening_scene_node(state: SetupGraphState) -> Dict:
    ai_core = state['ai_core']
    opening_scene = await ai_core.generate_opening_scene()
    
    if not opening_scene or not opening_scene.strip():
        opening_scene = (f"在一片柔和的光芒中，你和 {ai_core.profile.ai_profile.name} 發現自己身處於一個寧靜的空間裡...")
        
    return {"opening_scene": opening_scene}

def create_setup_graph() -> StateGraph:
    """創建設定圖"""
    graph = StateGraph(SetupGraphState)
    graph.add_node("process_canon", process_canon_node)
    graph.add_node("complete_profiles", complete_profiles_node)
    graph.add_node("world_genesis", world_genesis_node)
    graph.add_node("generate_opening_scene", generate_opening_scene_node)
    graph.set_entry_point("process_canon")
    graph.add_edge("process_canon", "complete_profiles")
    graph.add_edge("complete_profiles", "world_genesis")
    graph.add_edge("world_genesis", "generate_opening_scene")
    graph.add_edge("generate_opening_scene", END)
    return graph.compile()















