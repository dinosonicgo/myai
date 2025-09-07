# src/graph.py 的中文註釋(v22.2 - 最终完整性修复)
# 更新紀錄:
# v22.2 (2025-09-22): [災難性BUG修復] 恢复了在 v19.1 修复中被意外覆盖删除的 `direct_nsfw_node` 函式的定义。此修改确保了图中所有被引用的节点（包括 `direct_nsfw_node` 和 `purification_node`）都拥有其对应的函数实现，从而彻底解决所有 NameError 启动失败问题。
# v22.1 (2025-09-22): [災難性BUG修復] 恢复了在 v16.0 中被错误移除的 `purification_node` 函式的定义。
# v22.0 (2025-09-22): [災難性BUG修復] 解决了因重命名渲染节点导致的 NameError。
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

async def retrieve_memories_node(state: ConversationGraphState) -> Dict:
    """[2] 專用記憶檢索節點，執行RAG操作。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input_for_retrieval = state['messages'][-1].content
    
    logger.info(f"[{user_id}] (Graph|2) Node: retrieve_memories -> 正在基於原始查詢 '{user_input_for_retrieval[:30]}...' 檢索相關長期記憶...")
    
    rag_context_str = await ai_core.retrieve_and_summarize_memories(user_input_for_retrieval)
    return {"rag_context": rag_context_str}

async def query_lore_node(state: ConversationGraphState) -> Dict:
    """[3] 專用LORE查詢節點，從資料庫獲取與當前輸入和【整個場景】相關的所有【非主角】LORE對象。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|3) Node: query_lore -> 正在執行【上下文優先】的LORE查詢...")

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

    is_remote = gs.viewing_mode == 'remote'
    lores_from_input = await ai_core._query_lore_from_entities(user_input, is_remote_scene=is_remote)
    logger.info(f"[{user_id}] (LORE Querier) 從使用者輸入中提取並查詢到 {len(lores_from_input)} 條相關LORE。")

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

async def perceive_and_set_view_node(state: ConversationGraphState) -> Dict:
    """一個統一的節點，負責分析場景、根據意圖設定視角、並持久化狀態。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    intent = state['intent_classification'].intent_type
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: perceive_and_set_view -> 正在基於意圖 '{intent}' 統一處理感知與視角...")

    if not ai_core.profile:
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='錯誤：AI profile 未加載。', action_summary=user_input)}

    gs = ai_core.profile.game_state
    new_viewing_mode = gs.viewing_mode
    new_target_path = gs.remote_target_path

    if 'descriptive' in intent:
        logger.info(f"[{user_id}] (View Mode) 檢測到描述性意圖，準備進入/更新遠程視角。")
        
        scene_context_lores = [lore.content for lore in state.get('raw_lore_objects', []) if lore.category == 'npc_profile']
        scene_context_json_str = json.dumps(scene_context_lores, ensure_ascii=False, indent=2)
        
        location_chain = ai_core.get_contextual_location_chain()
        location_result = await ai_core.ainvoke_with_rotation(
            location_chain, 
            {
                "user_input": user_input,
                "world_settings": ai_core.profile.world_settings or "未設定",
                "scene_context_json": scene_context_json_str
            }
        )
        
        extracted_path = location_result.location_path if location_result else None
        
        if extracted_path:
            new_viewing_mode = 'remote'
            new_target_path = extracted_path
        else:
            logger.warning(f"[{user_id}] (Perception Hub) 描述性意圖未能推斷出有效地點，將回退到本地模式。")
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
        logger.info(f"[{user_id}] (Perception Hub) GameState 無需更新。")

    scene_analysis = SceneAnalysisResult(
        viewing_mode=gs.viewing_mode,
        reasoning=f"基於意圖 '{intent}' 的統一感知結果。",
        target_location_path=gs.remote_target_path,
        focus_entity=None,
        action_summary=user_input
    )
    
    return {"scene_analysis": scene_analysis, "structured_context": ai_core._assemble_context_from_lore(state['raw_lore_objects'], is_remote_scene=(gs.viewing_mode == 'remote'))}

async def expansion_decision_node(state: ConversationGraphState) -> Dict:
    """[5] LORE擴展決策節點，基於場景中是否已有合適角色來做決定。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    raw_lore_objects = state.get('raw_lore_objects', [])
    logger.info(f"[{user_id}] (Graph|5) Node: expansion_decision -> 正在基於語意匹配，判斷是否擴展...")
    
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
    
    logger.info(f"[{user_id}] (Graph|5) 注入決策鏈的【輕量化】現有角色JSON:\n{lore_json_str}")

    examples_str = """
- **情境 1**: 
    - 現有角色JSON: `[{"name": "海妖吟", "description": "一位販賣活魚的女性性神教徒..."}]`
    - 使用者輸入: `繼續描述那個賣魚的女人`
    - **你的決策**: `should_expand: false` (理由應類似於: 場景中已存在符合 '賣魚的女人' 描述的角色 (例如 '海妖吟')，應優先與其互動。)
- **情境 2**:
    - 現有角色JSON: `[{"name": "海妖吟", "description": "一位女性性神教徒..."}]`
    - 使用者輸入: `這時一個衛兵走了過來`
    - **你的決策**: `should_expand: true` (理由應類似於: 場景中缺乏能夠扮演 '衛兵' 的角色，需要創建新角色以響應指令。)
"""

    decision_chain = ai_core.get_expansion_decision_chain()
    decision = await ai_core.ainvoke_with_rotation(
        decision_chain, 
        {
            "user_input": user_input, 
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
            "system_prompt": ai_core.profile.one_instruction, 
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
            "system_prompt": ai_core.profile.one_instruction, 
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

# [v22.2 新增] 恢复 direct_nsfw_node 的定义
async def direct_nsfw_node(state: ConversationGraphState) -> Dict[str, str]:
    """// NSFW 測試方案 v19.0 // 一步到位直接生成最终的 NSFW 小说文本。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|NSFW Direct) Node: direct_nsfw_node -> 正在執行NSFW直通生成...")

    if not ai_core.profile:
        return {"llm_response": "错误：AI profile 未加载，无法生成场景。"}

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
        'npc_context': "(已弃用，请参考 planning_subjects_json)",
        'relevant_npc_context': "(已弃用，请参考 planning_subjects_json)",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': gs.viewing_mode,
        'remote_target_path_str': " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定",
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)
    
    final_text = await ai_core.ainvoke_with_rotation(
        ai_core.get_direct_nsfw_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction,
            "action_sexual_content_prompt": ai_core.modular_prompts.get("action_sexual_content", "警告:性愛模組未加載"),
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str,
            "planning_subjects_json": planning_subjects_json,
            "user_input": user_input,
        },
        retry_strategy='force'
    )

    if not final_text:
        final_text = "（安全備援：NSFW 直通生成鏈最终失败，可能因为内容审查或API临时故障。）"

    return {"llm_response": final_text}

async def tool_execution_node(state: ConversationGraphState) -> Dict[str, str]:
    """[8] 統一的工具執行節點 (主要用於 SFW 路徑)。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    plan = state.get('turn_plan')
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
    
    chain_input = {
        "system_prompt": ai_core.profile.one_instruction if ai_core.profile else "預設系統指令",
        "action_sexual_content_prompt": "（SFW 模式下不適用）",
        "response_style_prompt": ai_core.profile.response_style_prompt if ai_core.profile else "預設風格",
        "turn_plan": turn_plan
    }
        
    narrative_text = await ai_core.ainvoke_with_rotation(
        ai_core.get_narrative_chain(),
        chain_input,
        retry_strategy='euphemize'
    )
    if not narrative_text:
        narrative_text = "（AI 在將 SFW 計劃轉化為故事時遭遇了內容安全限制。）"
    return {"llm_response": narrative_text}

# [v22.1 新增] 恢复 purification_node 的定义
async def purification_node(state: ConversationGraphState) -> Dict[str, str]:
    """對渲染後的文本進行最終的、強制性的詞彙與格式淨化。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    raw_llm_response = state['llm_response']
    logger.info(f"[{user_id}] (Graph|Purification) Node: purification -> 正在對文本進行總編輯級的校對...")

    if not raw_llm_response or not raw_llm_response.strip() or "安全備援" in raw_llm_response:
        return {"llm_response": raw_llm_response}

    purification_chain = ai_core.get_purification_chain()
    
    purified_text = await ai_core.ainvoke_with_rotation(
        purification_chain,
        {
            "action_sexual_content_prompt": ai_core.modular_prompts.get("action_sexual_content", "警告:性愛模組未加載"),
            "raw_text": raw_llm_response
        },
        retry_strategy='force'
    )
    
    if not purified_text:
        logger.warning(f"[{user_id}] (Graph|Purification) 淨化鏈返回空值，將使用未經淨化的原始文本。")
        return {"llm_response": raw_llm_response}

    return {"llm_response": purified_text}

async def validate_and_rewrite_node(state: ConversationGraphState) -> Dict:
    """[10] 統一的輸出驗證與淨化節點。"""
    user_id = state['user_id']
    initial_response = state['llm_response']
    logger.info(f"[{user_id}] (Graph|10) Node: validate_and_rewrite -> 正在對 LLM 輸出進行最終清理與驗證...")
    
    if not initial_response or not initial_response.strip():
        logger.error(f"[{user_id}] 核心鏈在淨化前返回了空的或無效的回應。")
        return {"final_output": "（...）"}
    
    clean_response = re.sub(r'^[=\-*]{3,}\s*$', '', initial_response, flags=re.MULTILINE)
    clean_response = re.sub(r'^【.*?】\s*$', '', clean_response, flags=re.MULTILINE)
    clean_response = re.sub(r'^\s*(旁白|對話)\s*[:：]\s*', '', clean_response, flags=re.MULTILINE)
    
    final_response = clean_response.strip()
    if not final_response:
        logger.warning(f"[{user_id}] LLM 原始輸出在指令洩漏清理後為空。原始輸出為: '{initial_response[:200]}...'")
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

def create_main_response_graph() -> StateGraph:
    """創建主回應圖"""
    graph = StateGraph(ConversationGraphState)
    
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_memories", retrieve_memories_node)
    graph.add_node("query_lore", query_lore_node)
    graph.add_node("perceive_and_set_view", perceive_and_set_view_node)
    graph.add_node("expansion_decision", expansion_decision_node)
    graph.add_node("character_quantification", character_quantification_node)
    graph.add_node("lore_expansion", lore_expansion_node)
    graph.add_node("sfw_planning", sfw_planning_node)
    graph.add_node("remote_sfw_planning", remote_sfw_planning_node)
    graph.add_node("direct_nsfw_node", direct_nsfw_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("sfw_narrative_rendering", sfw_narrative_rendering_node)
    graph.add_node("purification", purification_node)
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)
    graph.add_node("planner_junction", lambda state: {})
    
    def prepare_existing_subjects_node(state: ConversationGraphState) -> Dict:
        lore_objects = state.get('raw_lore_objects', [])
        planning_subjects = [lore.content for lore in lore_objects if lore.category == 'npc_profile']
        logger.info(f"[{state['user_id']}] (Graph) Node: prepare_existing_subjects -> 已將 {len(planning_subjects)} 個現有NPC打包為規劃主體。")
        return {"planning_subjects": planning_subjects}
        
    graph.add_node("prepare_existing_subjects", prepare_existing_subjects_node)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "retrieve_memories")
    graph.add_edge("retrieve_memories", "query_lore")
    graph.add_edge("query_lore", "perceive_and_set_view")
    graph.add_edge("perceive_and_set_view", "expansion_decision")
    
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
            return "direct_nsfw_test"
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
            "direct_nsfw_test": "direct_nsfw_node"
        }
    )
    
    graph.add_edge("sfw_planning", "tool_execution")
    graph.add_edge("remote_sfw_planning", "tool_execution")
    graph.add_edge("tool_execution", "sfw_narrative_rendering")
    graph.add_edge("sfw_narrative_rendering", "purification")
    graph.add_edge("direct_nsfw_node", "purification")
    graph.add_edge("purification", "validate_and_rewrite")
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", END)
    
    return graph.compile()

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
