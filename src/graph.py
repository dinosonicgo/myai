# src/graph.py 的中文註釋(v21.1 - 完整性修正與拓撲健壯化)
# 更新紀錄:
# v21.1 (2025-09-10): [災難性BUG修復] 恢復了所有被先前版本錯誤省略的 `SetupGraph` 相關節點的完整程式碼，解決了 NameError 問題。同時，對主圖的路由匯合點 (Junction) 進行了標準化重構，提高了圖拓撲的健壯性。
# v21.0 (2025-09-09): [重大架構重構] 根據“一功能一節點”藍圖，對圖的拓撲結構進行了根本性的精細化重構。
# v20.1 (2025-09-06): [災難性BUG修復] 徹底修正了圖的拓撲定義。

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
                      WorldGenesisResult, IntentClassificationResult, StyleAnalysisResult)
from .tool_context import tool_context

# --- 主對話圖 (Main Conversation Graph) 的節點 v21.1 ---


# 函式：場景與動作分析節點 (v2.0 - 預處理強化)
# 更新紀錄:
# v2.0 (2025-09-06): [災難性BUG修復] 徹底重構了此節點的邏輯，以解決因將露骨的原始使用者輸入直接傳遞給分析鏈而導致的內容審查失敗問題。現在，此節點會先調用一個簡單的實體提取鏈來獲取中性關鍵詞，然後用這些關鍵詞構建一個“淨化”後的安全查詢，最後再將此安全查詢傳遞給場景分析鏈。此修改旨在從根本上規避在分析階段的內容審查。
# v1.0 (2025-09-13): [恢復] 恢复在重构中被遗漏的 SFW 路径核心节点，用于判断本地/远程视角。
async def scene_and_action_analysis_node(state: ConversationGraphState) -> Dict:
    """[SFW Path] 專用節點，分析 SFW 場景的視角（本地 vs 遠程）。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: scene_and_action_analysis -> 正在進行場景視角分析...")

    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph) 在 scene_and_action_analysis 中 ai_core.profile 未加載。")
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='錯誤：AI profile 未加載。', action_summary=user_input)}

    # [v2.0 核心修正] "先淨化，後分析" 策略
    try:
        logger.info(f"[{user_id}] (Scene Analysis) 正在對輸入進行預處理以創建安全查詢...")
        entity_chain = ai_core.get_entity_extraction_chain()
        entity_result = await ai_core.ainvoke_with_rotation(entity_chain, {"text_input": user_input})

        if entity_result and entity_result.names:
            sanitized_input_for_analysis = "觀察場景：" + " ".join(entity_result.names)
            logger.info(f"[{user_id}] (Scene Analysis) 已生成安全查詢: '{sanitized_input_for_analysis}'")
        else:
            sanitized_input_for_analysis = user_input
            logger.warning(f"[{user_id}] (Scene Analysis) 未能從輸入中提取實體，將使用原始輸入進行分析，可能存在風險。")
    except Exception as e:
        logger.error(f"[{user_id}] (Scene Analysis) 預處理失敗: {e}", exc_info=True)
        sanitized_input_for_analysis = user_input

    current_location_path = ai_core.profile.game_state.location_path
    scene_analysis_chain = ai_core.get_scene_analysis_chain()
    scene_analysis = await ai_core.ainvoke_with_rotation(
        scene_analysis_chain,
        {"user_input": sanitized_input_for_analysis, "current_location_path_str": " > ".join(current_location_path)},
        retry_strategy='euphemize'
    )

    if not scene_analysis:
        logger.warning(f"[{user_id}] (Graph) 場景分析鏈委婉化重試失敗，啟動安全備援。")
        scene_analysis = SceneAnalysisResult(
            viewing_mode='local', 
            reasoning='安全備援：場景分析鏈失敗。', 
            action_summary=user_input
        )
    
    # 即使分析成功，也要將原始的使用者意圖傳遞下去
    scene_analysis.action_summary = user_input

    return {"scene_analysis": scene_analysis}
# 函式：場景與動作分析節點 (v2.0 - 預處理強化)


# 函式：視角模式路由器
# 更新紀錄:
# v1.0 (2025-09-13): [恢復] 恢复在重构中被遗漏的 SFW 路径核心路由器，用于分发本地/远程流量。
def route_viewing_mode(state: ConversationGraphState) -> Literal["remote_scene", "local_scene"]:
    """[SFW Path] 根據視角分析結果，決定是生成遠程場景還是繼續本地流程。"""
    user_id = state['user_id']
    scene_analysis = state.get("scene_analysis")
    if scene_analysis and scene_analysis.viewing_mode == 'remote':
        logger.info(f"[{user_id}] (Graph) Router: SFW 視角分析為遠程，進入 remote_sfw_planning。")
        return "remote_scene"
    else:
        logger.info(f"[{user_id}] (Graph) Router: SFW 視角分析為本地，繼續本地主流程。")
        return "local_scene"
# 函式：視角模式路由器

# --- 階段一：感知 (Perception) ---

# 函式：[新建] 導演視角狀態管理節點 (v1.0)
# 更新紀錄:
# v1.0 (2025-09-06): [災難性BUG修復] 創建此新節點，專門用於調用 ai_core._update_viewing_mode。將其插入圖的早期流程中，確保在任何規劃開始之前，系統的“導演視角”（本地或遠程）狀態都已被明確設定和持久化。
async def update_viewing_mode_node(state: ConversationGraphState) -> None:
    """調用 ai_core 中的輔助函式來更新並持久化導演視角模式。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph) Node: update_viewing_mode -> 正在更新並持久化導演視角...")
    
    await ai_core._update_viewing_mode(state)
    
    # 這個節點不直接返回更新到 state，因為 ai_core 的方法會直接修改資料庫
    # 和 ai_core 實例內的 GameState。後續節點將讀取到這個最新的狀態。
    return {}
# 函式：[新建] 導演視角狀態管理節點 (v1.0)


async def classify_intent_node(state: ConversationGraphState) -> Dict:
    """[1] 圖的入口點，唯一職責是對原始輸入進行意圖分類。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|1) Node: classify_intent -> 正在對 '{user_input[:30]}...' 進行意圖分類...")
    
    classification_chain = ai_core.get_intent_classification_chain()
    classification_result = await ai_core.ainvoke_with_rotation(
        classification_chain,
        {"user_input": user_input},
        retry_strategy='none'
    )
    
    if not classification_result:
        logger.warning(f"[{user_id}] (Graph|1) 意圖分類鏈失敗，啟動安全備援，預設為 SFW。")
        classification_result = IntentClassificationResult(intent_type='sfw', reasoning="安全備援：分類鏈失敗。")
        
    return {"intent_classification": classification_result}

async def retrieve_memories_node(state: ConversationGraphState) -> Dict:
    """[2] 專用記憶檢索節點，執行RAG操作。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|2) Node: retrieve_memories -> 正在檢索相關長期記憶...")
    
    # 這是 ai_core.py 中為新節點準備的輔助函式
    rag_context_str = await ai_core.retrieve_and_summarize_memories(user_input)
    return {"rag_context": rag_context_str}

async def query_lore_node(state: ConversationGraphState) -> Dict:
    """[3] 專用LORE查詢節點，從資料庫獲取原始LORE對象。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    intent_type = state['intent_classification'].intent_type
    logger.info(f"[{user_id}] (Graph|3) Node: query_lore -> 正在查詢相關LORE實體...")
    
    is_remote = intent_type == 'nsfw_descriptive'
    # 這是 ai_core.py 中為新節點準備的輔助函式
    raw_lore_objects = await ai_core._query_lore_from_entities(user_input, is_remote_scene=is_remote)
    return {"raw_lore_objects": raw_lore_objects}

# 函式：專用上下文組裝節點 (v1.1 - 傳遞原始LORE)
# 更新紀錄:
# v1.1 (2025-09-06): [災難性BUG修復] 修改此節點的返回值，使其除了生成格式化的 `structured_context` 外，還將未經修改的 `raw_lore_objects` 直接透傳下去。這是實現“LORE事實鎖定”機制的關鍵一步，確保後續的規劃節點能夠訪問到完整的、未經摘要的原始 LORE 數據。
# v1.0 (2025-09-12): [架構重構] 創建此專用函式，將上下文格式化邏輯分離。
async def assemble_context_node(state: ConversationGraphState) -> Dict:
    """[4] 專用上下文組裝節點，將原始LORE格式化為LLM可讀的字符串，並透傳原始LORE對象。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    raw_lore = state['raw_lore_objects']
    intent_type = state['intent_classification'].intent_type
    logger.info(f"[{user_id}] (Graph|4) Node: assemble_context -> 正在組裝最終上下文簡報並透傳原始LORE...")
    
    is_remote = intent_type == 'nsfw_descriptive'
    structured_context = ai_core._assemble_context_from_lore(raw_lore, is_remote_scene=is_remote)
    
    # [v1.1 核心修正] 將原始 LORE 對象也加入返回字典中，以便後續節點使用
    return {
        "structured_context": structured_context,
        "raw_lore_objects": raw_lore 
    }
# 函式：專用上下文組裝節點 (v1.1 - 傳遞原始LORE)



# 函式：統一NSFW規劃節點 (v4.0 - 適配規劃主體)
# 更新紀錄:
# v4.0 (2025-09-18): [重大架構重構] 修改了數據源，現在從 state['planning_subjects'] 或 state['raw_lore_objects'] 獲取角色數據，並將其格式化為 planning_subjects_json 傳遞給規劃鏈。
# v3.0 (2025-09-18): [重大架構升級] 修改了此節點的輸入源，改為使用 `sanitized_user_input`。
# v2.0 (2025-09-17): [災難性BUG修復] 創建此統一節點以取代舊的思維鏈。
async def nsfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7B] 統一的 NSFW 互動路徑規劃器，直接生成最終的、露骨的行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['sanitized_user_input'] or state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|7B) Node: nsfw_planning -> 正在基於指令 '{user_input[:50]}...' 生成統一NSFW行動計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}

    # [v4.0 核心修正] 確定規劃主體
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
        ai_core.get_nsfw_planning_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction,
            "action_sexual_content_prompt": ai_core.modular_prompts.get("action_sexual_content", "警告:性愛模組未加載"),
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str,
            "planning_subjects_json": planning_subjects_json, # [v4.0 核心修正]
            "user_input": user_input,
            "username": ai_core.profile.user_profile.name,
        },
        retry_strategy='force'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：NSFW統一規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}
# 函式：統一NSFW規劃節點 (v4.0 - 適配規劃主體)




def _get_formatted_chat_history(ai_core: AILover, user_id: str, num_messages: int = 10) -> str:
    """從 AI 核心實例中提取並格式化最近的對話歷史。"""
    chat_history_manager = ai_core.session_histories.get(user_id, ChatMessageHistory())
    if not chat_history_manager.messages:
        return "（沒有最近的對話歷史）"
    
    # 提取最近的 N 條訊息
    recent_messages = chat_history_manager.messages[-num_messages:]
    
    formatted_history = []
    for msg in recent_messages:
        role = "使用者" if isinstance(msg, HumanMessage) else ai_core.profile.ai_profile.name if ai_core.profile else "AI"
        formatted_history.append(f"{role}: {msg.content}")
        
    return "\n".join(formatted_history)


    
# 函式：LORE擴展決策節點 (v3.0 - 實體存在性優先)
# 更新紀錄:
# v3.0 (2025-09-18): [災難性BUG修復] 根據 ai_core v3.0 的重構，修改了此節點的邏輯。不再進行飽和度分析，而是生成一個簡潔的 LORE 摘要 (lore_summary)，並將其注入到新的決策鏈中，以實現“實體存在性優先”的決策邏輯。
# v2.0 (2025-09-06): [災難性BUG修復] 徹底重構了此節點的邏輯，引入飽和度分析。
# v1.0 (2025-09-09): [架構重構] 創建此專用節點。
async def expansion_decision_node(state: ConversationGraphState) -> Dict:
    """[5] LORE擴展決策節點，基於核心實體是否已存在於LORE中來做決定。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    raw_lore_objects = state.get('raw_lore_objects', [])
    logger.info(f"[{user_id}] (Graph|5) Node: expansion_decision -> 正在基於實體存在性，判斷是否擴展...")
    
    # [v3.0 核心修正] 生成 LORE 摘要以供決策鏈使用
    lore_summary_list = []
    if isinstance(raw_lore_objects, list):
        for lore in raw_lore_objects:
            name = lore.content.get('name', lore.content.get('title', lore.key))
            lore_summary_list.append(f"- [{lore.category}] {name}")
    lore_summary = "\n".join(lore_summary_list) if lore_summary_list else "（當前場景無任何已知 LORE）"
    
    logger.info(f"[{user_id}] (Graph|5) 生成的LORE摘要:\n{lore_summary}")

    decision_chain = ai_core.get_expansion_decision_chain()
    decision = await ai_core.ainvoke_with_rotation(
        decision_chain, 
        {
            "user_input": user_input, 
            "lore_summary": lore_summary # 將 LORE 摘要注入 Prompt
        },
        retry_strategy='euphemize'
    )

    if not decision:
        logger.warning(f"[{user_id}] (Graph|5) LORE擴展決策鏈失敗，安全備援為不擴展。")
        decision = ExpansionDecision(should_expand=False, reasoning="安全備援：決策鏈失敗。")
    
    logger.info(f"[{user_id}] (Graph|5) LORE擴展決策: {decision.should_expand}。理由: {decision.reasoning}")
    return {"expansion_decision": decision}
# 函式：LORE擴展決策節點 (v3.0 - 實體存在性優先)


# 函式：無害化輸入節點 (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-09-18): [重大架構升級] 創建此專用節點，作為所有規劃器前的“淨化層”。它調用委婉化鏈，將可能觸發內容審查的原始使用者輸入，轉化為一個保留核心意圖但用詞中性的安全指令，旨在從根本上解決因輸入“投毒”導致整個規劃鏈被 API 攔截的問題。
async def sanitize_input_node(state: ConversationGraphState) -> Dict:
    """將原始使用者輸入轉化為一個對規劃器安全的、中性的指令。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    intent_type = state['intent_classification'].intent_type
    
    # 只有 NSFW 相關的意圖需要淨化
    if 'nsfw' not in intent_type:
        logger.info(f"[{user_id}] (Graph) Node: sanitize_input -> 意圖為 SFW，跳過淨化。")
        return {"sanitized_user_input": user_input}

    logger.info(f"[{user_id}] (Graph) Node: sanitize_input -> 正在對 NSFW 指令進行無害化處理...")
    
    entity_extraction_chain = ai_core.get_entity_extraction_chain()
    entity_result = await ai_core.ainvoke_with_rotation(entity_extraction_chain, {"text_input": user_input})
    
    if not (entity_result and entity_result.names):
        logger.warning(f"[{user_id}] (Sanitizer) 未能從輸入中提取實體，將使用原始輸入作為安全備援。")
        return {"sanitized_user_input": user_input}
        
    euphemization_chain = ai_core.get_euphemization_chain()
    sanitized_input = await ai_core.ainvoke_with_rotation(
        euphemization_chain,
        {"keywords": entity_result.names},
        retry_strategy='none' # 委婉化本身失敗則無法挽救
    )
    
    if not sanitized_input:
        logger.error(f"[{user_id}] (Sanitizer) 委婉化重構鏈失敗，將使用原始輸入，這極可能導致後續規劃失敗！")
        sanitized_input = user_input
    
    logger.info(f"[{user_id}] (Sanitizer) 指令淨化成功: '{user_input}' -> '{sanitized_input}'")
    return {"sanitized_user_input": sanitized_input}
# 函式：無害化輸入節點 (v1.0 - 全新創建)


# 函式：專用的LORE擴展執行節點 (v5.0 - 輸入淨化)
# 更新紀錄:
# v5.0 (2025-09-06): [災難性BUG修復] 徹底重構了此節點的邏輯，以解決因將露骨的原始使用者輸入直接傳遞給選角鏈而導致的內容審查失敗問題。現在，此節點會先對輸入進行淨化，提取中性關鍵詞，然後再將這些安全的關鍵詞傳遞給 `scene_casting_chain`，從根本上規避了在 LORE 創建階段的內容審查。
# v4.0 (2025-09-18): [重大架構重構] 引入“數據綁定”策略。
# v3.0 (2025-09-18): [災難性BUG修復] 徹底重構了此節點的地點上下文處理邏輯。
async def lore_expansion_node(state: ConversationGraphState) -> Dict:
    """[6A] 專用的LORE擴展執行節點，執行選角，並將新角色綁定為規劃主體。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|6A) Node: lore_expansion -> 正在執行場景選角與LORE擴展...")
    
    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|6A) ai_core.profile 未加載，跳過 LORE 擴展。")
        return {}

    scene_analysis = state.get('scene_analysis')
    gs = ai_core.profile.game_state
    effective_location_path: List[str]

    if gs.viewing_mode == 'remote' and scene_analysis and scene_analysis.target_location_path:
        effective_location_path = scene_analysis.target_location_path
        logger.info(f"[{user_id}] (Graph|6A) LORE擴展檢測到遠程視角，目標地點: {effective_location_path}")
    else:
        effective_location_path = gs.location_path
        logger.info(f"[{user_id}] (Graph|6A) LORE擴展使用本地視角，目標地點: {effective_location_path}")

    # [v5.0 核心修正] "先淨化，後選角" 策略
    try:
        logger.info(f"[{user_id}] (LORE Expansion) 正在對輸入進行預處理以創建安全的選角上下文...")
        entity_chain = ai_core.get_entity_extraction_chain()
        entity_result = await ai_core.ainvoke_with_rotation(entity_chain, {"text_input": user_input})

        if entity_result and entity_result.names:
            sanitized_context_for_casting = "為場景選角：" + " ".join(entity_result.names)
            logger.info(f"[{user_id}] (LORE Expansion) 已生成安全的選角上下文: '{sanitized_context_for_casting}'")
        else:
            sanitized_context_for_casting = user_input
            logger.warning(f"[{user_id}] (LORE Expansion) 未能從輸入中提取實體，將使用原始輸入進行選角，可能存在風險。")
    except Exception as e:
        logger.error(f"[{user_id}] (LORE Expansion) 預處理失敗: {e}", exc_info=True)
        sanitized_context_for_casting = user_input
        
    game_context_for_casting = json.dumps(state.get('structured_context', {}), ensure_ascii=False, indent=2)

    cast_result = await ai_core.ainvoke_with_rotation(
        ai_core.get_scene_casting_chain(),
        {
            "world_settings": ai_core.profile.world_settings or "", 
            "current_location_path": effective_location_path,
            "game_context": game_context_for_casting, 
            "recent_dialogue": sanitized_context_for_casting # 使用淨化後的上下文
        },
        retry_strategy='euphemize'
    )
    
    updates: Dict[str, Any] = {"planning_subjects": []} # 預設為空列表
    if cast_result and (cast_result.newly_created_npcs or cast_result.supporting_cast):
        created_names = await ai_core._add_cast_to_scene(cast_result)
        logger.info(f"[{user_id}] (Graph|6A) 選角完成，創建了 {len(created_names)} 位新角色: {', '.join(created_names)}.")
        
        if created_names:
            lore_query_tasks = [lore_book.get_lores_by_category_and_filter(user_id, 'npc_profile', lambda c: c.get('name') in created_names)]
            results = await asyncio.gather(*lore_query_tasks)
            newly_created_lores = results[0]
            
            if newly_created_lores:
                planning_subjects = [lore.content for lore in newly_created_lores]
                updates["planning_subjects"] = planning_subjects
                logger.info(f"[{user_id}] (Graph|6A) 已將 {len(planning_subjects)} 位新角色成功綁定為本回合的規劃主體。")
    else:
         logger.info(f"[{user_id}] (Graph|6A) 場景選角鏈未返回新角色，規劃主體為空。")

    return updates
# 函式：專用的LORE擴展執行節點 (v5.0 - 輸入淨化)

# --- 階段二：規劃 (Planning) ---




    # 函式：NSFW 初步規劃節點 (v2.0 - 注入對話歷史)
    # 更新紀錄:
    # v2.0 (2025-09-16): [重大邏輯強化] 新增了對話歷史的提取與傳遞，確保初步規劃時能緊密銜接上下文。
    # v1.0 (2025-09-15): [重大架構重構] 创建此新节点，作为 NSFW 思维链的第一步。
async def nsfw_initial_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7B.1] NSFW思维链-步骤1: 生成初步的行动计划草稿。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|7B.1) Node: nsfw_initial_planning -> 正在生成NSFW初步行动计划...")
    
    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}

    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': state.get('structured_context', {}).get('possessions_context', ''),
        'quests_context': state.get('structured_context', {}).get('quests_context', ''),
        'location_context': state.get('structured_context', {}).get('location_context', ''),
        'npc_context': state.get('structured_context', {}).get('npc_context', ''),
        'relevant_npc_context': state.get('structured_context', {}).get('relevant_npc_context', ''),
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)

    # [v2.0 新增] 獲取格式化的對話歷史
    chat_history_str = _get_formatted_chat_history(ai_core, user_id)

    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_nsfw_initial_planning_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "world_snapshot": world_snapshot, 
            "chat_history": chat_history_str, # [v2.0 核心修正] 傳遞對話歷史
            "user_input": state['messages'][-1].content
        },
        retry_strategy='force'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：NSFW初步規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}
    # 函式：NSFW 初步規劃節點 (v2.0 - 注入對話歷史)


    # 函式：NSFW 词汇注入節點 (v2.0 - 注入對話歷史)
    # 更新紀錄:
    # v2.0 (2025-09-16): [功能強化] 新增了對話歷史的提取與傳遞，為詞彙修正提供更完整的上下文。
    # v1.2 (2025-09-05): [災難性BUG修復] 修正了調用鏈時的參數傳遞，補上了缺失的 `system_prompt`。
    # v1.1 (2025-09-15): [災難性BUG修復] 修正了 full_context_dict 的构建逻辑。
async def nsfw_lexicon_injection_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7B.2] NSFW思维链-步骤2: 强制修正计划中的词汇为露骨术语。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph|7B.2) Node: nsfw_lexicon_injection -> 正在注入NSFW露骨词汇...")

    if not ai_core.profile or not turn_plan:
        return {}

    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': state.get('structured_context', {}).get('possessions_context', ''),
        'quests_context': state.get('structured_context', {}).get('quests_context', ''),
        'location_context': state.get('structured_context', {}).get('location_context', ''),
        'npc_context': state.get('structured_context', {}).get('npc_context', ''),
        'relevant_npc_context': state.get('structured_context', {}).get('relevant_npc_context', ''),
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)
    
    # [v2.0 新增] 獲取格式化的對話歷史
    chat_history_str = _get_formatted_chat_history(ai_core, user_id)
    
    corrected_plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_nsfw_lexicon_injection_chain(),
        {
            # [v1.2 核心修正] 傳入完整的系統指令
            "system_prompt": ai_core.profile.one_instruction,
            "action_sexual_content_prompt": ai_core.modular_prompts.get("action_sexual_content", "警告:性愛模組未加載"),
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str, # [v2.0 核心修正] 傳遞對話歷史
            "turn_plan_json": turn_plan.model_dump_json(indent=2)
        },
        retry_strategy='force'
    )
    if not corrected_plan:
        logger.warning(f"[{user_id}] (Graph|7B.2) NSFW词汇注入鏈返回空值，保留原始计划。")
        return {}
        
    return {"turn_plan": corrected_plan}
    # 函式：NSFW 词汇注入節點 (v2.0 - 注入對話歷史)




# 函式：SFW規劃節點 (v24.0 - 適配規劃主體)
# 更新紀錄:
# v24.0 (2025-09-18): [重大架構重構] 修改了數據源，現在從 state['planning_subjects'] 或 state['raw_lore_objects'] 獲取角色數據，並將其格式化為 planning_subjects_json 傳遞給規劃鏈。
# v23.0 (2025-09-18): [重大架構升級] 修改了此節點的輸入源，改為使用 `sanitized_user_input`。
# v22.0 (2025-09-16): [重大邏輯強化] 調用 `_get_formatted_chat_history` 輔助函式來獲取最近的對話歷史。
async def sfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7A] SFW路徑專用規劃器，生成結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['sanitized_user_input'] or state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|7A) Node: sfw_planning -> 正在基於指令 '{user_input[:50]}...' 生成SFW行動計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}

    # [v24.0 核心修正] 確定規劃主體
    planning_subjects_raw = state.get('planning_subjects')
    if planning_subjects_raw is None: # 如果 planning_subjects 未被設置 (來自不擴展的分支)
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
            "planning_subjects_json": planning_subjects_json, # [v24.0 核心修正]
            "user_input": user_input,
        },
        retry_strategy='euphemize'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：SFW規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}
# 函式：SFW規劃節點 (v24.0 - 適配規劃主體)



    # 函式：NSFW 風格合規節點 (v2.0 - 注入對話歷史)
    # 更新紀錄:
    # v2.0 (2025-09-16): [功能強化] 新增了對話歷史的提取與傳遞，為風格修正提供更完整的上下文，以生成更貼切的對話。
    # v1.2 (2025-09-05): [災難性BUG修復] 修正了調用鏈時的參數傳遞，補上了缺失的 `system_prompt`。
    # v1.1 (2025-09-15): [災難性BUG修復] 与 nsfw_lexicon_injection_node 同步，修正了 full_context_dict 的构建逻辑。
async def nsfw_style_compliance_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7B.3] NSFW思维链-步骤3: 检查并补充对话，确保计划符合用户风格。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph|7B.3) Node: nsfw_style_compliance -> 正在进行NSFW风格合规检查...")

    if not ai_core.profile or not turn_plan:
        return {}

    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': state.get('structured_context', {}).get('possessions_context', ''),
        'quests_context': state.get('structured_context', {}).get('quests_context', ''),
        'location_context': state.get('structured_context', {}).get('location_context', ''),
        'npc_context': state.get('structured_context', {}).get('npc_context', ''),
        'relevant_npc_context': state.get('structured_context', {}).get('relevant_npc_context', ''),
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)

    # [v2.0 新增] 獲取格式化的對話歷史
    chat_history_str = _get_formatted_chat_history(ai_core, user_id)

    final_plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_nsfw_style_compliance_chain(),
        {
            # [v1.2 核心修正] 傳入完整的系統指令
            "system_prompt": ai_core.profile.one_instruction,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str, # [v2.0 核心修正] 傳遞對話歷史
            "turn_plan_json": turn_plan.model_dump_json(indent=2)
        },
        retry_strategy='force'
    )
    if not final_plan:
        logger.warning(f"[{user_id}] (Graph|7B.3) NSFW风格合规鏈返回空值，保留修正前计划。")
        return {}

    return {"turn_plan": final_plan}
    # 函式：NSFW 風格合規節點 (v2.0 - 注入對話歷史)


# 函式：遠程 SFW 規劃節點 (v5.0 - 狀態持久化與三級備援)
# 更新紀錄:
# v5.0 (2025-09-18): [災難性BUG修復] 徹底重構了 target_location_path 的獲取邏輯，引入“三級備援”機制：1. 優先使用當前回合的 scene_analysis 結果。2. 如果失敗，則回退到從持久化的 game_state 中讀取 remote_target_path。3. 如果都失敗，才返回錯誤。此修改旨在從根本上解決因 scene_analysis 節點暫時性失敗而導致整個遠程流程崩潰的問題。
# v4.0 (2025-09-18): [重大架構重構] 修改了數據源，現在從 state['planning_subjects'] 或 state['raw_lore_objects'] 獲取角色數據。
# v3.0 (2025-09-18): [重大架構升級] 修改了此節點的輸入源，改為使用 `sanitized_user_input`。
async def remote_sfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7D] SFW 描述路徑專用規劃器，生成遠景場景的結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['sanitized_user_input'] or state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|7D) Node: remote_sfw_planning -> 正在基於指令 '{user_input[:50]}...' 生成遠程SFW場景計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}

    # [v5.0 核心修正] 三級備援獲取目標路徑
    scene_analysis = state.get('scene_analysis')
    gs = ai_core.profile.game_state
    target_location_path: Optional[List[str]] = None

    # 1. 優先使用當前回合的分析結果
    if scene_analysis and scene_analysis.target_location_path:
        target_location_path = scene_analysis.target_location_path
        logger.info(f"[{user_id}] (Graph|7D) 已從當前回合分析中獲取遠程目標: {target_location_path}")
    # 2. 如果分析失敗，回退到持久化的 GameState
    elif gs.viewing_mode == 'remote' and gs.remote_target_path:
        target_location_path = gs.remote_target_path
        logger.warning(f"[{user_id}] (Graph|7D) 當前回合分析未提供目標，已從持久化 GameState 中成功回退。目標: {target_location_path}")
    # 3. 如果都失敗，則返回錯誤
    else:
        error_msg = "錯誤：未能從當前回合分析或持久化狀態中解析出遠程觀察的目標地點。"
        logger.error(f"[{user_id}] (Graph|7D) {error_msg}")
        return {"turn_plan": TurnPlan(thought=error_msg, character_actions=[])}

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
            "username": ai_core.profile.user_profile.name,
            "ai_name": ai_core.profile.ai_profile.name
        },
        retry_strategy='euphemize'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：遠程SFW規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}
# 函式：遠程 SFW 規劃節點 (v5.0 - 狀態持久化與三級備援)



# 函式：遠程NSFW規劃節點 (v5.0 - 狀態持久化與三級備援)
# 更新紀錄:
# v5.0 (2025-09-18): [災難性BUG修復] 與 SFW 版本同步，徹底重構了 target_location_path 的獲取邏輯，引入“三級備援”機制，確保在 scene_analysis 節點失敗時，流程依然可以從持久化狀態中獲取目標地點，保證遠程觀察的連續性。
# v4.0 (2025-09-18): [重大架構重構] 修改了數據源，現在從 state['planning_subjects'] 或 state['raw_lore_objects'] 獲取角色數據。
# v3.0 (2025-09-18): [重大架構升級] 修改了此節點的輸入源，改為使用 `sanitized_user_input`。
async def remote_nsfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7C] NSFW描述路徑專用規劃器，生成遠景場景的結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['sanitized_user_input'] or state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|7C) Node: remote_nsfw_planning -> 正在基於指令 '{user_input[:50]}...' 生成遠程NSFW場景計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}
    
    # [v5.0 核心修正] 三級備援獲取目標路徑
    scene_analysis = state.get('scene_analysis')
    gs = ai_core.profile.game_state
    target_location_path: Optional[List[str]] = None

    # 1. 優先使用當前回合的分析結果
    if scene_analysis and scene_analysis.target_location_path:
        target_location_path = scene_analysis.target_location_path
        logger.info(f"[{user_id}] (Graph|7C) 已從當前回合分析中獲取遠程目標: {target_location_path}")
    # 2. 如果分析失敗，回退到持久化的 GameState
    elif gs.viewing_mode == 'remote' and gs.remote_target_path:
        target_location_path = gs.remote_target_path
        logger.warning(f"[{user_id}] (Graph|7C) 當前回合分析未提供目標，已從持久化 GameState 中成功回退。目標: {target_location_path}")
    # 3. 如果都失敗，則返回錯誤
    else:
        error_msg = "錯誤：未能從當前回合分析或持久化狀態中解析出遠程觀察的目標地點。"
        logger.error(f"[{user_id}] (Graph|7C) {error_msg}")
        return {"turn_plan": TurnPlan(thought=error_msg, character_actions=[])}

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
        ai_core.get_remote_nsfw_planning_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str,
            "planning_subjects_json": planning_subjects_json,
            "target_location_path_str": target_location_path_str,
            "user_input": user_input,
            "username": ai_core.profile.user_profile.name,
            "ai_name": ai_core.profile.ai_profile.name
        },
        retry_strategy='force'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：遠程NSFW規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}
# 函式：遠程NSFW規劃節點 (v5.0 - 狀態持久化與三級備援)



# --- 階段三：執行與渲染 (Execution & Rendering) ---

async def tool_execution_node(state: ConversationGraphState) -> Dict[str, str]:
    """[8] 統一的工具執行節點。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    plan = state['turn_plan']
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

# 函式：統一的敘事渲染節點 (v22.0 - 數據流修正)
# 更新紀錄:
# v22.0 (2025-09-05): [災難性BUG修復] 修正了調用鏈時的參數傳遞，補上了缺失的 `system_prompt` 和 `response_style_prompt`。這是實現“指令淹沒”策略、解決最終渲染環節被內容審查攔截的關鍵一步。
# v21.0 (2025-09-12): [架構重構] 強化此鏈，使其成為能夠處理所有類型 TurnPlan (SFW, NSFW, 遠景) 的統一“小說家”節點。
# v204.0 (2025-09-06): [重大功能修正] 賦予敘事鏈在計畫對話不足時，根據風格指令補充對話的權力。
async def narrative_rendering_node(state: ConversationGraphState) -> Dict[str, str]:
    """[9] 統一的敘事渲染節點，將行動計劃轉化為小說文本。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph|9) Node: narrative_rendering -> 正在將行動計劃渲染為小說...")

    if not turn_plan:
        return {"llm_response": "（系統錯誤：未能生成有效的行動計劃。）"}
    if turn_plan.execution_rejection_reason:
        return {"llm_response": turn_plan.execution_rejection_reason}
        
    narrative_text = await ai_core.ainvoke_with_rotation(
        ai_core.get_narrative_chain(),
        {
            # [v22.0 核心修正] 傳入完整的系統指令和風格指令
            "system_prompt": ai_core.profile.one_instruction if ai_core.profile else "預設系統指令",
            "response_style_prompt": ai_core.profile.response_style_prompt if ai_core.profile else "預設風格",
            "turn_plan": turn_plan
        },
        retry_strategy='force'
    )
    if not narrative_text:
        narrative_text = "（AI 在將計劃轉化為故事時遭遇了內容安全限制。）"
    return {"llm_response": narrative_text}
# 函式：統一的敘事渲染節點 (v22.0 - 數據流修正)

# --- 階段四：收尾 (Finalization) ---

async def validate_and_rewrite_node(state: ConversationGraphState) -> Dict:
    """[10] 統一的輸出驗證與淨化節點。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
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
    """[11] 統一的狀態持久化節點。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    clean_response = state['final_output']
    logger.info(f"[{user_id}] (Graph|11) Node: persist_state -> 正在持久化狀態與記憶...")
    
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

# --- 主對話圖的路由 v21.1 ---

def route_expansion_decision(state: ConversationGraphState) -> Literal["expand_lore", "continue_to_planner"]:
    """根據LORE擴展決策，決定是否進入擴展節點。"""
    if state.get("expansion_decision") and state["expansion_decision"].should_expand:
        return "expand_lore"
    else:
        return "continue_to_planner"



# --- 主對話圖的建構器 v21.1 ---








# 函式：創建主回應圖 (v30.0 - 終極路由修正)
# 更新紀錄:
# v30.0 (2025-09-06): [災難性BUG修復] 徹底重構了 `route_to_planner` 路由器的核心邏輯。舊版本會錯誤地將 `nsfw_descriptive` 意圖在 `local` 視角下路由到 SFW 或本地 NSFW 管道。新版本採用“意圖優先”原則：只要意圖包含 'descriptive'，就必定進入遠程分支；只要意圖包含 'nsfw'，就必定進入 NSFW 分支。此修改從根本上解決了 NSFW 描述內容被錯誤鏈處理的問題。
# v29.0 (2025-09-18): [災難性BUG修復] 徹底重構了 `route_to_planner` 路由器的核心邏輯。
# v28.0 (2025-09-18): [重大架構重構] 修正了 LORE 擴展分支的數據流。
def create_main_response_graph() -> StateGraph:
    graph = StateGraph(ConversationGraphState)
    
    # --- 1. 註冊所有節點 ---
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_memories", retrieve_memories_node)
    graph.add_node("query_lore", query_lore_node)
    graph.add_node("assemble_context", assemble_context_node)
    graph.add_node("expansion_decision", expansion_decision_node)
    graph.add_node("lore_expansion", lore_expansion_node)
    graph.add_node("scene_and_action_analysis", scene_and_action_analysis_node)
    graph.add_node("update_viewing_mode", update_viewing_mode_node)
    graph.add_node("sanitize_input", sanitize_input_node)

    graph.add_node("sfw_planning", sfw_planning_node)
    graph.add_node("remote_sfw_planning", remote_sfw_planning_node)
    graph.add_node("remote_nsfw_planning", remote_nsfw_planning_node)
    graph.add_node("nsfw_planning", nsfw_planning_node)

    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("narrative_rendering", narrative_rendering_node)
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)
    
    graph.add_node("planner_junction", lambda state: {})
    
    def prepare_existing_subjects_node(state: ConversationGraphState) -> Dict:
        """如果決定不擴展LORE，則將現有的NPC打包成規劃主體。"""
        lore_objects = state.get('raw_lore_objects', [])
        planning_subjects = [lore.content for lore in lore_objects if lore.category == 'npc_profile']
        logger.info(f"[{state['user_id']}] (Graph) Node: prepare_existing_subjects -> 已將 {len(planning_subjects)} 個現有NPC打包為規劃主體。")
        return {"planning_subjects": planning_subjects}
        
    graph.add_node("prepare_existing_subjects", prepare_existing_subjects_node)


    # --- 2. 定義圖的拓撲結構 ---
    graph.set_entry_point("classify_intent")
    
    graph.add_edge("classify_intent", "scene_and_action_analysis")
    graph.add_edge("scene_and_action_analysis", "update_viewing_mode")
    graph.add_edge("update_viewing_mode", "retrieve_memories")
    
    graph.add_edge("retrieve_memories", "query_lore")
    graph.add_edge("query_lore", "assemble_context")
    graph.add_edge("assemble_context", "expansion_decision")
    
    graph.add_conditional_edges(
        "expansion_decision",
        route_expansion_decision,
        { 
            "expand_lore": "lore_expansion", 
            "continue_to_planner": "prepare_existing_subjects"
        }
    )
    graph.add_edge("lore_expansion", "sanitize_input")
    graph.add_edge("prepare_existing_subjects", "sanitize_input")
    
    graph.add_edge("sanitize_input", "planner_junction")

    # [v30.0 核心修正] 重寫主規劃器路由，採用“意圖優先”原則
    def route_to_planner(state: ConversationGraphState) -> str:
        """根據意圖分類將流量路由到不同的規劃器，意圖優先於視角模式。"""
        user_id = state['user_id']
        intent = state['intent_classification'].intent_type
        
        logger.info(f"[{user_id}] (Router) Routing to planner. Intent: '{intent}'")
        
        # 意圖優先的路由邏輯
        if intent == 'nsfw_descriptive':
            logger.info(f"[{user_id}] (Router) Intent is 'nsfw_descriptive' -> remote_nsfw_planner")
            return "remote_nsfw_planner"
        elif intent == 'nsfw_interactive':
            logger.info(f"[{user_id}] (Router) Intent is 'nsfw_interactive' -> nsfw_planner")
            return "nsfw_planner"
        else: # sfw
            # 對於 SFW，我們仍然可以參考視角模式
            ai_core = state['ai_core']
            viewing_mode = ai_core.profile.game_state.viewing_mode if ai_core.profile else 'local'
            if viewing_mode == 'remote':
                logger.info(f"[{user_id}] (Router) Intent is SFW & Mode is remote -> remote_sfw_planner")
                return "remote_sfw_planner"
            else:
                logger.info(f"[{user_id}] (Router) Intent is SFW & Mode is local -> sfw_planner")
                return "sfw_planner"

    graph.add_conditional_edges(
        "planner_junction",
        route_to_planner,
        { 
            "sfw_planner": "sfw_planning", 
            "nsfw_planner": "nsfw_planning",
            "remote_sfw_planner": "remote_sfw_planning",
            "remote_nsfw_planner": "remote_nsfw_planning"
        }
    )
    
    graph.add_edge("sfw_planning", "tool_execution")
    graph.add_edge("remote_sfw_planning", "tool_execution")
    graph.add_edge("remote_nsfw_planning", "tool_execution")
    graph.add_edge("nsfw_planning", "tool_execution")
    
    graph.add_edge("tool_execution", "narrative_rendering")
    graph.add_edge("narrative_rendering", "validate_and_rewrite")
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", END)
    
    return graph.compile()
# 函式：創建主回應圖 (v30.0 - 終極路由修正)




# --- 設定圖 (Setup Graph) 的節點與建構器 (完整版) ---

async def process_canon_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
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
    user_id = state['user_id']
    ai_core = state['ai_core']
    opening_scene = await ai_core.generate_opening_scene()
    
    if not opening_scene or not opening_scene.strip():
        opening_scene = (f"在一片柔和的光芒中，你和 {ai_core.profile.ai_profile.name} 發現自己身處於一個寧靜的空間裡...")
        
    return {"opening_scene": opening_scene}

def create_setup_graph() -> StateGraph:
    """
    創建設定圖
    """
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
# 函式：創建設定圖
