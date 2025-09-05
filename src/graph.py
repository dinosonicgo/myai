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


# 函式：場景與動作分析節點
# 更新紀錄:
# v1.0 (2025-09-13): [恢復] 恢复在重构中被遗漏的 SFW 路径核心节点，用于判断本地/远程视角。
async def scene_and_action_analysis_node(state: ConversationGraphState) -> Dict:
    """[SFW Path] 專用節點，分析 SFW 場景的視角（本地 vs 遠程）。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: scene_and_action_analysis [SFW Path] -> 正在進行場景視角分析...")

    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph) 在 scene_and_action_analysis 中 ai_core.profile 未加載。")
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='錯誤：AI profile 未加載。', action_summary=user_input)}

    current_location_path = ai_core.profile.game_state.location_path
    scene_analysis = await ai_core.ainvoke_with_rotation(
        ai_core.get_scene_analysis_chain(),
        {"user_input": user_input, "current_location_path_str": " > ".join(current_location_path)},
        retry_strategy='euphemize'
    )
    if not scene_analysis:
        logger.warning(f"[{user_id}] (Graph) 場景分析鏈委婉化重試失敗，啟動安全備援。")
        scene_analysis = SceneAnalysisResult(
            viewing_mode='local', 
            reasoning='安全備援：場景分析鏈失敗。', 
            action_summary=user_input
        )
    return {"scene_analysis": scene_analysis}
# 函式：場景與動作分析節點


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

async def assemble_context_node(state: ConversationGraphState) -> Dict:
    """[4] 專用上下文組裝節點，將原始LORE格式化為LLM可讀的字符串。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    raw_lore = state['raw_lore_objects']
    intent_type = state['intent_classification'].intent_type
    logger.info(f"[{user_id}] (Graph|4) Node: assemble_context -> 正在組裝最終上下文簡報...")
    
    is_remote = intent_type == 'nsfw_descriptive'
    # 這是 ai_core.py 中為新節點準備的輔助函式
    structured_context = ai_core._assemble_context_from_lore(raw_lore, is_remote_scene=is_remote)
    return {"structured_context": structured_context}
    
async def expansion_decision_node(state: ConversationGraphState) -> Dict:
    """[5] LORE擴展決策節點。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|5) Node: expansion_decision -> 正在判斷是否需要擴展LORE...")
    
    chat_history_manager = ai_core.session_histories.get(user_id, ChatMessageHistory())
    recent_dialogue = "\n".join([f"{'使用者' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in chat_history_manager.messages[-6:]])
    decision = await ai_core.ainvoke_with_rotation(
        ai_core.get_expansion_decision_chain(), 
        {"user_input": user_input, "recent_dialogue": recent_dialogue},
        retry_strategy='euphemize'
    )
    if not decision:
        logger.warning(f"[{user_id}] (Graph|5) LORE擴展決策鏈失敗，安全備援為不擴展。")
        decision = ExpansionDecision(should_expand=False, reasoning="安全備援：決策鏈失敗。")
    
    logger.info(f"[{user_id}] (Graph|5) LORE擴展決策: {decision.should_expand}。理由: {decision.reasoning}")
    return {"expansion_decision": decision}

async def lore_expansion_node(state: ConversationGraphState) -> Dict:
    """[6A] 專用的LORE擴展執行節點，執行選角並刷新上下文。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|6A) Node: lore_expansion -> 正在執行場景選角...")
    
    current_location_path = ai_core.profile.game_state.location_path if ai_core.profile else []
    game_context_for_casting = json.dumps(state.get('structured_context', {}), ensure_ascii=False, indent=2)
    
    cast_result = await ai_core.ainvoke_with_rotation(
        ai_core.get_scene_casting_chain(),
        {"world_settings": ai_core.profile.world_settings or "", "current_location_path": current_location_path, "game_context": game_context_for_casting, "recent_dialogue": user_input},
        retry_strategy='euphemize'
    )
    
    updates: Dict[str, Any] = {}
    if cast_result and (cast_result.newly_created_npcs or cast_result.supporting_cast):
        await ai_core._add_cast_to_scene(cast_result)
        logger.info(f"[{user_id}] (Graph|6A) 選角完成，正在刷新LORE和上下文...")
        intent_type = state['intent_classification'].intent_type
        is_remote = intent_type == 'nsfw_descriptive'
        refreshed_lore = await ai_core._query_lore_from_entities(user_input, is_remote_scene=is_remote)
        refreshed_context = ai_core._assemble_context_from_lore(refreshed_lore, is_remote_scene=is_remote)
        updates = {"raw_lore_objects": refreshed_lore, "structured_context": refreshed_context}
    else:
         logger.info(f"[{user_id}] (Graph|6A) 場景選角鏈未返回新角色，無需刷新。")

    return updates

# --- 階段二：規劃 (Planning) ---

# 函式：SFW規劃節點 (v21.1 - 數據流修正)
# 更新紀錄:
# v21.1 (2025-09-12): [災難性BUG修復] 重構了 `full_context_dict` 的構建方式，從隱式的 model_dump() 改為手動明確賦值。此修改確保了模板所需的所有鍵（特別是 `ai_settings`）都被正確提供，從根本上解決了 KeyError。
async def sfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7A] SFW路徑專用規劃器，生成結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|7A) Node: sfw_planning -> 正在生成SFW行動計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}

    # SFW路徑需要額外的風格和場景分析
    style_analysis_chain = ai_core.get_style_analysis_chain()
    style_result = await ai_core.ainvoke_with_rotation(style_analysis_chain, {"user_input": state['messages'][-1].content, "response_style_prompt": ai_core.profile.response_style_prompt or ""}, retry_strategy='euphemize')
    if not style_result:
        style_result = StyleAnalysisResult(dialogue_requirement="AI角色應做出回應。", narration_level="中等", proactive_suggestion=None)

    # [v21.1 核心修正] 手動、明確地構建上下文辭典，確保所有鍵都存在
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
    
    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_sfw_planning_chain(), 
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "world_snapshot": world_snapshot, 
            "user_input": state['messages'][-1].content, 
            "style_analysis": style_result.model_dump_json()
        },
        retry_strategy='euphemize'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：SFW規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}
# 函式：SFW規劃節點 (v21.1 - 數據流修正)

# 函式：NSFW規劃節點 (v21.1 - 數據流修正)
# 更新紀錄:
# v21.1 (2025-09-12): [災難性BUG修復] 與 sfw_planning_node 同步，重構了 `full_context_dict` 的構建方式，改為手動明確賦值，解決了 KeyError。
async def nsfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7B] NSFW互動路徑專用規劃器，生成結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|7B) Node: nsfw_planning -> 正在生成NSFW互動行動計劃...")
    
    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}

    # [v21.1 核心修正] 手動、明確地構建上下文辭典，確保所有鍵都存在
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

    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_nsfw_planning_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "world_snapshot": world_snapshot, 
            "user_input": state['messages'][-1].content
        },
        retry_strategy='force'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：NSFW規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}
# 函式：NSFW規劃節點 (v21.1 - 數據流修正)


# 函式：遠程 SFW 規劃節點 (v1.1 - 接收場景分析)
# 更新紀錄:
# v1.1 (2025-09-14): [架構重構] 修改此節點，使其能接收並使用上游 `scene_and_action_analysis_node` 解析出的 `target_location_path`，確保地点正确。
# v1.0 (2025-09-13): [架構重構] 新增此節點。
async def remote_sfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7D] SFW 描述路徑專用規劃器，生成遠景場景的結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|7D) Node: remote_sfw_planning -> 正在生成遠程SFW場景計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}

    # [v1.1 核心修正] 從 state 中獲取場景分析結果
    scene_analysis = state.get('scene_analysis')
    if not scene_analysis or not scene_analysis.target_location_path:
        logger.error(f"[{user_id}] (Graph|7D) 錯誤：進入 remote_sfw_planning_node 但未找到 target_location_path。")
        return {"turn_plan": TurnPlan(thought="錯誤：未能解析出遠程觀察的目標地點。", character_actions=[])}
    
    target_location_path_str = " > ".join(scene_analysis.target_location_path)

    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_remote_sfw_planning_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "world_settings": ai_core.profile.world_settings, 
            "target_location_path_str": target_location_path_str,
            "remote_scene_context": json.dumps(state['structured_context'], ensure_ascii=False), 
            "user_input": state['messages'][-1].content,
            "username": ai_core.profile.user_profile.name,
            "ai_name": ai_core.profile.ai_profile.name
        },
        retry_strategy='euphemize'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：遠程SFW規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}
# 函式：遠程 SFW 規劃節點 (v1.1 - 接收場景分析)




# 函式：遠程NSFW規劃節點 (v21.1 - 接收場景分析)
# 更新紀錄:
# v21.1 (2025-09-14): [架構重構] 修改此節點，使其能接收並使用上游 `scene_and_action_analysis_node` 解析出的 `target_location_path`，確保地点正确。
async def remote_nsfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7C] NSFW描述路徑專用規劃器，生成遠景場景的結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|7C) Node: remote_nsfw_planning -> 正在生成遠程NSFW場景計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}

    # [v21.1 核心修正] 從 state 中獲取場景分析結果
    scene_analysis = state.get('scene_analysis')
    if not scene_analysis or not scene_analysis.target_location_path:
        logger.error(f"[{user_id}] (Graph|7C) 錯誤：進入 remote_nsfw_planning_node 但未找到 target_location_path。")
        return {"turn_plan": TurnPlan(thought="錯誤：未能解析出遠程觀察的目標地點。", character_actions=[])}

    target_location_path_str = " > ".join(scene_analysis.target_location_path)

    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_remote_nsfw_planning_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "world_settings": ai_core.profile.world_settings,
            "target_location_path_str": target_location_path_str,
            "remote_scene_context": json.dumps(state['structured_context'], ensure_ascii=False), 
            "user_input": state['messages'][-1].content,
            "username": ai_core.profile.user_profile.name,
            "ai_name": ai_core.profile.ai_profile.name
        },
        retry_strategy='force'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：遠程NSFW規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}
# 函式：遠程NSFW規劃節點 (v21.1 - 接收場景分析)

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
        {"turn_plan": turn_plan},
        retry_strategy='force'
    )
    if not narrative_text:
        narrative_text = "（AI 在將計劃轉化為故事時遭遇了內容安全限制。）"
    return {"llm_response": narrative_text}

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








# 函式：創建主回應圖 (v21.5 - 拓撲最終修正)
# 更新紀錄:
# v21.5 (2025-09-14): [災難性BUG修復] 彻底修复了因路由器返回值与节点名不匹配导致的 `Unknown node` 编译错误。
#    1. [內聯路由邏輯] 将所有路由器的定义内联到 `create_main_response_graph` 内部，以提高代码内聚性并避免 NameError。
#    2. [修正路由出口] 修正了 `route_after_perception` 的返回值，确保其 `interactive_planner` 出口正确指向已注册的 `planner_junction` 节点。
#    3. [简化拓扑] 删除了一个多余的路由器，使整个图的拓扑结构更清晰、更健壮。
# v21.4 (2025-09-14): [重大架構修正] 再次重构路由拓撲。
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
    graph.add_node("sfw_planning", sfw_planning_node)
    graph.add_node("nsfw_planning", nsfw_planning_node)
    graph.add_node("remote_nsfw_planning", remote_nsfw_planning_node)
    graph.add_node("remote_sfw_planning", remote_sfw_planning_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("narrative_rendering", narrative_rendering_node)
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)
    
    # 匯合點 (Junctions)
    graph.add_node("after_perception_junction", lambda state: {})
    graph.add_node("planner_junction", lambda state: {})

    # --- 2. 定義圖的拓撲結構 ---
    graph.set_entry_point("classify_intent")
    
    # 感知流程
    graph.add_edge("classify_intent", "retrieve_memories")
    graph.add_edge("retrieve_memories", "query_lore")
    graph.add_edge("query_lore", "assemble_context")
    graph.add_edge("assemble_context", "expansion_decision")
    
    # LORE擴展分支
    graph.add_conditional_edges(
        "expansion_decision",
        route_expansion_decision,
        { "expand_lore": "lore_expansion", "continue_to_planner": "after_perception_junction" }
    )
    graph.add_edge("lore_expansion", "after_perception_junction")

    # [v21.5 核心修正] 主路由，區分互動和描述
    def route_after_perception(state: ConversationGraphState) -> str:
        intent = state['intent_classification'].intent_type
        if 'descriptive' in intent:
            return "analyze_scene"
        else:
            # 修正：返回值 'interactive_planner' 必须指向一个已注册的节点
            return "interactive_planner_entry"

    graph.add_conditional_edges(
        "after_perception_junction",
        route_after_perception,
        { 
            "analyze_scene": "scene_and_action_analysis", 
            "interactive_planner_entry": "planner_junction" # 指向已注册的汇合点
        }
    )

    # 描述性路徑的路由 (SFW vs NSFW)
    def route_descriptive_planner(state: ConversationGraphState) -> str:
        intent = state['intent_classification'].intent_type
        # SFW 路径中的远程路由
        if state.get('scene_analysis') and state['scene_analysis'].viewing_mode == 'remote':
             return "remote_sfw_planner"
        # NSFW 描述性路由
        if intent == 'nsfw_descriptive':
            return "remote_nsfw_planner"
        # 默认或本地 SFW 路由
        return "local_sfw_planner"

    graph.add_conditional_edges(
        "scene_and_action_analysis",
        route_viewing_mode, # 使用这个路由器来决定是本地还是远程
        {
            "remote_scene": "remote_sfw_planning",
            "local_scene": "sfw_planning" 
        }
    )

    # 互動性路徑的路由 (SFW vs NSFW)
    def route_interactive_planner(state: ConversationGraphState) -> str:
        intent = state['intent_classification'].intent_type
        if intent == 'nsfw_interactive':
            return "nsfw_planner"
        else: # sfw (interactive)
            return "sfw_planner"

    graph.add_conditional_edges(
        "planner_junction",
        route_interactive_planner,
        { 
            "sfw_planner": "sfw_planning", 
            "nsfw_planner": "nsfw_planning"
        }
    )

    # 所有規劃器都匯合到工具執行
    graph.add_edge("sfw_planning", "tool_execution")
    graph.add_edge("nsfw_planning", "tool_execution")
    graph.add_edge("remote_nsfw_planning", "tool_execution")
    graph.add_edge("remote_sfw_planning", "tool_execution")
    
    # 執行與渲染流程
    graph.add_edge("tool_execution", "narrative_rendering")
    graph.add_edge("narrative_rendering", "validate_and_rewrite")
    
    # 收尾流程
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", END)
    
    return graph.compile()
# 函式：創建主回應圖 (v21.5 - 拓撲最終修正)



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
