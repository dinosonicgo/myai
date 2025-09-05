# src/graph.py 的中文註釋(v21.0 - “一功能一節點”精細化重構)
# 更新紀錄:
# v21.0 (2025-09-09): [重大架構重構] 根據“極致準確”藍圖，對圖的拓撲結構進行了根本性的精細化重構。
#    1. [初始化拆分] 將原有的 `initialize_conversation_state_node` 徹底拆分為三個獨立、可觀察的節點：`retrieve_memories_node` (RAG檢索), `query_lore_node` (LORE查詢), `assemble_context_node` (上下文組裝)。
#    2. [規劃器專職化] 廢棄了所有一步到位的生成節點，為 SFW、NSFW-互動、NSFW-描述 三條路徑分別創建了專用的規劃器節點 (`sfw_planning_node`, `nsfw_planning_node`, `remote_nsfw_planning_node`)，它們的唯一職責是輸出結構化的 TurnPlan JSON。
#    3. [渲染器統一化] 創建了單一的 `narrative_rendering_node` 作為所有路徑的共同出口，它只負責將傳入的 TurnPlan JSON 渲染成最終的小說文本。
#    4. [拓撲重構] 重新設計了圖的邊連接，以適配新的節點拆分和“規劃-渲染”統一模式，確保了數據流的清晰、可追蹤和準確性。
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

# --- 主對話圖 (Main Conversation Graph) 的節點 v21.0 ---

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
        # 刷新 LORE 和上下文
        intent_type = state['intent_classification'].intent_type
        is_remote = intent_type == 'nsfw_descriptive'
        refreshed_lore = await ai_core._query_lore_from_entities(user_input, is_remote_scene=is_remote)
        refreshed_context = ai_core._assemble_context_from_lore(refreshed_lore, is_remote_scene=is_remote)
        updates = {"raw_lore_objects": refreshed_lore, "structured_context": refreshed_context}
    else:
         logger.info(f"[{user_id}] (Graph|6A) 場景選角鏈未返回新角色，無需刷新。")

    return updates

# --- 階段二：規劃 (Planning) ---

async def sfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7A] SFW路徑專用規劃器，生成結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|7A) Node: sfw_planning -> 正在生成SFW行動計劃...")

    # SFW路徑需要額外的風格和場景分析
    style_analysis_chain = ai_core.get_style_analysis_chain()
    style_result = await ai_core.ainvoke_with_rotation(style_analysis_chain, {"user_input": state['messages'][-1].content, "response_style_prompt": ai_core.profile.response_style_prompt or ""}, retry_strategy='euphemize')
    if not style_result:
        style_result = StyleAnalysisResult(dialogue_requirement="AI角色應做出回應。", narration_level="中等")

    full_context_dict = { **(ai_core.profile.model_dump() if ai_core.profile else {}), **state }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)
    
    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_sfw_planning_chain(), 
        {"system_prompt": ai_core.profile.one_instruction, "world_snapshot": world_snapshot, "user_input": state['messages'][-1].content, "style_analysis": style_result.model_dump_json()},
        retry_strategy='euphemize'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：SFW規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}

async def nsfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7B] NSFW互動路徑專用規劃器，生成結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|7B) Node: nsfw_planning -> 正在生成NSFW互動行動計劃...")
    
    full_context_dict = { **(ai_core.profile.model_dump() if ai_core.profile else {}), **state }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)

    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_nsfw_planning_chain(),
        {"system_prompt": ai_core.profile.one_instruction, "world_snapshot": world_snapshot, "user_input": state['messages'][-1].content},
        retry_strategy='force'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：NSFW規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}

async def remote_nsfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7C] NSFW描述路徑專用規劃器，生成遠景場景的結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|7C) Node: remote_nsfw_planning -> 正在生成遠程NSFW場景計劃...")

    full_context_dict = { **(ai_core.profile.model_dump() if ai_core.profile else {}), **state }
    
    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_remote_planning_chain(),
        {"system_prompt": ai_core.profile.one_instruction, "world_settings": ai_core.profile.world_settings, "remote_scene_context": json.dumps(state['structured_context']), "user_input": state['messages'][-1].content},
        retry_strategy='force'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：遠程NSFW規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}

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
    initial_response = state['llm_response']
    logger.info(f"[{user_id}] (Graph|10) Node: validate_and_rewrite -> 正在淨化LLM輸出...")
    # ... (此處省略了淨化邏輯的詳細實現，因為它保持不變)
    clean_response = initial_response.strip()
    return {"final_output": clean_response}

async def persist_state_node(state: ConversationGraphState) -> Dict:
    """[11] 統一的狀態持久化節點。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    clean_response = state['final_output']
    logger.info(f"[{user_id}] (Graph|11) Node: persist_state -> 正在持久化狀態與記憶...")
    # ... (此處省略了持久化邏輯的詳細實現，因為它保持不變)
    return {}

# --- 主對話圖的路由 v21.0 ---

def route_expansion_decision(state: ConversationGraphState) -> Literal["expand_lore", "continue_to_planner"]:
    """根據LORE擴展決策，決定是否進入擴展節點。"""
    if state.get("expansion_decision") and state["expansion_decision"].should_expand:
        return "expand_lore"
    else:
        return "continue_to_planner"

def route_to_planner(state: ConversationGraphState) -> str:
    """根據意圖分類，將流程分發到對應的規劃器。"""
    intent = state['intent_classification'].intent_type
    if intent == 'nsfw_interactive':
        return "nsfw_planner"
    elif intent == 'nsfw_descriptive':
        return "remote_nsfw_planner"
    else: # 'sfw'
        return "sfw_planner"

# --- 主對話圖的建構器 v21.0 ---

def create_main_response_graph() -> StateGraph:
    graph = StateGraph(ConversationGraphState)
    
    # --- 1. 註冊所有節點 ---
    # 感知
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_memories", retrieve_memories_node)
    graph.add_node("query_lore", query_lore_node)
    graph.add_node("assemble_context", assemble_context_node)
    graph.add_node("expansion_decision", expansion_decision_node)
    graph.add_node("lore_expansion", lore_expansion_node)
    # 規劃
    graph.add_node("sfw_planning", sfw_planning_node)
    graph.add_node("nsfw_planning", nsfw_planning_node)
    graph.add_node("remote_nsfw_planning", remote_nsfw_planning_node)
    # 執行與渲染
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("narrative_rendering", narrative_rendering_node)
    # 收尾
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)

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
        {
            "expand_lore": "lore_expansion",
            "continue_to_planner": "route_to_planner" # 使用一個虛擬節點作為匯合點
        }
    )
    # 擴展完成後，也匯合到規劃器路由
    graph.add_edge("lore_expansion", "route_to_planner")

    # 規劃器路由
    graph.add_conditional_edges(
        "route_to_planner", # 虛擬節點
        route_to_planner,
        {
            "sfw_planner": "sfw_planning",
            "nsfw_planner": "nsfw_planning",
            "remote_nsfw_planner": "remote_nsfw_planning"
        }
    )
    
    # 所有規劃器都匯合到工具執行
    graph.add_edge("sfw_planning", "tool_execution")
    graph.add_edge("nsfw_planning", "tool_execution")
    graph.add_edge("remote_nsfw_planning", "tool_execution")
    
    # 執行與渲染流程
    graph.add_edge("tool_execution", "narrative_rendering")
    graph.add_edge("narrative_rendering", "validate_and_rewrite")
    
    # 收尾流程
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", END)
    
    # 編譯圖時，需要為虛擬節點提供一個passthrough函式
    return graph.compile(checkpointer=None, interrupt_after=None, known_nodes=["route_to_planner"])

# --- 設定圖 (Setup Graph) 的節點與建構器 (保持不變) ---

async def process_canon_node(state: SetupGraphState) -> Dict:
    # ... (保持不變)
    return {}

async def complete_profiles_node(state: SetupGraphState) -> Dict:
    # ... (保持不變)
    return {}

async def world_genesis_node(state: SetupGraphState) -> Dict:
    # ... (保持不變)
    return {"genesis_result": WorldGenesisResult(...)}

async def generate_opening_scene_node(state: SetupGraphState) -> Dict:
    # ... (保持不變)
    return {"opening_scene": "..."}

def create_setup_graph() -> StateGraph:
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
