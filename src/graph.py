# src/graph.py 的中文註釋(v16.0 - 風格分析節點架構)
# 更新紀錄:
# v16.0 (2025-09-06): [災難性BUG修復] 為了解決 SFW 路徑中模型頑固地忽略風格指令（特別是“對話”要求）的問題，進行了根本性的架構重構。
#    1. [新增風格分析節點] 引入了全新的 `style_analysis_node`。此節點專門負責將用戶冗長的風格 Prompt 提煉成結構化的、給規劃器的具體硬性指令。
#    2. [重構SFW路徑] 將新節點插入到 `planning_node` 之前，確保在規劃前必須先完成風格分析。
#    3. [強化規劃節點] `planning_node` 現在接收結構化的風格指令作為最高優先級輸入，從根本上解決了“指令稀釋”問題。
# v15.0 (2025-09-06): [災難性BUG修復] 根據“先檢測，後處理”原則，對圖的拓撲進行了根本性的重構。

import sys
print(f"[DEBUG] graph.py loaded from: {__file__}", file=sys.stderr)
import asyncio
import json
import re
from typing import Dict, List, Literal, Optional

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

# --- 主對話圖 (Main Conversation Graph) 的節點 ---

async def classify_intent_node(state: ConversationGraphState) -> Dict:
    """圖的新入口點，唯一職責是對原始輸入進行意圖分類。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: classify_intent_node -> 正在對 '{user_input[:30]}...' 進行意圖分類...")
    
    classification_chain = ai_core.get_intent_classification_chain()
    classification_result = await ai_core.ainvoke_with_rotation(
        classification_chain,
        {"user_input": user_input},
        retry_strategy='none'
    )
    
    if not classification_result:
        logger.warning(f"[{user_id}] (Graph) 意圖分類鏈失敗，啟動安全備援，預設為 SFW。")
        classification_result = IntentClassificationResult(intent_type='sfw', reasoning="安全備援：分類鏈失敗。")
        
    return {"intent_classification": classification_result}

async def initialize_conversation_state_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: initialize_conversation_state_node [SFW Path] -> 正在為 '{user_input[:30]}...' 初始化狀態...")
    
    rag_task = ai_core.ainvoke_with_rotation(ai_core.retriever, user_input, retry_strategy='euphemize')
    structured_context_task = ai_core._get_structured_context(user_input)
    
    retrieved_docs, structured_context = await asyncio.gather(rag_task, structured_context_task)
    
    if retrieved_docs is None:
        logger.warning(f"[{user_id}] RAG 檢索返回 None (可能因委婉化失敗)，使用空列表作為備援。")
        retrieved_docs = []
        
    rag_context_str = await ai_core._preprocess_rag_context(retrieved_docs)
    return {"structured_context": structured_context, "rag_context": rag_context_str}

async def analyze_input_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: analyze_input_node [SFW Path] -> 正在分析輸入意圖...")
    analysis = await ai_core.ainvoke_with_rotation(
        ai_core.get_input_analysis_chain(), 
        {"user_input": user_input},
        retry_strategy='euphemize'
    )
    if not analysis:
        logger.warning(f"[{user_id}] (Graph) 輸入分析鏈委婉化重試失敗，啟動安全備援。")
        analysis = UserInputAnalysis(
            input_type='dialogue_or_command', 
            summary_for_planner=user_input, 
            narration_for_turn=""
        )
    return {"input_analysis": analysis}

async def expansion_decision_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: expansion_decision_node [SFW Path] -> 正在判斷是否需要擴展LORE...")
    chat_history_manager = ai_core.session_histories.get(user_id, ChatMessageHistory())
    recent_dialogue = "\n".join([f"{'使用者' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in chat_history_manager.messages[-6:]])
    decision = await ai_core.ainvoke_with_rotation(
        ai_core.get_expansion_decision_chain(), 
        {"user_input": user_input, "recent_dialogue": recent_dialogue},
        retry_strategy='euphemize'
    )
    if not decision:
        logger.warning(f"[{user_id}] (Graph) LORE擴展決策鏈委婉化重試失敗，啟動安全備援。")
        decision = ExpansionDecision(
            should_expand=False,
            reasoning="安全備援：決策鏈未能返回有效結果。"
        )
    logger.info(f"[{user_id}] (Graph) LORE擴展決策: {decision.should_expand}。理由: {decision.reasoning}")
    return {"expansion_decision": decision}

async def scene_and_action_analysis_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: scene_and_action_analysis_node [SFW Path] -> 正在進行場景視角分析與潛在選角...")
    current_location_path = ai_core.profile.game_state.location_path if ai_core.profile else []
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
    if scene_analysis.viewing_mode == 'local':
        logger.info(f"[{user_id}] (Graph) ...視角為本地，繼續執行選角流程。")
        game_context_for_casting = json.dumps(state.get('structured_context', {}), ensure_ascii=False, indent=2)
        cast_result = await ai_core.ainvoke_with_rotation(
            ai_core.get_scene_casting_chain(),
            {
                "world_settings": ai_core.profile.world_settings or "",
                "current_location_path": current_location_path, 
                "game_context": game_context_for_casting,
                "recent_dialogue": user_input
            },
            retry_strategy='euphemize'
        )
        if cast_result:
            new_npc_names = await ai_core._add_cast_to_scene(cast_result)
            if new_npc_names:
                final_structured_context = await ai_core._get_structured_context(user_input, override_location_path=current_location_path)
                return {"scene_analysis": scene_analysis, "structured_context": final_structured_context}
        else:
             logger.warning(f"[{user_id}] (Graph) 場景選角鏈委婉化重試失敗，本輪跳過選角。")
    return {"scene_analysis": scene_analysis}

# [v16.0 新增] 風格分析節點
async def style_analysis_node(state: ConversationGraphState) -> Dict:
    """分析用戶的風格指令，並將其轉化為給規劃器的結構化硬性指令。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: style_analysis_node [SFW Path] -> 正在分析回應風格...")
    
    style_analysis_chain = ai_core.get_style_analysis_chain()
    style_result = await ai_core.ainvoke_with_rotation(
        style_analysis_chain,
        {
            "user_input": user_input,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格"
        },
        retry_strategy='euphemize'
    )
    
    if not style_result:
        logger.warning(f"[{user_id}] (Graph) 風格分析鏈委婉化重試失敗，啟動安全備援。")
        style_result = StyleAnalysisResult(
            dialogue_requirement="AI角色必須至少說一句符合其性格和情境的話",
            narration_level="中等",
            proactive_suggestion="可以嘗試描述周圍的環境來回應。"
        )
        
    return {"style_analysis": style_result}

# [v16.0 重構] 規劃節點現在接收結構化的風格指令
async def planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    style_analysis = state['style_analysis']
    logger.info(f"[{user_id}] (Graph) Node: planning_node [SFW Path] -> 正在根據具體風格指令進行規劃...")
    
    # 將結構化的風格指令注入到一個新的 Prompt 模板中
    planning_system_prompt_template = """
# ==============================================================================
# == 🧠 系統核心指令 (規劃模式) 🧠
# ==============================================================================
# 說明：你是一位擁有極高智慧和創造力的遊戲管理大師（GM）。你的任務是分析所有
#       上下文，為接下來的回合制定一份詳細、合理的【行動計劃 JSON】。
#
# **【【【最高規劃原則：絕對遵守風格指令】】】**
# 你的首要且最高優先級的任務是，生成一個**嚴格且完全**滿足下方【具體風格指令】的 JSON 計畫。
# 這些指令不是建議，而是你必須執行的**硬性約束**。
# ------------------------------------------------------------------------------
# **【具體風格指令 (硬性約束)】**
# - **對話要求**: {dialogue_requirement}
# - **旁白詳細度**: {narration_level}
# - **可選行動建議**: {proactive_suggestion}
# ------------------------------------------------------------------------------
{system_prompt}
"""
    # 填充風格指令
    style_driven_prompt = planning_system_prompt_template.format(
        dialogue_requirement=style_analysis.dialogue_requirement,
        narration_level=style_analysis.narration_level,
        proactive_suggestion=style_analysis.proactive_suggestion or "無",
        system_prompt=ai_core.profile.one_instruction or ""
    )

    # 填充其餘上下文
    structured_context = state.get('structured_context', {})
    full_context_dict = {
        "username": ai_core.profile.user_profile.name,
        "ai_name": ai_core.profile.ai_profile.name,
        "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
        "world_settings": ai_core.profile.world_settings or "未設定",
        "ai_settings": ai_core.profile.ai_profile.description or "未設定",
        "retrieved_context": state['rag_context'],
        "user_input": user_input,
        "latest_user_input": user_input,
        **structured_context
    }
    def safe_format(template: str, data: dict) -> str:
        for key, value in data.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template
    
    final_system_prompt = safe_format(style_driven_prompt, full_context_dict)
    world_snapshot = safe_format(ai_core.world_snapshot_template, full_context_dict)
    
    params_for_chain = {
        "system_prompt": final_system_prompt,
        "world_snapshot": world_snapshot,
        "user_input": user_input,
    }
    
    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_planning_chain(), 
        params_for_chain,
        retry_strategy='euphemize'
    )
    
    if not plan:
        logger.error(f"[{user_id}] SFW 規劃鏈即使在委婉化後也最終失敗。")
        plan = TurnPlan(thought="安全備援：規劃鏈失敗。", character_actions=[
            CharacterAction(
                character_name=ai_core.profile.ai_profile.name,
                reasoning="規劃鏈失敗，需要給出一個安全的備用回應。",
                dialogue="抱歉，主人，我的思緒有些混亂，可以請您換一種方式說嗎？"
            )
        ])
    return {"turn_plan": plan, "world_snapshot": world_snapshot}

async def generate_nsfw_response_node(state: ConversationGraphState) -> Dict[str, str]:
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: generate_nsfw_response_node [NSFW Path] -> 正在直接生成【互動式】NSFW 回應...")
    structured_context = await ai_core._get_structured_context(user_input)
    full_context_for_chain = {
        "user_input": user_input,
        "retrieved_context": "（NSFW 路徑不執行深度記憶檢索）",
        **structured_context
    }
    response_text = await ai_core.ainvoke_with_rotation(
        ai_core.get_direct_nsfw_chain(),
        full_context_for_chain,
        retry_strategy='force'
    )
    if not response_text:
        response_text = "（AI 在處理您的請求時遭遇了不可恢復的內容安全限制，請嘗試調整您的指令。）"
    return {"llm_response": response_text}

async def remote_scene_generation_node(state: ConversationGraphState) -> Dict[str, str]:
    user_id = state['user_id']
    ai_core = state['ai_core']
    scene_analysis = state['scene_analysis']
    if not (scene_analysis and scene_analysis.target_location_path):
        logger.error(f"[{user_id}] 進入了 remote_scene_generation_node，但 scene_analysis 中沒有 target_location_path。")
        return {"llm_response": "（系統錯誤：無法確定要觀察的遠程目標。）"}
    target_path = scene_analysis.target_location_path
    logger.info(f"[{user_id}] (Graph) Node: remote_scene_generation_node [SFW Path] -> 正在為遠程地點 '{' > '.join(target_path)}' 生成【SFW】場景...")
    remote_context = await ai_core._get_structured_context(
        user_input="", 
        override_location_path=target_path
    )
    remote_scene_context_str = "\n".join([f"【{k.replace('_context', '').title()}】\n{v}" for k, v in remote_context.items()])
    scene_text = await ai_core.ainvoke_with_rotation(
        ai_core.get_remote_scene_generator_chain(),
        {
            "username": ai_core.profile.user_profile.name,
            "ai_name": ai_core.profile.ai_profile.name,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格：平衡的敘事與對話。",
            "world_settings": ai_core.profile.world_settings or "未設定",
            "remote_scene_context": remote_scene_context_str,
        },
        retry_strategy='euphemize'
    )
    if not scene_text:
        scene_text = "（由於內容限制，無法生成遠程場景的詳細描述。）"
    return {"llm_response": scene_text}

async def remote_nsfw_scene_generator_node(state: ConversationGraphState) -> Dict[str, str]:
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: remote_nsfw_scene_generator_node [NSFW Path] -> 正在生成【遠程NSFW】場景...")
    remote_context = await ai_core._get_structured_context(user_input)
    remote_scene_context_str = "\n".join([f"【{k.replace('_context', '').title()}】\n{v}" for k, v in remote_context.items()])
    scene_text = await ai_core.ainvoke_with_rotation(
        ai_core.get_remote_nsfw_scene_generator_chain(),
        {
            "user_input": user_input,
            "world_settings": ai_core.profile.world_settings or "未設定",
            "remote_scene_context": remote_scene_context_str,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格：平衡的敘事與對話。",
            "username": ai_core.profile.user_profile.name,
            "ai_name": ai_core.profile.ai_profile.name,
        },
        retry_strategy='force'
    )
    if not scene_text:
        scene_text = "（由於內容限制，AI無法生成您所要求的遠程場景的詳細描述。）"
    return {"llm_response": scene_text}

async def tool_execution_node(state: ConversationGraphState) -> Dict[str, str]:
    user_id = state['user_id']
    ai_core = state['ai_core']
    plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph) Node: tool_execution_node [SFW Path] -> 正在執行行動計劃中的工具...")
    if not plan or not plan.character_actions:
        return {"tool_results": "系統事件：無任何工具被調用。"}
    try:
        results_summary = await ai_core._execute_planned_actions(plan)
    except Exception as e:
        logger.error(f"[{user_id}] (Graph) Node: tool_execution_node -> 在執行工具時發生未捕獲的異常: {e}", exc_info=True)
        results_summary = f"系統事件：工具執行時發生嚴重錯誤: {e}"
    finally:
        tool_context.set_context(None, None)
        logger.info(f"[{user_id}] (Graph) Node: tool_execution_node -> 工具上下文已清理。")
    return {"tool_results": results_summary}

async def narrative_node(state: ConversationGraphState) -> Dict[str, str]:
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    tool_results = state['tool_results']
    logger.info(f"[{user_id}] (Graph) Node: narrative_node [SFW Path] -> 正在处理行动计划...")
    if not turn_plan:
        logger.error(f"[{user_id}] 叙事节点接收到空的行动计划，无法生成回应。")
        return {"llm_response": "（系统错误：未能生成有效的行动计划。）"}
    if turn_plan.execution_rejection_reason:
        logger.info(f"[{user_id}] (Graph) Node: narrative_node -> 检测到拒绝执行的理由，将直接输出。理由: {turn_plan.execution_rejection_reason}")
        return {"llm_response": turn_plan.execution_rejection_reason}
    logger.info(f"[{user_id}] (Graph) Node: narrative_node -> 正在将行动计划和工具结果渲染为小说文本...")
    turn_plan.thought += f"\n\n[系统后台执行结果]:\n{tool_results}"
    narrative_text = await ai_core.ainvoke_with_rotation(
        ai_core.get_narrative_chain(),
        {"turn_plan": turn_plan},
        retry_strategy='force'
    )
    if not narrative_text:
        narrative_text = "（AI 在將計劃轉化為故事時遭遇了內容安全限制。）"
    return {"llm_response": narrative_text}

async def validate_and_rewrite_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    initial_response = state['llm_response']
    logger.info(f"[{user_id}] (Graph) Node: validate_and_rewrite_node -> 正在對 LLM 原始輸出進行內容保全式淨化...")
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
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    clean_response = state['final_output']
    logger.info(f"[{user_id}] (Graph) Node: persist_state_node -> 正在持久化狀態與記憶...")
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

async def background_world_expansion_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    clean_response = state['final_output']
    expansion_decision = state.get('expansion_decision')
    if expansion_decision and expansion_decision.should_expand:
        logger.info(f"[{user_id}] (Graph) Node: background_world_expansion_node -> 正在觸發背景任務...")
        scene_analysis = state.get('scene_analysis')
        effective_location_path = ai_core.profile.game_state.location_path
        if scene_analysis and scene_analysis.target_location_path:
            effective_location_path = scene_analysis.target_location_path
        if clean_response and clean_response != "（...）":
            asyncio.create_task(ai_core._background_scene_expansion(state['messages'][-1].content, clean_response, effective_location_path))
    return {}

async def finalization_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    logger.info(f"[{user_id}] (Graph) Node: finalization_node -> 對話流程圖執行完畢。")
    return {}

# --- 主對話圖的路由 ---

async def route_after_classification(state: ConversationGraphState) -> Literal["sfw_path", "nsfw_interactive_path", "nsfw_descriptive_path"]:
    """根據初始意圖分類結果，將流程路由到隔離的處理路徑。"""
    intent = state['intent_classification'].intent_type
    user_id = state['user_id']
    logger.info(f"[{user_id}] (Graph) Router: 主路由器根據意圖 '{intent}' 進行分發。")
    if intent == 'nsfw_interactive':
        return "nsfw_interactive_path"
    elif intent == 'nsfw_descriptive':
        return "nsfw_descriptive_path"
    else: # 'sfw'
        return "sfw_path"

def route_expansion(state: ConversationGraphState) -> Literal["remote_scene", "expand_lore", "skip_expansion"]:
    user_id = state['user_id']
    scene_analysis = state.get("scene_analysis")
    if scene_analysis and scene_analysis.viewing_mode == 'remote':
        logger.info(f"[{user_id}] (Graph) Router: SFW 內部路由判定為【遠程觀察】。")
        return "remote_scene"
    should_expand = state.get("expansion_decision")
    if should_expand and should_expand.should_expand:
        logger.info(f"[{user_id}] (Graph) Router: SFW 內部路由判定為【本地LORE擴展】。")
        return "expand_lore"
    else:
        logger.info(f"[{user_id}] (Graph) Router: SFW 內部路由判定為【跳過LORE擴展】。")
        return "skip_expansion"

# --- 主對話圖的建構器 ---

# 函式：創建主回應圖 (v16.0 - 風格分析節點架構)
def create_main_response_graph() -> StateGraph:
    graph = StateGraph(ConversationGraphState)
    
    # --- 1. 註冊所有節點 ---
    graph.add_node("classify_intent", classify_intent_node)
    
    # SFW 路徑節點
    graph.add_node("initialize_state", initialize_conversation_state_node)
    graph.add_node("analyze_input", analyze_input_node)
    graph.add_node("expansion_decision", expansion_decision_node)
    graph.add_node("scene_and_action_analysis", scene_and_action_analysis_node)
    graph.add_node("style_analysis", style_analysis_node) # [v16.0 新增]
    graph.add_node("remote_scene_generation", remote_scene_generation_node)
    graph.add_node("planning", planning_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("narrative", narrative_node)
    
    # NSFW 路徑節點
    graph.add_node("generate_nsfw_response", generate_nsfw_response_node)
    graph.add_node("remote_nsfw_scene_generation", remote_nsfw_scene_generator_node)
    
    # 共同路徑節點
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)
    graph.add_node("background_expansion", background_world_expansion_node)
    graph.add_node("finalization", finalization_node)

    # --- 2. 定義圖的拓撲結構 ---
    
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "sfw_path": "initialize_state",
            "nsfw_interactive_path": "generate_nsfw_response",
            "nsfw_descriptive_path": "remote_nsfw_scene_generation"
        }
    )
    
    # [v16.0 核心修正] 重構 SFW 路徑以包含 style_analysis_node
    graph.add_edge("initialize_state", "analyze_input")
    graph.add_edge("analyze_input", "expansion_decision")
    graph.add_edge("expansion_decision", "scene_and_action_analysis")
    graph.add_conditional_edges(
        "scene_and_action_analysis",
        route_expansion,
        {
            "remote_scene": "remote_scene_generation",
            # 如果需要擴展或跳過，都進入風格分析，然後再規劃
            "expand_lore": "style_analysis",
            "skip_expansion": "style_analysis"
        }
    )
    graph.add_edge("style_analysis", "planning") # 風格分析後進入規劃
    graph.add_edge("planning", "tool_execution")
    graph.add_edge("tool_execution", "narrative")
    
    # 將所有生成路徑的終點匯合到驗證節點
    graph.add_edge("generate_nsfw_response", "validate_and_rewrite")
    graph.add_edge("remote_nsfw_scene_generation", "validate_and_rewrite")
    graph.add_edge("remote_scene_generation", "validate_and_rewrite")
    graph.add_edge("narrative", "validate_and_rewrite")
    
    # 連接共同的收尾流程
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", "background_expansion")
    graph.add_edge("background_expansion", "finalization")
    graph.add_edge("finalization", END)
    
    return graph.compile()
# 函式：創建主回應圖 (v16.0 - 風格分析節點架構)

# --- 設定圖 (Setup Graph) 的節點 ---
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
    completed_user_profile = await ai_core.ainvoke_with_rotation(completion_chain, {"profile_json": ai_core.profile.user_profile.model_dump_json()}, retry_strategy='euphemize')
    completed_ai_profile = await ai_core.ainvoke_with_rotation(completion_chain, {"profile_json": ai_core.profile.ai_profile.model_dump_json()}, retry_strategy='euphemize')
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
