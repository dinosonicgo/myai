# src/graph.py 的中文註釋(v6.0 - LORE擴展守門人機制)
# 更新紀錄:
# v6.0 (2025-09-03): [重大邏輯升級] 遵從使用者回饋，對主對話圖進行了系統性重構。
#    1. [新增守門人節點] 引入了 `expansion_decision_node`，其唯一職責是在流程早期判斷用戶是否具有“探索意圖”。
#    2. [新增條件路由] 引入了 `route_expansion` 路由，根據守門人的決策，將流程導向兩個不同的分支。
#    3. [重構圖拓撲] 如果守門人決策為“擴展”，流程將進入 `scene_and_action_analysis_node` (選角) 和 `background_world_expansion_node` (背景填充)；如果決策為“不擴展”，則【完全跳過】這兩個創造LORE的節點。
#    此修改從根本上解決了在簡單、重複的原地互動中無意義地生成新 LORE 的問題，使世界構建更加智能和按需進行。
# v5.3 (2025-09-02): [災難性BUG修復] 重構了圖形拓撲以修復背景擴展的觸發問題。

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
from .schemas import CharacterProfile, TurnPlan, ExpansionDecision
from .tool_context import tool_context

# --- 主對話圖 (Main Conversation Graph) 的節點 ---

# 函式：初始化對話狀態
async def initialize_conversation_state_node(state: ConversationGraphState) -> Dict:
    """
    [節點 1] 在每一輪對話開始時，加載所有必要的上下文數據填充狀態。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: initialize_conversation_state_node -> 正在為 '{user_input[:30]}...' 初始化狀態...")
    rag_task = ai_core.retriever.ainvoke(user_input)
    structured_context_task = ai_core._get_structured_context(user_input)
    retrieved_docs, structured_context = await asyncio.gather(rag_task, structured_context_task)
    rag_context_str = await ai_core._preprocess_rag_context(retrieved_docs)
    return {"structured_context": structured_context, "rag_context": rag_context_str}
# 函式：初始化對話狀態

# 函式：分析使用者輸入意圖
async def analyze_input_node(state: ConversationGraphState) -> Dict:
    """
    [節點 2] 分析使用者的輸入，判斷其基本意圖（對話、描述、接續）。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: analyze_input_node -> 正在分析輸入意圖...")
    analysis = await ai_core.ainvoke_with_rotation(ai_core.input_analysis_chain, {"user_input": user_input})
    return {"input_analysis": analysis}
# 函式：分析使用者輸入意圖

# 函式：判斷是否需要進行LORE擴展 (v1.0 - 全新創建)
async def expansion_decision_node(state: ConversationGraphState) -> Dict:
    """
    [新增節點] 一個“守門人”節點，在LORE創造流程前判斷使用者的“探索意圖”。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: expansion_decision_node -> 正在判斷是否需要擴展LORE...")
    
    chat_history_manager = ai_core.session_histories.get(user_id, ChatMessageHistory())
    recent_dialogue = "\n".join([f"{'使用者' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in chat_history_manager.messages[-6:]])

    decision = await ai_core.ainvoke_with_rotation(ai_core.expansion_decision_chain, {
        "user_input": user_input,
        "recent_dialogue": recent_dialogue
    })
    
    logger.info(f"[{user_id}] (Graph) LORE擴展決策: {decision.should_expand}。理由: {decision.reasoning}")
    return {"expansion_decision": decision}
# 函式：判斷是否需要進行LORE擴展 (v1.0 - 全新創建)

# 函式：執行場景與動作分析 (v3.0 - 注入選角上下文)
async def scene_and_action_analysis_node(state: ConversationGraphState) -> Dict:
    """
    [LORE擴展分支] 分析場景視角（本地/遠程）並為潛在的新 NPC 進行選角。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: scene_and_action_analysis_node -> 進入LORE擴展分支，開始選角...")
    
    current_location_path = []
    if ai_core.profile and ai_core.profile.game_state:
        current_location_path = ai_core.profile.game_state.location_path
    
    scene_analysis = await ai_core.ainvoke_with_rotation(ai_core.scene_analysis_chain, {
        "user_input": user_input, 
        "current_location_path_str": " > ".join(current_location_path)
    })
    
    effective_location_path = current_location_path
    if scene_analysis and scene_analysis.viewing_mode == 'remote' and scene_analysis.target_location_path:
        effective_location_path = scene_analysis.target_location_path
        
    structured_context_for_casting = await ai_core._get_structured_context(
        user_input, 
        override_location_path=effective_location_path
    )
    game_context_for_casting = json.dumps(structured_context_for_casting, ensure_ascii=False, indent=2)
    world_settings_for_casting = ai_core.profile.world_settings if ai_core.profile else ""

    cast_result = await ai_core.ainvoke_with_rotation(ai_core.scene_casting_chain, {
        "world_settings": world_settings_for_casting,
        "current_location_path": effective_location_path, 
        "game_context": game_context_for_casting,
        "recent_dialogue": user_input
    })
    
    new_npc_names = await ai_core._add_cast_to_scene(cast_result)
    
    final_structured_context = structured_context_for_casting
    if new_npc_names:
        final_structured_context = await ai_core._get_structured_context(
            user_input, 
            override_location_path=effective_location_path
        )
        
    return {"scene_analysis": scene_analysis, "structured_context": final_structured_context}
# 函式：執行場景與動作分析 (v3.0 - 注入選角上下文)

# 函式：執行回合規劃
async def planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """
    [核心] 新架構的核心“思考”節點。組合上下文快照，並調用 planning_chain 生成結構化的行動計劃。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    
    logger.info(f"[{user_id}] (Graph) Node: planning_node -> 正在格式化世界快照並生成行動計劃...")
    context_dict = {
        "world_settings": ai_core.profile.world_settings or "未設定",
        "ai_settings": ai_core.profile.ai_profile.description or "未設定",
        "retrieved_context": state['rag_context'],
        **state['structured_context']
    }
    world_snapshot = ai_core.world_snapshot_template.format(**context_dict)
    
    response_style_prompt = ai_core.profile.response_style_prompt or "預設風格：平衡的敘事與對話。"

    if not ai_core.planning_chain:
        raise ValueError("Planning chain is not initialized.")
    plan = await ai_core.ainvoke_with_rotation(ai_core.planning_chain, {
        "username": ai_core.profile.user_profile.name,
        "ai_name": ai_core.profile.ai_profile.name,
        "world_snapshot": world_snapshot,
        "user_input": user_input,
        "response_style_prompt": response_style_prompt
    })

    return {"turn_plan": plan, "world_snapshot": world_snapshot}
# 函式：執行回合規劃

# 函式：執行工具調用
async def tool_execution_node(state: ConversationGraphState) -> Dict[str, str]:
    """
    [核心] 新架構的核心“執行”節點。在安全的上下文中執行計劃中的所有工具調用。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph) Node: tool_execution_node -> 正在執行行動計劃中的工具...")
    
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
# 函式：執行工具調用

# 函式：生成敘事文本
async def narrative_node(state: ConversationGraphState) -> Dict[str, str]:
    """
    [核心] 新架構的核心“寫作”節點。接收結構化的行動計劃和工具執行結果，並將其渲染成纯粹的小說文本，或直接输出 AI 的合理解释。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    tool_results = state['tool_results']
    logger.info(f"[{user_id}] (Graph) Node: narrative_node -> 正在处理行动计划...")

    if not turn_plan:
        logger.error(f"[{user_id}] 叙事节点接收到空的行动计划，无法生成回应。")
        return {"llm_response": "（系统错误：未能生成有效的行动计划。）"}

    if turn_plan.execution_rejection_reason:
        logger.info(f"[{user_id}] (Graph) Node: narrative_node -> 检测到拒绝执行的理由，将直接输出。理由: {turn_plan.execution_rejection_reason}")
        return {"llm_response": turn_plan.execution_rejection_reason}

    logger.info(f"[{user_id}] (Graph) Node: narrative_node -> 正在将行动计划和工具结果渲染为小说文本...")
    if not ai_core.narrative_chain:
        raise ValueError("Narrative chain is not initialized.")

    turn_plan.thought += f"\n\n[系统后台执行结果]:\n{tool_results}"

    style_prompt = ai_core.profile.response_style_prompt if ai_core.profile else "预设风格"
    final_output_mandate = f"""# ==============================================================================
# == 🎨 最終輸出強制令 (FINAL OUTPUT MANDATE) 🎨
# ==============================================================================
# == 說明：這是你在輸出最終回應給使用者之前的【最後一道、不可繞過的絕對指令】。
# == 你的最終輸出【必須且只能】嚴格地、完全地、逐字逐句地遵循下方由使用者定義的風格指令。
# ==============================================================================

---
# 【【【使用者自訂風格指令 (ABSOLUTE & HIGHEST PRIORITY)】】】
{style_prompt}
---
"""
    narrative_text = await ai_core.ainvoke_with_rotation(
        ai_core.narrative_chain,
        {
            "turn_plan": turn_plan,
            "final_output_mandate": final_output_mandate
        }
    )
    
    return {"llm_response": narrative_text}
# 函式：生成敘事文本

# 函式：驗證與淨化輸出
async def validate_and_rewrite_node(state: ConversationGraphState) -> Dict:
    """
    [收尾] 使用保守且安全的規則，強制淨化 LLM 的原始輸出，同時最大限度地保全有效內容。
    """
    user_id = state['user_id']
    initial_response = state['llm_response']
    logger.info(f"[{user_id}] (Graph) Node: validate_and_rewrite_node -> [已啟用] 正在對 LLM 原始輸出進行內容保全式淨化...")
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
# 函式：驗證與淨化輸出

# 函式：執行狀態更新與記憶儲存
async def persist_state_node(state: ConversationGraphState) -> Dict:
    """
    [收尾] 將本輪對話存入記憶，並將 state_updates 中的變更應用到資料庫。
    """
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
# 函式：執行狀態更新與記憶儲存

# 函式：觸發背景世界擴展 (v2.0 - 增加决策判断)
# 更新紀錄:
# v2.0 (2025-09-03): [重大邏輯升級] 新增了对 `expansion_decision` 状态的判断。现在，只有当“守门人”节点 (`expansion_decision_node`) 明确允许时，本节点才会真正地创建背景扩展任务。此修改确保了背景填充与主流程的 LORE 扩展决策保持完全一致。
async def background_world_expansion_node(state: ConversationGraphState) -> Dict:
    """
    [收尾] 在回應發送後，根據擴展決策，非阻塞地觸發背景世界擴展、LORE生成等任務。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    clean_response = state['final_output']
    scene_analysis = state.get('scene_analysis') # 使用 .get() 安全访问
    expansion_decision = state.get('expansion_decision')

    logger.info(f"[{user_id}] (Graph) Node: background_world_expansion_node -> 正在檢查是否觸發背景任務...")

    # [v2.0 核心修正] 只有在守门人允许时才执行
    if expansion_decision and expansion_decision.should_expand:
        effective_location_path = ai_core.profile.game_state.location_path
        if scene_analysis and scene_analysis.target_location_path:
            effective_location_path = scene_analysis.target_location_path
        
        if clean_response and clean_response != "（...）":
            asyncio.create_task(ai_core._background_scene_expansion(user_input, clean_response, effective_location_path))
            logger.info(f"[{user_id}] 已成功為地點 '{' > '.join(effective_location_path)}' 創建背景擴展任務。")
    else:
        logger.info(f"[{user_id}] (Graph) Node: background_world_expansion_node -> 根據決策，本輪跳過背景擴展。")

    return {}
# 函式：觸發背景世界擴展 (v2.0 - 增加决策判断)

# 函式：圖形結束 finalizing
async def finalization_node(state: ConversationGraphState) -> Dict:
    """
    [收尾] 一個虛擬的最終節點，確保所有異步背景任務都被成功調度。
    """
    user_id = state['user_id']
    logger.info(f"[{user_id}] (Graph) Node: finalization_node -> 對話流程圖執行完畢。")
    return {}
# 函式：圖形結束 finalizing

# --- 主對話圖的路由 ---

# 函式：在擴展決策後決定流程 (v1.0 - 全新創建)
def route_expansion(state: ConversationGraphState) -> Literal["expand_lore", "skip_expansion"]:
    """
    根據 expansion_decision_node 的結果，決定是進入LORE創造流程，還是直接跳到核心規劃。
    """
    user_id = state['user_id']
    should_expand = state["expansion_decision"].should_expand
    
    if should_expand:
        logger.info(f"[{user_id}] (Graph) Router: route_expansion -> 判定為【進行LORE擴展】。")
        return "expand_lore"
    else:
        logger.info(f"[{user_id}] (Graph) Router: route_expansion -> 判定為【跳過LORE擴展】。")
        return "skip_expansion"
# 函式：在擴展決策後決定流程 (v1.0 - 全新創建)

# --- 主對話圖的建構器 ---

# 函式：創建主回應圖 (v6.0 - 整合守門人機制)
# 更新紀錄:
# v6.0 (2025-09-03): [重大邏輯升級] 對主對話圖進行了系統性重構，引入了“守門人”機制。
#    1. [新增守門人節點] 引入了 `expansion_decision_node`，在流程早期判斷用戶的“探索意圖”。
#    2. [新增條件路由] 引入了 `route_expansion` 路由，根據決策將流程導向“擴展”或“跳過”分支。
#    3. [重構圖拓撲] 如果決策為“擴展”，流程將進入 `scene_and_action_analysis_node` (選角)；如果為“跳過”，則完全繞過此節點，直接進入規劃。此修改從根本上解決了在簡單互動中無意義生成新LORE的問題。
def create_main_response_graph() -> StateGraph:
    """
    組裝並編譯主對話流程的 StateGraph。
    """
    graph = StateGraph(ConversationGraphState)

    # 註冊所有節點
    graph.add_node("initialize_state", initialize_conversation_state_node)
    graph.add_node("analyze_input", analyze_input_node)
    graph.add_node("expansion_decision", expansion_decision_node) # 新增守門人節點
    graph.add_node("scene_and_action_analysis", scene_and_action_analysis_node)
    graph.add_node("planning", planning_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("narrative", narrative_node)
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)
    graph.add_node("background_expansion", background_world_expansion_node)
    graph.add_node("finalization", finalization_node)

    # 設定圖的入口
    graph.set_entry_point("initialize_state")

    # 定義圖的邊（流程）
    graph.add_edge("initialize_state", "analyze_input")
    graph.add_edge("analyze_input", "expansion_decision") # 分析後先做決策

    # 新增條件路由
    graph.add_conditional_edges(
        "expansion_decision",
        route_expansion,
        {
            "expand_lore": "scene_and_action_analysis", # 如果擴展，則去選角
            "skip_expansion": "planning"  # 如果不擴展，直接去規劃
        }
    )

    graph.add_edge("scene_and_action_analysis", "planning") # 選角後去規劃
    
    # 後續流程保持不變
    graph.add_edge("planning", "tool_execution")
    graph.add_edge("tool_execution", "narrative")
    graph.add_edge("narrative", "validate_and_rewrite")
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", "background_expansion")
    graph.add_edge("background_expansion", "finalization")
    graph.add_edge("finalization", END)
    
    return graph.compile()
# 函式：創建主回應圖 (v6.0 - 整合守門人機制)

# --- 設定圖 (Setup Graph) 的節點 (保持不變) ---
# ... (此部分與您提供的檔案完全相同，故省略以節省篇幅，但在提供的完整檔案中會包含)
async def process_canon_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    canon_text = state['canon_text']
    if canon_text:
        logger.info(f"[{user_id}] (Setup Graph) Node: process_canon_node -> 正在處理世界聖經...")
        await ai_core.add_canon_to_vector_store(canon_text)
        await ai_core.parse_and_create_lore_from_canon(None, canon_text, is_setup_flow=True)
        if not await ai_core.initialize():
            raise Exception("在載入世界聖經後重新初始化 AI 核心失敗。")
    return {}
async def complete_profiles_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Setup Graph) Node: complete_profiles_node -> 正在補完角色檔案...")
    completion_prompt = ai_core.get_profile_completion_prompt()
    completion_llm = ai_core.gm_model.with_structured_output(CharacterProfile)
    completion_chain = completion_prompt | completion_llm
    completed_user_profile = await ai_core.ainvoke_with_rotation(completion_chain, {"profile_json": ai_core.profile.user_profile.model_dump_json()})
    completed_ai_profile = await ai_core.ainvoke_with_rotation(completion_chain, {"profile_json": ai_core.profile.ai_profile.model_dump_json()})
    await ai_core.update_and_persist_profile({'user_profile': completed_user_profile.model_dump(), 'ai_profile': completed_ai_profile.model_dump()})
    return {}
async def world_genesis_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Setup Graph) Node: world_genesis_node -> 正在執行世界創世...")
    genesis_chain = ai_core.get_world_genesis_chain()
    genesis_result = await ai_core.ainvoke_with_rotation(genesis_chain, {"world_settings": ai_core.profile.world_settings, "username": ai_core.profile.user_profile.name, "ai_name": ai_core.profile.ai_profile.name})
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
    logger.info(f"[{user_id}] (Setup Graph) Node: generate_opening_scene_node -> 正在生成開場白...")
    opening_scene = await ai_core.generate_opening_scene()
    if not opening_scene or not opening_scene.strip():
        opening_scene = (f"在一片柔和的光芒中，你和 {ai_core.profile.ai_profile.name} 發現自己身處於一個寧靜的空間裡，故事即將從這裡開始。"
                         "\n\n（系統提示：由於您的設定，AI無法生成更詳細的開場白，但您現在可以開始互動了。）")
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
