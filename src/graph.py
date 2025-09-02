# src/graph.py 的中文註釋(v5.2 - 安全上下文管理)
# 更新紀錄:
# v5.2 (2025-09-02): [重大架構重構 - 健壯性] 重構了 `tool_execution_node`，引入了 `try...finally` 結構。此修改確保了無論工具執行成功與否，共享的 `tool_context` 都會被【絕對可靠地】清理。這從根本上解決了因異常導致上下文狀態洩漏到後續對話中的潛在風險，極大地提高了系統的穩定性。
# v5.1 (2025-09-02): [重大架構重構 - 流程簡化] 移除了 `assemble_world_snapshot_node` 並將其職責合併進 `planning_node`。
# v5.0 (2025-09-02): [重大架構重構 - 執行分離] 引入了 `tool_execution_node`，完成了“思考->執行->寫作”的閉環。

import sys
print(f"[DEBUG] graph.py loaded from: {__file__}", file=sys.stderr)
import asyncio
import json
import re
from typing import Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from .ai_core import AILover
from .logger import logger
from .graph_state import ConversationGraphState, SetupGraphState
from . import lore_book, tools
from .schemas import CharacterProfile, TurnPlan
# [v5.2 新增] 導入共享的工具上下文
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

# 函式：執行場景與動作分析 (v3.0 - 注入選角上下文)
# 更新紀錄:
# v3.0 (2025-09-03): [重大邏輯升級] 遵从用户反馈和日志分析，重构了此节点的执行流程。现在，在调用 `scene_casting_chain` 之前，会先调用 `_get_structured_context` 来获取包含【当前所有已知NPC】的完整场景上下文，并将其注入到选角链中。这为选角链提供了避免重复创造角色的关键判断依据，旨在从根本上解决无限生成相似 NPC 的问题。
# v1.1 (2025-09-02): [災難性BUG修復] 移除了在調用 `_get_structured_context` 時傳遞的過時參數。
async def scene_and_action_analysis_node(state: ConversationGraphState) -> Dict:
    """
    [節點 3A - 敘事路徑] 分析場景視角（本地/遠程）並為潛在的新 NPC 進行選角。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: scene_and_action_analysis_node -> 進入敘事路徑分析...")
    
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
        
    # [v3.0 核心修正] 先獲取一次完整的上下文，提供給選角鏈
    structured_context_for_casting = await ai_core._get_structured_context(
        user_input, 
        override_location_path=effective_location_path
    )
    game_context_for_casting = json.dumps(structured_context_for_casting, ensure_ascii=False, indent=2)
    world_settings_for_casting = ai_core.profile.world_settings if ai_core.profile else ""

    # [v3.0 核心修正] 將獲取到的上下文注入選角鏈
    cast_result = await ai_core.ainvoke_with_rotation(ai_core.scene_casting_chain, {
        "world_settings": world_settings_for_casting,
        "current_location_path": effective_location_path, 
        "game_context": game_context_for_casting, # <--- 注入上下文
        "recent_dialogue": user_input
    })
    
    new_npc_names = await ai_core._add_cast_to_scene(cast_result)
    
    final_structured_context = structured_context_for_casting
    if new_npc_names:
        # 如果創建了新NPC，再次獲取上下文以包含他們，確保後續流程能感知到新角色
        final_structured_context = await ai_core._get_structured_context(
            user_input, 
            override_location_path=effective_location_path
        )
        
    return {"scene_analysis": scene_analysis, "structured_context": final_structured_context}
# 函式：執行場景與動作分析 (v3.0 - 注入選角上下文)

# 函式：執行回合規劃 (v1.2 - 傳遞風格指令)
# 更新紀錄:
# v1.2 (2025-09-02): [災難性BUG修復] 重構此節點，使其現在負責從 ai_core.profile 中明確讀取 `response_style_prompt`，並將其作為一個關鍵參數傳遞給 `planning_chain`。這確保了“思考”節點能夠感知到使用者的風格要求，從而制定出包含正確元素（如對話）的行動計劃，徹底解決了 AI 忽略風格指令的問題。
# v1.1 (2025-09-02): [架構重構] 將上下文組合的職責合併進此節點。
async def planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """
    [節點 4] 新架構的核心“思考”節點。組合上下文快照，並調用 planning_chain 生成結構化的行動計劃。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    
    # 步驟 1: 組合世界快照
    logger.info(f"[{user_id}] (Graph) Node: planning_node -> 正在格式化世界快照並生成行動計劃...")
    context_dict = {
        "world_settings": ai_core.profile.world_settings or "未設定",
        "ai_settings": ai_core.profile.ai_profile.description or "未設定",
        "retrieved_context": state['rag_context'],
        **state['structured_context']
    }
    world_snapshot = ai_core.world_snapshot_template.format(**context_dict)
    
    # 步驟 2: [v1.2 新增] 獲取風格指令
    response_style_prompt = ai_core.profile.response_style_prompt or "預設風格：平衡的敘事與對話。"

    # 步驟 3: 執行規劃，並傳入風格指令
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
# 函式：執行回合規劃 (v1.2 - 傳遞風格指令)

# 函式：執行工具調用 (v1.1 - 安全上下文管理)
async def tool_execution_node(state: ConversationGraphState) -> Dict[str, str]:
    """
    [節點 5] 新架構的核心“執行”節點。在安全的上下文中執行計劃中的所有工具調用。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph) Node: tool_execution_node -> 正在執行行動計劃中的工具...")
    
    if not plan:
        return {"tool_results": "系統事件：因行動計劃為空，未執行任何工具。"}
    
    # [v5.2 核心修正] 使用 try...finally 確保上下文被絕對可靠地清理
    try:
        results_summary = await ai_core._execute_planned_actions(plan)
    except Exception as e:
        logger.error(f"[{user_id}] (Graph) Node: tool_execution_node -> 在執行工具時發生未捕獲的異常: {e}", exc_info=True)
        results_summary = f"系統事件：工具執行時發生嚴重錯誤: {e}"
    finally:
        # 無論成功或失敗，都必須清理上下文，防止狀態洩漏
        tool_context.set_context(None, None)
        logger.info(f"[{user_id}] (Graph) Node: tool_execution_node -> 工具上下文已清理。")

    return {"tool_results": results_summary}
# 函式：執行工具調用 (v1.1 - 安全上下文管理)

# 函式：生成敘事文本 (v2.0 - 增加对拒绝执行的处理逻辑)
# 更新紀錄:
# v2.0 (2025-09-03): [重大智能升級] 新增了对行动计划中 `execution_rejection_reason` 字段的处理逻辑。如果该字段存在，本节点将不再渲染常规的小说场景，而是直接将 AI 给出的“拒绝执行的合理解释”作为最终输出。这使得 AI 能够智能地回应不合逻辑的用户指令，是实现“智慧型 GM”的关键一步。
async def narrative_node(state: ConversationGraphState) -> Dict[str, str]:
    """
    [節點 6] 新架構的核心“寫作”節點。接收結構化的行動計劃和工具執行結果，並將其渲染成纯粹的小說文本，或直接输出 AI 的合理解释。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    tool_results = state['tool_results']
    logger.info(f"[{user_id}] (Graph) Node: narrative_node -> 正在处理行动计划...")

    if not turn_plan:
        logger.error(f"[{user_id}] 叙事节点接收到空的行动计划，无法生成回应。")
        return {"llm_response": "（系统错误：未能生成有效的行动计划。）"}

    # [v2.0 核心修正] 检查是否存在拒绝执行的理由
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
# 函式：生成敘事文本 (v2.0 - 增加对拒绝执行的处理逻辑)

# 函式：驗證與淨化輸出
async def validate_and_rewrite_node(state: ConversationGraphState) -> Dict:
    """
    [節點 7] 使用保守且安全的規則，強制淨化 LLM 的原始輸出，同時最大限度地保全有效內容。
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
    [節點 8] 將本輪對話存入記憶，並將 state_updates 中的變更應用到資料庫。
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

# 函式：觸發背景世界擴展
async def background_world_expansion_node(state: ConversationGraphState) -> Dict:
    """
    [節點 9] 在回應發送後，非阻塞地觸發背景世界擴展、LORE生成等任務。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    clean_response = state['final_output']
    scene_analysis = state['scene_analysis']
    logger.info(f"[{user_id}] (Graph) Node: background_world_expansion_node -> 正在觸發背景任務...")
    effective_location_path = ai_core.profile.game_state.location_path
    if scene_analysis and scene_analysis.target_location_path:
        effective_location_path = scene_analysis.target_location_path
    if scene_analysis and (scene_analysis.viewing_mode == 'local' or scene_analysis.target_location_path):
        if clean_response and clean_response != "（...）":
            asyncio.create_task(ai_core._background_scene_expansion(user_input, clean_response, effective_location_path))
            logger.info(f"[{user_id}] 已成功為地點 '{' > '.join(effective_location_path)}' 創建背景擴展任務。")
    return {}
# 函式：觸發背景世界擴展

# 函式：圖形結束 finalizing
async def finalization_node(state: ConversationGraphState) -> Dict:
    """
    [節點 10] 一個虛擬的最終節點，確保所有異步背景任務都被成功調度。
    """
    user_id = state['user_id']
    logger.info(f"[{user_id}] (Graph) Node: finalization_node -> 對話流程圖執行完畢。")
    return {}
# 函式：圖形結束 finalizing

# --- 主對話圖的路由 ---

# 函式：在輸入分析後決定流程
def route_after_input_analysis(state: ConversationGraphState) -> Literal["narrative_flow", "dialogue_flow"]:
    input_type = state["input_analysis"].input_type
    user_id = state['user_id']
    if input_type in ['narration', 'continuation']:
        logger.info(f"[{user_id}] (Graph) Router: route_after_input_analysis -> 判定為「敘事流程」。")
        return "narrative_flow"
    else:
        logger.info(f"[{user_id}] (Graph) Router: route_after_input_analysis -> 判定為「對話流程」。")
        return "dialogue_flow"
# 函式：在輸入分析後決定流程

# --- 主對話圖的建構器 ---

# 函式：創建主回應圖 (v5.3 - 修復背景擴展觸發)
# 更新紀錄:
# v5.3 (2025-09-02): [災難性BUG修復] 徹底重構了圖形的拓撲結構。舊結構在“對話流程”中會跳過 `scene_and_action_analysis_node`，導致 `scene_analysis` 狀態為空，從而使 `background_world_expansion_node` 永遠無法觸發背景任務。新結構將 `scene_and_action_analysis_node` 提升為所有流程的必經節點，確保了背景世界擴展功能在每一次對話後都能被可靠地觸發。
# v5.2 (2025-09-02): [重大架構重構 - 健壯性] 在 `tool_execution_node` 中引入了 `try...finally` 結構，確保工具上下文的絕對清理。
def create_main_response_graph() -> StateGraph:
    """
    組裝並編譯主對話流程的 StateGraph。
    """
    graph = StateGraph(ConversationGraphState)

    graph.add_node("initialize_state", initialize_conversation_state_node)
    graph.add_node("analyze_input", analyze_input_node)
    graph.add_node("scene_and_action_analysis", scene_and_action_analysis_node)
    graph.add_node("planning", planning_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("narrative", narrative_node)
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)
    graph.add_node("background_expansion", background_world_expansion_node)
    graph.add_node("finalization", finalization_node)

    graph.set_entry_point("initialize_state")

    graph.add_edge("initialize_state", "analyze_input")
    
    # [v5.3 修正] 不再使用條件路由，analyze_input 統一指向 scene_and_action_analysis
    graph.add_edge("analyze_input", "scene_and_action_analysis")
    
    # [v5.3 修正] scene_and_action_analysis 成為 planning 之前的必經節點
    graph.add_edge("scene_and_action_analysis", "planning")
    
    graph.add_edge("planning", "tool_execution")
    graph.add_edge("tool_execution", "narrative")
    graph.add_edge("narrative", "validate_and_rewrite")
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", "background_expansion")
    graph.add_edge("background_expansion", "finalization")
    graph.add_edge("finalization", END)
    
    return graph.compile()
# 函式：創建主回應圖 (v5.3 - 修復背景擴展觸發)

# --- 設定圖 (Setup Graph) 的節點 (保持不變) ---
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
