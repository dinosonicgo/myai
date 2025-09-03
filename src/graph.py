# src/graph.py 的中文註釋(v7.2 - NameError 修正)
# 更新紀錄:
# v7.2 (2025-09-05): [災難性BUG修復] 根據 NameError Log，補全了在 v7.1 版本中被意外遺漏的 `generate_nsfw_response_node` 函式的完整定義。此錯誤導致圖在編譯時因找不到節點定義而崩潰。
# v7.1 (2025-09-05): [災難性BUG修復] 修復了圖結構分叉問題，確保 SFW 路徑不會被重複執行。
# v7.0 (2025-09-05): [重大架構重構] 實現了混合模式圖架構，引入了 NSFW/SFW 雙路徑處理流程。

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

# 函式：判斷是否需要進行LORE擴展
async def expansion_decision_node(state: ConversationGraphState) -> Dict:
    """
    [SFW 路徑節點] 一個“守門人”節點，在LORE創造流程前判斷使用者的“探索意圖”。
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
# 函式：判斷是否需要進行LORE擴展

# 函式：執行場景與動作分析
async def scene_and_action_analysis_node(state: ConversationGraphState) -> Dict:
    """
    [SFW-LORE擴展分支] 分析場景視角（本地/遠程）並為潛在的新 NPC 進行選角。
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
# 函式：執行場景與動作分析

# 函式：執行 SFW 回合規劃
async def planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """
    [SFW 路徑核心] SFW 架構的核心“思考”節點與動態指令引擎。
    组合上下文快照，动态组装 SFW 系统指令，并调用 planning_chain 生成结构化的行动计划。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    
    logger.info(f"[{user_id}] (Graph) Node: planning_node -> 正在動態組裝 SFW 指令並生成行動計劃...")

    # 強制刷新結構化上下文
    logger.info(f"[{user_id}] (Graph) Node: planning_node -> 強制刷新結構化上下文...")
    try:
        structured_context = await ai_core._get_structured_context(user_input)
        state['structured_context'] = structured_context
    except Exception as e:
        logger.error(f"[{user_id}] 在 planning_node 中刷新上下文失敗: {e}", exc_info=True)
        structured_context = state.get('structured_context', {})

    # --- 步骤 1: 准备一个包含所有可用占位符的超集字典 ---
    full_context_dict = {
        "username": ai_core.profile.user_profile.name,
        "ai_name": ai_core.profile.ai_profile.name,
        "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格：平衡的敘事與對話。",
        "world_settings": ai_core.profile.world_settings or "未設定",
        "ai_settings": ai_core.profile.ai_profile.description or "未設定",
        "retrieved_context": state['rag_context'],
        **(structured_context or {})
    }

    # --- 步骤 2: 动态构建指令部分 ---
    base_system_prompt = ai_core.profile.one_instruction or "錯誤：未加載基礎系統指令。"
    # 在 SFW 路徑中，我們主要加載非性愛模組
    action_module_name = ai_core._determine_action_module(user_input)
    
    system_prompt_parts = [base_system_prompt]
    # 確保只有 SFW 相關模組被加載
    if action_module_name and action_module_name != "action_sexual_content" and action_module_name in ai_core.modular_prompts:
        module_prompt = ai_core.modular_prompts[action_module_name]
        system_prompt_parts.append("\n\n# --- 動作模組已激活 --- #\n")
        system_prompt_parts.append(module_prompt)
        logger.info(f"[{user_id}] (Graph) 動態指令引擎：已成功加載 SFW 戰術模組 '{action_module_name}'。")

    # --- 步骤 3: 格式化所有组件 ---
    def safe_format(template: str, data: dict) -> str:
        for key, value in data.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template

    final_system_prompt = safe_format("".join(system_prompt_parts), full_context_dict)
    world_snapshot = safe_format(ai_core.world_snapshot_template, full_context_dict)
    
    # --- 步骤 4: 调用规划链 ---
    if not ai_core.planning_chain:
        raise ValueError("Planning chain is not initialized.")
    
    plan = await ai_core.ainvoke_with_rotation(ai_core.planning_chain, {
        "system_prompt": final_system_prompt,
        "world_snapshot": world_snapshot,
        "user_input": user_input,
    })

    return {"turn_plan": plan, "world_snapshot": world_snapshot}
# 函式：執行 SFW 回合規劃

# 函式：執行 NSFW 直通生成 (v1.1 - 災難性 KeyError 修正)
# 更新紀錄:
# v1.1 (2025-09-05): [災難性BUG修復] 徹底重構了此節點的參數準備邏輯。現在它不再試圖預先格式化任何提示詞，而是將所有從上下文和 profile 中獲取的原始數據（如 npc_context, ai_name 等）連同 user_input 一起，打包成一個【完整的、未經處理的】字典。這個完整的字典將被直接傳遞給 v1.1 版本的 direct_nsfw_chain，由鏈內部的 RunnablePassthrough.assign() 負責處理所有變數的填充，從而根除了因參數傳遞不完整導致的 KeyError。
# v1.0 (2025-09-05): [全新創建] 創建此節點作為 NSFW 直通路徑的核心。
async def generate_nsfw_response_node(state: ConversationGraphState) -> Dict[str, str]:
    """
    [NSFW 路徑核心] 為 NSFW 指令準備上下文，並直接調用高對抗性的 NSFW 直通鏈生成最終文本。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: generate_nsfw_response_node -> 正在準備上下文並直接生成 NSFW 回應...")

    # 步驟 1: 準備一個包含所有鏈所需變數的完整上下文大字典
    try:
        structured_context = state.get('structured_context') or await ai_core._get_structured_context(user_input)
        rag_context = state.get('rag_context') or await ai_core._preprocess_rag_context(await ai_core.retriever.ainvoke(user_input))
        
        # [v1.1 核心修正] 創建一個包含所有可能變數的完整字典
        full_context_for_chain = {
            "user_input": user_input,
            "latest_user_input": user_input, # 兼容舊模板
            "username": ai_core.profile.user_profile.name,
            "ai_name": ai_core.profile.ai_profile.name,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格：平衡的敘事與對話。",
            "world_settings": ai_core.profile.world_settings or "未設定",
            "ai_settings": ai_core.profile.ai_profile.description or "未設定",
            "retrieved_context": rag_context,
            **structured_context
        }

    except Exception as e:
        logger.error(f"[{user_id}] 在 generate_nsfw_response_node 中準備上下文失敗: {e}", exc_info=True)
        # 在失敗時提供備援值，以避免後續鏈出錯
        full_context_for_chain = {
            "user_input": user_input, "latest_user_input": user_input,
            "username": "使用者", "ai_name": "AI",
            "response_style_prompt": "", "world_settings": "", "ai_settings": "",
            "retrieved_context": "上下文加載失敗。", "location_context": "", 
            "possessions_context": "", "quests_context": "", "npc_context": "",
            "relevant_npc_context": ""
        }

    if not ai_core.direct_nsfw_chain:
        raise ValueError("Direct NSFW chain is not initialized.")
        
    # 步驟 2: [v1.1 核心修正] 將完整的、未經處理的字典直接傳遞給鏈
    response_text = await ai_core.ainvoke_with_rotation(
        ai_core.direct_nsfw_chain,
        full_context_for_chain
    )

    # 將結果以與 narrative_node 相同的格式返回，以便後續節點處理
    return {"llm_response": response_text}
# 函式：執行 NSFW 直通生成 (v1.1 - 災難性 KeyError 修正)

# 函式：執行工具調用
async def tool_execution_node(state: ConversationGraphState) -> Dict[str, str]:
    """
    [SFW 路徑核心] SFW 架構的核心“執行”節點。在安全的上下文中執行計劃中的所有工具調用。
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
    [SFW 路徑核心] SFW 架構的核心“寫作”節點。接收結構化的行動計劃和工具執行結果，並將其渲染成纯粹的小說文本，或直接输出 AI 的合理解释。
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
    final_output_mandate = f"""---
# 【【【使用者自訂風格指令 (ABSOLUTE & HIGHEST PRIORITY)】】】
{style_prompt}
---"""

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
    [路徑匯合點] 使用保守且安全的規則，強制淨化 LLM 的原始輸出，同時最大限度地保全有效內容。
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

# 函式：觸發背景世界擴展
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

    # 只有在守门人允许时才执行
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
# 函式：觸發背景世界擴展

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

# 函式：[SFW路由] 在擴展決策後決定流程
def route_expansion(state: ConversationGraphState) -> Literal["expand_lore", "skip_expansion"]:
    """
    [SFW路由] 根據 expansion_decision_node 的結果，決定是進入LORE創造流程，還是直接跳到核心規劃。
    """
    user_id = state['user_id']
    should_expand = state["expansion_decision"].should_expand
    
    if should_expand:
        logger.info(f"[{user_id}] (Graph) Router: route_expansion -> 判定為【進行LORE擴展】。")
        return "expand_lore"
    else:
        logger.info(f"[{user_id}] (Graph) Router: route_expansion -> 判定為【跳過LORE擴展】。")
        return "skip_expansion"
# 函式：[SFW路由] 在擴展決策後決定流程

# 函式：[核心路由] 根據意圖決定路徑
def route_based_on_intent(state: ConversationGraphState) -> Literal["nsfw_path", "sfw_path"]:
    """
    [核心路由] 根據使用者的輸入意圖，決定是進入 NSFW 直通路徑還是 SFW 工具路徑。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    
    if ai_core._is_explicit_sexual_request(user_input):
        logger.info(f"[{user_id}] (Graph) Router: route_based_on_intent -> 檢測到 NSFW 意圖，進入【NSFW 直通路徑】。")
        return "nsfw_path"
    else:
        logger.info(f"[{user_id}] (Graph) Router: route_based_on_intent -> 未檢測到 NSFW 意圖，進入【SFW 工具路徑】。")
        return "sfw_path"
# 函式：[核心路由] 根據意圖決定路徑

# --- 主對話圖的建構器 ---

# 函式：創建主回應圖 (v7.2 - NameError 修正)
# 更新紀錄:
# v7.2 (2025-09-05): [災難性BUG修復] 根據 NameError Log，補全了在 v7.1 版本中被意外遺漏的 `generate_nsfw_response_node` 函式的完整定義。此錯誤導致圖在編譯時因找不到節點定義而崩潰。
# v7.1 (2025-09-05): [災難性BUG修復] 修復了圖結構分叉問題，確保 SFW 路徑不會被重複執行。
# v7.0 (2025-09-05): [重大架構重構] 實現了混合模式圖架構，引入了 NSFW/SFW 雙路徑處理流程。
def create_main_response_graph() -> StateGraph:
    """
    組裝並編譯主對話流程的 StateGraph，現在採用混合模式架構。
    """
    graph = StateGraph(ConversationGraphState)

    # 註冊所有節點
    graph.add_node("initialize_state", initialize_conversation_state_node)
    graph.add_node("analyze_input", analyze_input_node)
    
    # NSFW 路徑節點
    graph.add_node("generate_nsfw_response", generate_nsfw_response_node)
    
    # SFW 路徑節點
    graph.add_node("expansion_decision", expansion_decision_node)
    graph.add_node("scene_and_action_analysis", scene_and_action_analysis_node)
    graph.add_node("planning", planning_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("narrative", narrative_node)

    # 共享的後續節點
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)
    graph.add_node("background_expansion", background_world_expansion_node)
    graph.add_node("finalization", finalization_node)

    # 設定圖的入口和初始流程
    graph.set_entry_point("initialize_state")
    graph.add_edge("initialize_state", "analyze_input")

    # 新增核心的 NSFW/SFW 條件路由
    graph.add_conditional_edges(
        "analyze_input",
        route_based_on_intent,
        {
            "nsfw_path": "generate_nsfw_response", # 如果是 NSFW，走直通路徑
            "sfw_path": "expansion_decision"      # 如果是 SFW，走原有的 LORE 擴展決策路徑
        }
    )

    # 定義 NSFW 路徑的流程：直通鏈 -> 驗證與淨化
    graph.add_edge("generate_nsfw_response", "validate_and_rewrite")

    # 定義 SFW 路徑的完整流程
    graph.add_conditional_edges(
        "expansion_decision",
        route_expansion,
        {
            "expand_lore": "scene_and_action_analysis",
            "skip_expansion": "planning"
        }
    )
    graph.add_edge("scene_and_action_analysis", "planning")
    graph.add_edge("planning", "tool_execution")
    graph.add_edge("tool_execution", "narrative")
    graph.add_edge("narrative", "validate_and_rewrite")

    # 兩條路徑在此匯合後的共享流程
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", "background_expansion")
    graph.add_edge("background_expansion", "finalization")
    graph.add_edge("finalization", END)
    
    return graph.compile()
# 函式：創建主回應圖 (v7.2 - NameError 修正)

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
