# src/graph.py 的中文註釋(v30.0 - 信息注入式架构)
# 更新紀錄:
# v30.0 (2025-10-05): [重大架構重構] 根据最终确立的 v7.0 蓝图，彻底重写了整个对话图。废弃了所有基于 TurnPlan JSON 的复杂规划和渲染节点。新的“信息注入式架构”流程更线性、更简单：1. 感知与信息收集。 2. (全新) 前置工具调用，用于处理明确的状态变更。 3. 将所有信息（LORE、记忆、工具结果）组装成一个巨大的 world_snapshot 上下文。 4. (全新) 单一的最终生成节点，将 world_snapshot 和用户指令直接交给一个由 00_supreme_directive.txt 驱动的强大 LLM 进行一步到位的自由创作。每个与 API 交互的节点都内置了强大的“功能重建”式备援方案。
# v22.0 (2025-09-22): [災難性BUG修復] 解决了因重命名渲染节点导致的 NameError。
# v21.1 (2025-09-10): [災難性BUG修復] 恢复了所有被先前版本错误省略的 `SetupGraph` 相关节点。

import sys
import asyncio
import json
import re
from typing import Dict, List, Literal, Optional, Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, END

from .ai_core import AILover
from .logger import logger
from .graph_state import ConversationGraphState, SetupGraphState
from . import lore_book, tools
from .schemas import (CharacterProfile, ExpansionDecision, 
                      UserInputAnalysis, SceneAnalysisResult, SceneCastingResult, 
                      WorldGenesisResult, IntentClassificationResult, StyleAnalysisResult,
                      CharacterQuantificationResult, ToolCall)
from .tool_context import tool_context
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- [v30.0 新架构] 主對話圖 (Main Conversation Graph) 的節點 ---

# 函式：[新] 场景感知节点
async def perceive_scene_node(state: ConversationGraphState) -> Dict:
    """[1] 分析用户输入，推断目标地点和视角模式（近景/远景）。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|1) Node: perceive_scene -> 正在感知场景...")

    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|1) ai_core.profile 未加载，无法感知场景。")
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='错误：AI profile 未加载。', action_summary=user_input)}
    
    # Plan A: 嘗試使用 LLM 進行智能分析
    try:
        scene_analysis_chain = ai_core.get_scene_analysis_chain()
        all_lores = await lore_book.get_all_lores_for_user(user_id)
        
        # 準備一個輕量級的上下文，只包含 NPC 和他們的位置
        scene_context_lores = [
            {"name": lore.content.get("name"), "location_path": lore.content.get("location_path")}
            for lore in all_lores if lore.category == 'npc_profile'
        ]
        scene_context_json_str = json.dumps(scene_context_lores, ensure_ascii=False, indent=2)

        # 傳入當前玩家的物理位置作為備用參考
        current_location_path_str = " > ".join(ai_core.profile.game_state.location_path)

        raw_analysis = await ai_core.ainvoke_with_rotation(
            scene_analysis_chain, 
            {
                "user_input": user_input, 
                "scene_context_json": scene_context_json_str,
                "current_location_path_str": current_location_path_str
            },
            retry_strategy='euphemize'
        )
        
        # 使用 Python 進行最終的邏輯校準
        final_analysis = ai_core.calibrate_scene_analysis(raw_analysis)
        if not final_analysis: raise Exception("场景分析校准失败")

    except Exception as e:
        # Plan B (備援): 如果 LLM 分析失敗，則退回到基於關鍵詞的簡單判斷
        logger.warning(f"[{user_id}] (Graph|1) 场景感知 Plan A (LLM) 失败: {e}。启动 Plan B (关键词备援)...")
        is_remote = any(kw in user_input for kw in ["觀察", "看看", "描述"])
        final_analysis = SceneAnalysisResult(
            viewing_mode='remote' if is_remote else 'local',
            reasoning="备援：基于关键词判断。",
            target_location_path=None, # 备援模式下无法提取路径
            action_summary=user_input
        )

    # 無論使用何種方案，都更新並持久化遊戲狀態
    await ai_core._update_viewing_mode(final_analysis)
    
    logger.info(f"[{user_id}] (Graph|1) 场景感知完成。最终视角: '{ai_core.profile.game_state.viewing_mode}'")
    return {"scene_analysis": final_analysis}
# 函式：[新] 场景感知节点

# 函式：[新] 記憶與 LORE 查詢節點
async def retrieve_and_query_node(state: ConversationGraphState) -> Dict:
    """[2] 清洗使用者輸入，檢索 RAG 記憶，並查詢所有相關的 LORE。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|2) Node: retrieve_and_query -> 正在檢索記憶與查詢LORE...")
    scene_analysis = state['scene_analysis']
    
    # 步驟 1: 清洗輸入以用於查詢
    sanitized_query = user_input
    try:
        # 使用更強大的文學評論家鏈來進行源頭清洗
        literary_chain = ai_core.get_literary_euphemization_chain()
        result = await ai_core.ainvoke_with_rotation(literary_chain, {"dialogue_history": user_input}, retry_strategy='euphemize')
        if result:
            sanitized_query = result.content if hasattr(result, 'content') else str(result)
    except Exception:
        logger.warning(f"[{user_id}] (Graph|2) 源頭清洗失敗，將使用原始輸入進行查詢。")

    # 步驟 2: 檢索 RAG 記憶
    rag_context_str = "沒有檢索到相關的長期記憶。"
    # 只有在輸入較長或包含疑問詞時才進行檢索，以節省配額
    if len(user_input) > 10 or any(kw in user_input for kw in ["誰", "什麼", "回憶", "記得"]):
        logger.info(f"[{user_id}] (Graph|2) 輸入較複雜，執行 RAG 檢索...")
        rag_context_str = await ai_core.retrieve_and_summarize_memories(sanitized_query)
    else:
        logger.info(f"[{user_id}] (Graph|2) 輸入為簡單指令，跳過 RAG 檢索以節省配額。")

    # 步驟 3: 查詢結構化 LORE
    is_remote = scene_analysis.viewing_mode == 'remote'
    final_lores = await ai_core._query_lore_from_entities(sanitized_query, is_remote_scene=is_remote)
        
    logger.info(f"[{user_id}] (Graph|2) 查詢完成。檢索到 {len(final_lores)} 條相關LORE。")
    
    return {
        "rag_context": rag_context_str,
        "raw_lore_objects": final_lores,
        "sanitized_query_for_tools": sanitized_query
    }
# 函式：[新] 記憶與 LORE 查詢節點





# graph.py 的 expansion_decision_and_execution_node 函式
# 函式：[新] LORE 擴展決策與執行節點 (v5.0 - 循環中斷)
# 更新紀錄:
# v5.0 (2025-10-10): [災難性BUG修復] 徹底重構了節點的輸出邏輯。移除了在節點末尾對 `retrieve_and_query_node` 的遞迴式調用，改為在節點內部手動合併新舊 LORE 列表後再輸出。此修改從根本上解決了因不穩定的數據流和 API 調用加倍而導致的「遞迴查詢」風暴問題。
# v4.0 (2025-10-15): [架構重構] 移除了 `character_quantification_chain`。
async def expansion_decision_and_execution_node(state: ConversationGraphState) -> Dict:
    """[3] 決策是否需要擴展 LORE，如果需要，則立即執行擴展。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    safe_query_text = state['sanitized_query_for_tools']
    raw_lore_objects = state.get('raw_lore_objects', [])
    logger.info(f"[{user_id}] (Graph|3) Node: expansion_decision_and_execution -> 正在決策是否擴展LORE...")

    lightweight_lore_json = json.dumps(
        [{"name": lore.content.get("name"), "description": lore.content.get("description")} for lore in raw_lore_objects if lore.category == 'npc_profile'],
        ensure_ascii=False
    )
    decision_chain = ai_core.get_expansion_decision_chain()
    
    decision_params = {
        "user_input": safe_query_text, 
        "existing_characters_json": lightweight_lore_json
    }
    decision = await ai_core.ainvoke_with_rotation(
        decision_chain, 
        decision_params,
        retry_strategy='euphemize'
    )

    if not decision:
        logger.warning(f"[{user_id}] (Graph|3) LORE擴展決策鏈失敗，啟動備援決策。")
        decision = ExpansionDecision(should_expand=False, reasoning="備援：決策鏈失敗，預設不擴展。")

    if not decision.should_expand:
        logger.info(f"[{user_id}] (Graph|3) 決策結果：無需擴展。理由: {decision.reasoning}")
        return {"planning_subjects": [lore.content for lore in raw_lore_objects]}

    logger.info(f"[{user_id}] (Graph|3) 決策結果：需要擴展。理由: {decision.reasoning}。正在執行LORE擴展...")
    
    newly_created_lores = [] # 用於存儲新創建的 LORE 對象
    try:
        logger.info(f"[{user_id}] (Graph|3) 擴展 Plan A: 嘗試使用原子化的主選角鏈...")
        gs = ai_core.profile.game_state
        effective_location_path = gs.remote_target_path if gs.viewing_mode == 'remote' and gs.remote_target_path else gs.location_path
        
        casting_chain = ai_core.get_scene_casting_chain()
        cast_result = await ai_core.ainvoke_with_rotation(
            casting_chain,
            {
                "world_settings": ai_core.profile.world_settings or "", 
                "current_location_path": effective_location_path, 
                "user_input": safe_query_text
            },
            retry_strategy='euphemize'
        )
        
        if not cast_result: raise Exception("原子化主選角鏈返回空值")

        # _add_cast_to_scene 現在應該返回 Lore 對象列表
        newly_created_lores = await ai_core._add_cast_to_scene(cast_result)
        created_names = [lore.content.get("name", "未知") for lore in newly_created_lores]
        logger.info(f"[{user_id}] (Graph|3) 擴展 Plan A 成功，創建了 {len(created_names)} 位新角色。")
        
    except Exception as e:
        logger.warning(f"[{user_id}] (Graph|3) 擴展 Plan A 失敗: {e}。啟動【Gemini子任務鏈備援】...")
        newly_created_lores = await ai_core.gemini_subtask_expansion_fallback(safe_query_text)
        if newly_created_lores:
             logger.info(f"[{user_id}] (Graph|3) 子任務鏈備援成功，創建了 {len(newly_created_lores)} 位新角色。")
        else:
             logger.error(f"[{user_id}] (Graph|3) 子任務鏈備援最終失敗。")
    
    # [v5.0 核心修正] 在節點內部合併新舊 LORE，然後直接輸出
    final_lore_objects = raw_lore_objects + newly_created_lores
    # 去重
    final_lores_map = {lore.key: lore for lore in final_lore_objects}
    
    return {"planning_subjects": [lore.content for lore in final_lores_map.values()]}
# graph.py 的 expansion_decision_and_execution_node 函式




# 函式：[新] 前置工具調用節點
async def preemptive_tool_call_node(state: ConversationGraphState) -> Dict:
    """[4] (全新) 判斷並執行使用者指令中明確的、需要改變世界狀態的動作。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|4) Node: preemptive_tool_call -> 正在解析前置工具調用...")

    # Plan A: 嘗試使用 LLM 解析工具調用
    tool_parsing_chain = ai_core.get_preemptive_tool_parsing_chain()
    tool_call_plan = await ai_core.ainvoke_with_rotation(
        tool_parsing_chain,
        {"user_input": user_input, "character_list_str": ", ".join([ps.get("name", "") for ps in state.get("planning_subjects", [])])},
        retry_strategy='euphemize'
    )
    
    # 簡單的備援：如果 LLM 無法解析，就認為沒有工具調用
    if not tool_call_plan or not tool_call_plan.plan:
        logger.info(f"[{user_id}] (Graph|4) 未解析到明確的工具調用。")
        return {"tool_results": "系統事件：無前置工具被調用。"}

    logger.info(f"[{user_id}] (Graph|4) 解析到 {len(tool_call_plan.plan)} 個工具調用，準備執行...")
    
    # 執行工具
    tool_context.set_context(user_id, ai_core)
    try:
        # 這是一個簡化的 TurnPlan，只用於工具執行
        from .schemas import TurnPlan, CharacterAction
        simple_turn_plan = TurnPlan(
            character_actions=[
                CharacterAction(character_name="system", reasoning="preemptive tool execution", tool_call=call) 
                for call in tool_call_plan.plan
            ]
        )
        results_summary = await ai_core._execute_planned_actions(simple_turn_plan)
    except Exception as e:
        logger.error(f"[{user_id}] (Graph|4) 前置工具執行時發生錯誤: {e}", exc_info=True)
        results_summary = f"系統事件：工具執行時發生嚴重錯誤: {e}"
    finally:
        tool_context.set_context(None, None)
    
    logger.info(f"[{user_id}] (Graph|4) 前置工具執行完畢。")
    return {"tool_results": results_summary}
# 函式：[新] 前置工具調用節點

# 函式：[新] 世界快照組裝節點
async def assemble_world_snapshot_node(state: ConversationGraphState) -> Dict:
    """[5] (核心) 匯集所有資訊，使用模板格式化成 world_snapshot 字符串。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|5) Node: assemble_world_snapshot -> 正在組裝世界快照...")
    
    planning_subjects = state.get("planning_subjects", [])
    tool_results = state.get("tool_results", "")
    
    npc_context_str = "\n".join([f"- **{npc.get('name', '未知NPC')}**: {npc.get('description', '無描述')}" for npc in planning_subjects])
    if not npc_context_str:
        npc_context_str = "當前場景沒有已知的特定角色。"

    # 將工具結果也加入到場景事實中
    if tool_results and "無前置工具被調用" not in tool_results:
        npc_context_str += f"\n\n--- 本回合即時事件 ---\n{tool_results}"

    gs = ai_core.profile.game_state
    
    context_vars = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', "無相關長期記憶。"),
        'possessions_context': f"團隊庫存: {', '.join(gs.inventory) or '空的'}",
        'quests_context': "當前無任務。",
        'location_context': f"當前地點: {' > '.join(gs.location_path)}",
        'npc_context': npc_context_str,
        'relevant_npc_context': "請參考上方在場角色列表。",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': gs.viewing_mode,
        'remote_target_path_str': " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定",
    }
    
    final_world_snapshot = ai_core.world_snapshot_template.format(**context_vars)
    
    logger.info(f"[{user_id}] (Graph|5) 世界快照組裝完畢。")
    return {"world_snapshot": final_world_snapshot}
# 函式：[新] 世界快照組裝節點

# 函式：[新] 最终生成节点
async def final_generation_node(state: ConversationGraphState) -> Dict:
    """[6] (全新) 组装所有上下文，并调用统一生成链来一步到位地创作小说。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    world_snapshot = state['world_snapshot']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|6) Node: final_generation -> 启动最终生成流程...")

    historical_context = await _get_summarized_chat_history(ai_core, user_id)
    
    prompt_template_runnable = ai_core.get_unified_generation_chain()
    
    # 這是我們的最終武器，啟用所有最高強度的策略
    final_response_raw = await ai_core.ainvoke_with_rotation(
        prompt_template_runnable,
        {
            "core_protocol_prompt": ai_core.core_protocol_prompt,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "historical_context": historical_context,
            "world_snapshot": world_snapshot,
            "latest_user_input": user_input
        },
        retry_strategy='force', # 核心：如果被審查，則使用指令轟炸策略重試
        use_degradation=True    # 核心：如果 gemini-2.5-pro 失敗，則自動降級
    )

    final_response = final_response_raw.content if hasattr(final_response_raw, 'content') else str(final_response_raw)

    if not final_response or not final_response.strip():
        logger.critical(f"[{user_id}] (Graph|6) 核心生成链在所有策略（包括 'force'）之后最终失败！")
        final_response = "（抱歉，我好像突然断线了，脑海中一片空白...）"
        
    logger.info(f"[{user_id}] (Graph|6) 最终生成流程完成。")
    return {"llm_response": final_response}
# 函式：[新] 最终生成节点

# 函式：驗證、學習與持久化節點
async def validate_and_persist_node(state: ConversationGraphState) -> Dict:
    """[7] 清理文本、事後 LORE 提取、保存對話歷史。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    llm_response_raw = state['llm_response']
    logger.info(f"[{user_id}] (Graph|7) Node: validate_and_persist -> 正在驗證、學習與持久化...")

    # 1. 驗證與清理
    if hasattr(llm_response_raw, 'content'):
        llm_response = llm_response_raw.content
    else:
        llm_response = str(llm_response_raw)
        logger.warning(f"[{user_id}] (Graph|7) LLM 回應不是 AIMessage 物件，直接轉換為字符串。")

    clean_response = llm_response.strip()
    
    # 2. 學習 (事後 LORE 提取)
    # 為了不阻塞使用者，我們將其作為一個背景任務啟動
    asyncio.create_task(ai_core._background_lore_extraction(user_input, clean_response))

    # 3. 持久化
    if clean_response and "抱歉" not in clean_response:
        chat_history_manager = ai_core.session_histories.setdefault(user_id, ChatMessageHistory())
        chat_history_manager.add_user_message(user_input)
        chat_history_manager.add_ai_message(clean_response)
        
        last_interaction_text = f"使用者: {user_input}\n\nAI:\n{clean_response}"
        # 也作為背景任務啟動
        asyncio.create_task(ai_core._save_interaction_to_dbs(last_interaction_text))
        
        logger.info(f"[{user_id}] (Graph|7) 對話歷史已更新並準備保存到 DB。")
    
    logger.info(f"[{user_id}] (Graph|7) 狀態持久化完成。")
    
    return {"final_output": clean_response}
# 函式：驗證、學習與持久化節點

# 函式：獲取摘要後的對話歷史 (v28.0 - 終極備援修正)
async def _get_summarized_chat_history(ai_core: AILover, user_id: str, num_messages: int = 8) -> str:
    """
    提取並摘要最近的對話歷史，並內建一個強大的、基於「文學評論家」重寫的 NSFW 內容安全備援機制。
    """
    if not ai_core.profile: return "（沒有最近的對話歷史）"
    chat_history_manager = ai_core.session_histories.get(user_id, ChatMessageHistory())
    if not chat_history_manager.messages:
        return "（沒有最近的對話歷史）"
        
    recent_messages = chat_history_manager.messages[-num_messages:]
    if not recent_messages:
        return "（沒有最近的對話歷史）"

    raw_history_text = "\n".join([f"{'使用者' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in recent_messages])

    try:
        # 尝试直接摘要
        literary_chain = ai_core.get_literary_euphemization_chain() # 使用这个更强大的链进行摘要
        summary = await ai_core.ainvoke_with_rotation(literary_chain, {"dialogue_history": raw_history_text}, retry_strategy='euphemize')

        if not summary or not summary.strip():
            raise Exception("Summarization returned empty content.")
            
        return f"【最近對話摘要】:\n{summary}"

    except Exception as e:
        logger.error(f"[{user_id}] (History Summarizer) 生成摘要時發生錯誤: {e}。返回中性提示。")
        return "（歷史對話摘要因错误而生成失败，部分上下文可能缺失。）"
# 函式：獲取摘要後的對話歷史 (v28.0 - 終極備援修正)

# --- [v30.0 新架构] 图的构建 ---
def create_main_response_graph() -> StateGraph:
    """创建并连接所有节点，构建最终的对话图。"""
    graph = StateGraph(ConversationGraphState)
    
    # 注册所有节点
    graph.add_node("perceive_scene", perceive_scene_node)
    graph.add_node("retrieve_and_query", retrieve_and_query_node)
    graph.add_node("expansion_decision_and_execution", expansion_decision_and_execution_node)
    graph.add_node("preemptive_tool_call", preemptive_tool_call_node)
    graph.add_node("assemble_world_snapshot", assemble_world_snapshot_node)
    graph.add_node("final_generation", final_generation_node)
    graph.add_node("validate_and_persist", validate_and_persist_node)
    
    # 设置入口点
    graph.set_entry_point("perceive_scene")
    
    # 连接流程
    graph.add_edge("perceive_scene", "retrieve_and_query")
    graph.add_edge("retrieve_and_query", "expansion_decision_and_execution")
    graph.add_edge("expansion_decision_and_execution", "preemptive_tool_call")
    graph.add_edge("preemptive_tool_call", "assemble_world_snapshot")
    graph.add_edge("assemble_world_snapshot", "final_generation")
    graph.add_edge("final_generation", "validate_and_persist")
    graph.add_edge("validate_and_persist", END)
    
    return graph.compile()

# --- 旧的 Setup Graph (保持不变，用于 /start 流程) ---
async def process_canon_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    canon_text = state['canon_text']
    logger.info(f"[{user_id}] (Setup Graph|1/4) Node: process_canon -> 節點已啟動。")
    try:
        if canon_text:
            logger.info(f"[{user_id}] (Setup Graph|1/4) 檢測到世界聖經文本 (長度: {len(canon_text)})，開始處理...")
            logger.info(f"[{user_id}] (Setup Graph|1/4) 步驟 A: 正在向量化文本...")
            await ai_core.add_canon_to_vector_store(canon_text)
            logger.info(f"[{user_id}] (Setup Graph|1/4) 步驟 A: 向量化儲存完成。")
            logger.info(f"[{user_id}] (Setup Graph|1/4) 步驟 B: 正在進行 LORE 智能解析...")
            await ai_core.parse_and_create_lore_from_canon(None, canon_text, is_setup_flow=True)
            logger.info(f"[{user_id}] (Setup Graph|1/4) 步驟 B: LORE 智能解析完成。")
        else:
            logger.info(f"[{user_id}] (Setup Graph|1/4) 未提供世界聖經文本，跳過處理。")
        logger.info(f"[{user_id}] (Setup Graph|1/4) Node: process_canon -> 節點執行成功。")
    except Exception as e:
        logger.error(f"[{user_id}] (Setup Graph|1/4) Node: process_canon -> 執行時發生嚴重錯誤: {e}", exc_info=True)
    finally:
        delay_seconds = 5.0
        logger.info(f"[{user_id}] (Setup Graph|1/4|Flow Control) 為平滑 API 請求，將強制等待 {delay_seconds} 秒後進入下一節點...")
        await asyncio.sleep(delay_seconds)
    return {}

async def complete_profiles_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Setup Graph|2/4) Node: complete_profiles_node -> 節點已啟動，準備補完角色檔案...")
    try:
        if not ai_core.profile:
            logger.error(f"[{user_id}] (Setup Graph|2/4) ai_core.profile 為空，無法繼續。")
            return {}
        completion_chain = ai_core.get_profile_completion_chain()
        literary_chain = ai_core.get_literary_euphemization_chain()
        async def _safe_complete_profile(original_profile: CharacterProfile) -> CharacterProfile:
            try:
                safe_profile_data = original_profile.model_dump()
                description_to_clean = safe_profile_data.get('description', '')
                appearance_to_clean = safe_profile_data.get('appearance', '')
                tasks_to_clean = {}
                if description_to_clean.strip(): tasks_to_clean['description'] = literary_chain.ainvoke({"dialogue_history": description_to_clean})
                if appearance_to_clean.strip(): tasks_to_clean['appearance'] = literary_chain.ainvoke({"dialogue_history": appearance_to_clean})
                if tasks_to_clean:
                    cleaned_results = await asyncio.gather(*tasks_to_clean.values(), return_exceptions=True)
                    results_dict = dict(zip(tasks_to_clean.keys(), cleaned_results))
                    if 'description' in results_dict and isinstance(results_dict['description'], str): safe_profile_data['description'] = results_dict['description']
                    if 'appearance' in results_dict and isinstance(results_dict['appearance'], str): safe_profile_data['appearance'] = results_dict['appearance']
                completed_safe_profile = await ai_core.ainvoke_with_rotation(completion_chain, {"profile_json": json.dumps(safe_profile_data, ensure_ascii=False)}, retry_strategy='euphemize')
                if not completed_safe_profile: return original_profile
                original_data = original_profile.model_dump()
                completed_data = completed_safe_profile.model_dump()
                for key, value in completed_data.items():
                    if not original_data.get(key) or original_data.get(key) in [[], {}, "未設定", "未知", ""]:
                        if value: original_data[key] = value
                original_data['description'] = original_profile.description
                original_data['appearance'] = original_profile.appearance
                original_data['name'] = original_profile.name
                return CharacterProfile.model_validate(original_data)
            except Exception as e:
                logger.error(f"[{user_id}] 為角色 '{original_profile.name}' 進行安全補完時發生錯誤: {e}", exc_info=True)
                return original_profile
        completed_user_profile_task = _safe_complete_profile(ai_core.profile.user_profile)
        completed_ai_profile_task = _safe_complete_profile(ai_core.profile.ai_profile)
        final_user_profile, final_ai_profile = await asyncio.gather(completed_user_profile_task, completed_ai_profile_task)
        await ai_core.update_and_persist_profile({'user_profile': final_user_profile.model_dump(), 'ai_profile': final_ai_profile.model_dump()})
        logger.info(f"[{user_id}] (Setup Graph|2/4) Node: complete_profiles_node -> 節點執行成功。")
    except Exception as e:
        logger.error(f"[{user_id}] (Setup Graph|2/4) Node: complete_profiles_node -> 執行時發生嚴重錯誤: {e}", exc_info=True)
    finally:
        delay_seconds = 5.0
        logger.info(f"[{user_id}] (Setup Graph|2/4|Flow Control) 為平滑 API 請求，將強制等待 {delay_seconds} 秒後進入下一節點...")
        await asyncio.sleep(delay_seconds)
    return {}

# graph.py 的 world_genesis_node 函式
async def world_genesis_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Setup Graph|3/4) Node: world_genesis -> 節點已啟動...")
    genesis_result = None
    try:
        if not ai_core.profile: raise Exception("AI Profile is not loaded.")
        genesis_chain = ai_core.get_world_genesis_chain()
        
        # [核心修正] 補全缺失的 world_settings 參數
        genesis_params = {
            "world_settings": ai_core.profile.world_settings or "未設定", 
            "username": ai_core.profile.user_profile.name, 
            "ai_name": ai_core.profile.ai_profile.name
        }
        
        genesis_result = await ai_core.ainvoke_with_rotation(
            genesis_chain, 
            genesis_params, 
            retry_strategy='force'
        )

        if not genesis_result: raise Exception("世界創世鏈返回了空結果。")
        gs = ai_core.profile.game_state
        gs.location_path = genesis_result.location_path
        await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})
        await lore_book.add_or_update_lore(user_id, 'location_info', " > ".join(genesis_result.location_path), genesis_result.location_info.model_dump())
        for npc in genesis_result.initial_npcs:
            npc_key = " > ".join(genesis_result.location_path) + f" > {npc.name}"
            await lore_book.add_or_update_lore(user_id, 'npc_profile', npc_key, npc.model_dump())
        logger.info(f"[{user_id}] (Setup Graph|3/4) Node: world_genesis -> 節點執行成功。")
    except Exception as e:
        logger.error(f"[{user_id}] (Setup Graph|3/4) Node: world_genesis -> 執行時發生嚴重錯誤: {e}", exc_info=True)
    finally:
        delay_seconds = 5.0
        logger.info(f"[{user_id}] (Setup Graph|3/4|Flow Control) 為平滑 API 請求，將強制等待 {delay_seconds} 秒後進入下一節點...")
        await asyncio.sleep(delay_seconds)
    return {"genesis_result": genesis_result}
# graph.py 的 world_genesis_node 函式




async def generate_opening_scene_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    opening_scene = ""
    logger.info(f"[{user_id}] (Setup Graph|4/4) Node: generate_opening_scene -> 節點已啟動...")
    try:
        opening_scene = await ai_core.generate_opening_scene()
        if not opening_scene or not opening_scene.strip():
            opening_scene = (f"在一片柔和的光芒中...")
        logger.info(f"[{user_id}] (Setup Graph|4/4) Node: generate_opening_scene -> 節點執行成功。")
    except Exception as e:
        logger.error(f"[{user_id}] (Setup Graph|4/4) Node: generate_opening_scene -> 執行時發生嚴重錯誤: {e}", exc_info=True)
        opening_scene = (f"在一片柔和的光芒中...")
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



