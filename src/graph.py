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
# 更新紀錄:
# v1.0 (2025-10-05): [重大架構重構] 根据最终确立的 v7.0 蓝图，彻底重写了整个对话图。废弃了所有基于 TurnPlan JSON 的复杂规划和渲染节点。新的“信息注入式架构”流程更线性、更简单：1. 感知与信息收集。 2. (全新) 前置工具调用，用于处理明确的状态变更。 3. 将所有信息（LORE、记忆、工具结果）组装成一个巨大的 world_snapshot 上下文。 4. (全新) 单一的最终生成节点，将 world_snapshot 和用户指令直接交给一个由 00_supreme_directive.txt 驱动的强大 LLM 进行一步到位的自由创作。每个与 API 交互的节点都内置了强大的“功能重建”式备援方案。
# v2.0 (2025-10-15): [災難性BUG修復] 引入了【上下文感知的視角保持】策略，以解決在連續性指令（如“继续”）下，場景視角被錯誤重置的問題。
async def perceive_scene_node(state: ConversationGraphState) -> Dict:
    """[1] 分析用户输入，推断目标地点和视角模式（近景/远景）。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content.strip()
    logger.info(f"[{user_id}] (Graph|1) Node: perceive_scene -> 正在感知场景...")

    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|1) ai_core.profile 未加载，无法感知场景。")
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='错误：AI profile 未加载。', action_summary=user_input)}

    gs = ai_core.profile.game_state
    
    # [v2.0 核心修正] 步驟 1: 處理連續性指令
    continuation_keywords = ["继续", "繼續", "然後呢", "接下來", "go on", "continue"]
    if any(user_input.lower().startswith(kw) for kw in continuation_keywords):
        logger.info(f"[{user_id}] (Graph|1) 檢測到連續性指令，將繼承上一輪的場景狀態。")
        # 直接繼承上一輪的狀態，無需任何分析
        scene_analysis = SceneAnalysisResult(
            viewing_mode=gs.viewing_mode,
            reasoning="繼承上一輪的場景狀態。",
            target_location_path=gs.remote_target_path,
            action_summary=user_input
        )
        return {"scene_analysis": scene_analysis}

    # [v2.0 核心修正] 步驟 2: 處理模式切換和目標更新
    new_viewing_mode = 'local'
    new_target_path = None

    # 嘗試使用 LLM 進行智能推斷
    location_chain = ai_core.get_contextual_location_chain()
    scene_context_lores = [lore.content for lore in state.get('raw_lore_objects_for_view_decision', []) if lore.category == 'npc_profile']
    scene_context_json_str = json.dumps(scene_context_lores, ensure_ascii=False, indent=2)

    location_result = await ai_core.ainvoke_with_rotation(
        location_chain, 
        {"user_input": user_input, "world_settings": ai_core.profile.world_settings or "未设定", "scene_context_json": scene_context_json_str},
        retry_strategy='euphemize'
    )

    if location_result and location_result.location_path:
        logger.info(f"[{user_id}] (Graph|1) LLM 感知成功。推斷出的目標地點: {location_result.location_path}")
        new_target_path = location_result.location_path
        new_viewing_mode = 'remote'
    
    # [v2.0 核心修正] 步驟 3: 應用「遠程優先」的狀態保持邏輯
    final_viewing_mode = gs.viewing_mode
    final_target_path = gs.remote_target_path

    if gs.viewing_mode == 'remote':
        # 當前處於遠程模式
        # 檢查是否存在明確的返回本地的信號
        is_explicit_local_move = any(user_input.startswith(kw) for kw in ["去", "前往", "移動到", "旅行到"])
        is_direct_ai_interaction = ai_core.profile.ai_profile.name in user_input
        
        if is_explicit_local_move or is_direct_ai_interaction:
            # 信號明確：切換回本地
            final_viewing_mode = 'local'
            final_target_path = None
            logger.info(f"[{user_id}] (Graph|1) 檢測到明確的本地指令，導演視角從 'remote' 切換回 'local'。")
        elif new_viewing_mode == 'remote' and new_target_path and new_target_path != gs.remote_target_path:
            # 信號不明確，但探測到新的遠程目標：更新目標
            final_target_path = new_target_path
            logger.info(f"[{user_id}] (Graph|1) 在遠程模式下，更新了觀察目標地點為: {final_target_path}")
        else:
            # 保持遠程模式和當前目標不變
            logger.info(f"[{user_id}] (Graph|1) 未檢測到本地切換信號，導演視角保持為 'remote'。")
    else: # gs.viewing_mode == 'local'
        # 當前處於本地模式，檢查是否需要切換到遠程
        if new_viewing_mode == 'remote' and new_target_path:
            final_viewing_mode = 'remote'
            final_target_path = new_target_path
            logger.info(f"[{user_id}] (Graph|1) 檢測到遠程描述指令，導演視角從 'local' 切換到 'remote'。目標: {final_target_path}")

    # 更新並持久化遊戲狀態
    if gs.viewing_mode != final_viewing_mode or gs.remote_target_path != final_target_path:
        gs.viewing_mode = final_viewing_mode
        gs.remote_target_path = final_target_path
        await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})
    
    scene_analysis = SceneAnalysisResult(
        viewing_mode=gs.viewing_mode,
        reasoning=f"場景感知完成。",
        target_location_path=gs.remote_target_path,
        action_summary=user_input
    )
    return {"scene_analysis": scene_analysis}
# 函式：[新] 场景感知节点





# 函式：[新] 記憶與 LORE 查詢節點
# 更新紀錄:
# v2.0 (2025-10-05): [重大架構重構] 根据最终确立的 v7.0 蓝图，彻底重写了整个对话图。废弃了所有基于 TurnPlan JSON 的复杂规划和渲染节点。新的“信息注入式架构”流程更线性、更简单：1. 感知与信息收集。 2. (全新) 前置工具调用，用于处理明确的状态变更。 3. 将所有信息（LORE、记忆、工具结果）组装成一个巨大的 world_snapshot 上下文。 4. (全新) 单一的最终生成节点，将 world_snapshot 和用户指令直接交给一个由 00_supreme_directive.txt 驱动的强大 LLM 进行一步到位的自由创作。每个与 API 交互的节点都内置了强大的“功能重建”式备援方案。
# v2.1 (2025-10-15): [性能優化] 新增了對短指令的判斷，避免對 "坐下" 等簡單指令執行不必要的 RAG 檢索，以節省 Embedding API 配額。
# v3.0 (2025-10-15): [功能優化] 此函式現在調用具有場景感知能力的 `_query_lore_from_entities`，查詢結果更精準。
async def retrieve_and_query_node(state: ConversationGraphState) -> Dict:
    """[2] 清洗使用者輸入，檢索 RAG 記憶，並查詢所有相關的 LORE。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    scene_analysis = state['scene_analysis']
    logger.info(f"[{user_id}] (Graph|2) Node: retrieve_and_query -> 正在檢索記憶與查詢LORE...")

    # 源頭清洗
    sanitized_query = user_input
    try:
        literary_chain = ai_core.get_literary_euphemization_chain()
        result = await ai_core.ainvoke_with_rotation(literary_chain, {"dialogue_history": user_input}, retry_strategy='euphemize')
        if result:
            sanitized_query = result.content if hasattr(result, 'content') else str(result)
    except Exception:
        logger.warning(f"[{user_id}] (Graph|2) 源頭清洗失敗，將使用原始輸入進行查詢。")

    # [v2.1 核心修正] RAG 檢索優化
    rag_context_str = "沒有檢索到相關的長期記憶。"
    if len(user_input) > 10 or any(kw in user_input for kw in ["誰", "什麼", "回憶", "記得"]):
        logger.info(f"[{user_id}] (Graph|2) 輸入較複雜，執行 RAG 檢索...")
        rag_context_str = await ai_core.retrieve_and_summarize_memories(sanitized_query)
    else:
        logger.info(f"[{user_id}] (Graph|2) 輸入為簡單指令，跳過 RAG 檢索以節省配額。")


    # [v3.0 核心修正] LORE 查詢現在具有場景感知能力
    is_remote = scene_analysis.viewing_mode == 'remote'
    final_lores = await ai_core._query_lore_from_entities(sanitized_query, is_remote_scene=is_remote)
        
    logger.info(f"[{user_id}] (Graph|2) 查詢完成。檢索到 {len(final_lores)} 條相關LORE。")
    
    return {
        "rag_context": rag_context_str,
        "raw_lore_objects": final_lores,
        "sanitized_query_for_tools": sanitized_query
    }
# 函式：[新] 記憶與 LORE 查詢節點




# 函式：[新] LORE 擴展決策與執行節點
# 更新紀錄:
# v3.0 (2025-10-05): [重大架構重構] 根据最终确立的 v7.0 蓝图，彻底重写了整个对话图。废弃了所有基于 TurnPlan JSON 的复杂规划和渲染节点。新的“信息注入式架构”流程更线性、更简单：1. 感知与信息收集。 2. (全新) 前置工具调用，用于处理明确的状态变更。 3. 将所有信息（LORE、记忆、工具结果）组装成一个巨大的 world_snapshot 上下文。 4. (全新) 单一的最终生成节点，将 world_snapshot 和用户指令直接交给一个由 00_supreme_directive.txt 驱动的强大 LLM 进行一步到位的自由创作。每个与 API 交互的节点都内置了强大的“功能重建”式备援方案。
# v2.0 (2025-10-07): [架構重構] 此节点的职责被扩展。它现在负责组装所有不同来源的上下文（RAG 记忆、短期对话历史、世界快照），并严格按照“历史 -> 事实 -> 指令”的顺序，将它们填充到新的提示词模板中，然后调用核心生成链。
# v3.1 (2025-10-15): [災難性BUG修復] 在調用 `expansion_decision_chain` 時，補全了缺失的 `username` 和 `ai_name` 參數，解決了 `KeyError`。
async def expansion_decision_and_execution_node(state: ConversationGraphState) -> Dict:
    """[3] 決策是否需要擴展 LORE，如果需要，則立即執行擴展。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    safe_query_text = state['sanitized_query_for_tools']
    raw_lore_objects = state.get('raw_lore_objects', [])
    logger.info(f"[{user_id}] (Graph|3) Node: expansion_decision_and_execution -> 正在決策是否擴展LORE...")

    # Plan A: 嘗試使用 LLM 進行決策
    lightweight_lore_json = json.dumps(
        [{"name": lore.content.get("name"), "description": lore.content.get("description")} for lore in raw_lore_objects if lore.category == 'npc_profile'],
        ensure_ascii=False
    )
    decision_chain = ai_core.get_expansion_decision_chain()
    
    # [v3.1 核心修正] 準備調用 `expansion_decision_chain` 所需的所有參數
    decision_params = {
        "user_input": safe_query_text, 
        "existing_characters_json": lightweight_lore_json, 
        "examples": "", # 範例目前在提示詞中硬編碼，未來可以動態注入
        "username": ai_core.profile.user_profile.name,
        "ai_name": ai_core.profile.ai_profile.name
    }
    decision = await ai_core.ainvoke_with_rotation(
        decision_chain, 
        decision_params,
        retry_strategy='euphemize'
    )

    if not decision:
        # Plan B (備援): LLM 失敗，啟動基於 LORE 覆蓋率的備援決策
        logger.warning(f"[{user_id}] (Graph|3) LORE擴展決策鏈失敗，啟動【基於LORE覆蓋率的備援決策】。")
        if len(raw_lore_objects) < 3 and len(safe_query_text) > 15:
            decision = ExpansionDecision(should_expand=True, reasoning="備援：場景中角色較少且使用者輸入較長，可能需要新角色。")
        else:
            decision = ExpansionDecision(should_expand=False, reasoning="備援：決策鏈失敗，預設不擴展。")

    if not decision.should_expand:
        logger.info(f"[{user_id}] (Graph|3) 決策結果：無需擴展。理由: {decision.reasoning}")
        return {"planning_subjects": [lore.content for lore in raw_lore_objects]}

    # --- 如果需要擴展，則執行擴展 ---
    logger.info(f"[{user_id}] (Graph|3) 決策結果：需要擴展。理由: {decision.reasoning}。正在執行LORE擴展...")
    
    # Plan A: 嘗試使用主 casting_chain
    try:
        logger.info(f"[{user_id}] (Graph|3) 擴展 Plan A: 嘗試使用主選角鏈...")
        quantification_chain = ai_core.get_character_quantification_chain()
        quant_result = await ai_core.ainvoke_with_rotation(quantification_chain, {"user_input": safe_query_text}, retry_strategy='euphemize')
        
        if quant_result and quant_result.character_descriptions:
            gs = ai_core.profile.game_state
            effective_location_path = gs.remote_target_path if gs.viewing_mode == 'remote' and gs.remote_target_path else gs.location_path
            
            casting_chain = ai_core.get_scene_casting_chain()
            cast_result = await ai_core.ainvoke_with_rotation(
                casting_chain,
                {"world_settings": ai_core.profile.world_settings or "", "current_location_path": effective_location_path, "character_descriptions_list": quant_result.character_descriptions},
                retry_strategy='euphemize'
            )
            
            if not cast_result: raise Exception("主選角鏈返回空值")

            created_names = await ai_core._add_cast_to_scene(cast_result)
            logger.info(f"[{user_id}] (Graph|3) 擴展 Plan A 成功，創建了 {len(created_names)} 位新角色。")
            
            # 獲取更新後的所有 LORE
            all_lores_after_expansion_state = await retrieve_and_query_node(state)
            return {"planning_subjects": [lore.content for lore in all_lores_after_expansion_state.get("raw_lore_objects", [])]}

    except Exception as e:
        # Plan B (備援): 主鏈失敗，啟動 Gemini 子任務鏈備援
        logger.warning(f"[{user_id}] (Graph|3) 擴展 Plan A 失敗: {e}。啟動【Gemini子任務鏈備援】...")
        newly_created_lores = await ai_core.gemini_subtask_expansion_fallback(safe_query_text)
        if newly_created_lores:
             logger.info(f"[{user_id}] (Graph|3) 子任務鏈備援成功，創建了 {len(newly_created_lores)} 位新角色。")
             all_current_lores = state.get('raw_lore_objects', [])
             all_current_lores.extend(newly_created_lores)
             return {"planning_subjects": [lore.content for lore in all_current_lores]}
        else:
             logger.error(f"[{user_id}] (Graph|3) 子任務鏈備援最終失敗。")
             return {"planning_subjects": [lore.content for lore in raw_lore_objects]}
# 函式：[新] LORE 擴展決策與執行節點




# 函式：[新] 前置工具調用節點
# 更新紀錄:
# v4.0 (2025-10-05): [重大架構重構] 根据最终确立的 v7.0 蓝图，彻底重写了整个对话图。废弃了所有基于 TurnPlan JSON 的复杂规划和渲染节点。新的“信息注入式架构”流程更线性、更简单：1. 感知与信息收集。 2. (全新) 前置工具调用，用于处理明确的状态变更。 3. 将所有信息（LORE、记忆、工具结果）组装成一个巨大的 world_snapshot 上下文。 4. (全新) 单一的最终生成节点，将 world_snapshot 和用户指令直接交给一个由 00_supreme_directive.txt 驱动的强大 LLM 进行一步到位的自由创作。每个与 API 交互的节点都内置了强大的“功能重建”式备援方案。
# v2.0 (2025-10-07): [架構重構] 此节点的职责被扩展。它现在负责组装所有不同来源的上下文（RAG 记忆、短期对话历史、世界快照），并严格按照“历史 -> 事实 -> 指令”的顺序，将它们填充到新的提示词模板中，然后调用核心生成链。
# v2.1 (2025-10-14): [災難性BUG修復] 修正了 `CharacterAction` 驗證錯誤，為系統角色添加了默認的 `action_description`。
async def preemptive_tool_call_node(state: ConversationGraphState) -> Dict:
    """[4] (全新) 判斷並執行使用者指令中明確的、需要改變世界狀態的動作。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|4) Node: preemptive_tool_call -> 正在解析前置工具調用...")

    # Plan A: 嘗試使用 LLM 解析工具調用
    tool_parsing_chain = ai_core.get_preemptive_tool_parsing_chain()
    # 確保 ai_name 作為 partial_variables 傳入
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
        # [v2.1 核心修正] 為 CharacterAction 添加一個默認的 action_description
        simple_turn_plan = TurnPlan(
            character_actions=[
                CharacterAction(
                    character_name="system", 
                    reasoning="preemptive tool execution", 
                    tool_call=call,
                    action_description=f"執行工具 {call.tool_name}." # 提供一個默認描述
                ) 
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

# 函式：[新] 世界快照組裝節點 (v2.0 - 職責簡化)
# 更新紀錄:
# v2.0 (2025-10-07): [架構重構] 根据新的信息顺序要求，此节点的职责被简化。它现在只负责将【当前场景】的所有事实（LORE、游戏状态、工具结果）格式化为 world_snapshot 字符串，不再处理历史上下文。
# v2.1 (2025-10-14): [災難性BUG修復] 增加了 `username` 和 `ai_name` 到 `context_vars`，解決 `KeyError`。
async def assemble_world_snapshot_node(state: ConversationGraphState) -> Dict:
    """[5] (核心) 匯集所有【當前場景】的信息，使用模板格式化成 world_snapshot 字符串。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|5) Node: assemble_world_snapshot -> 正在組裝【當前場景事實】...")
    
    planning_subjects = state.get("planning_subjects", [])
    tool_results = state.get("tool_results", "")
    
    npc_context_str = "\n".join([f"- **{npc.get('name', '未知NPC')}**: {npc.get('description', '無描述')}" for npc in planning_subjects])
    if not npc_context_str: npc_context_str = "當前場景沒有已知的特定角色。"

    # 將工具結果也加入到場景事實中
    if tool_results and "無前置工具被調用" not in tool_results:
        npc_context_str += f"\n\n--- 本回合即時事件 ---\n{tool_results}"

    gs = ai_core.profile.game_state
    
    # [v2.1 核心修正] 增加 username 和 ai_name
    context_vars = {
        'username': ai_core.profile.user_profile.name, # 新增
        'ai_name': ai_core.profile.ai_profile.name,     # 新增
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': "（此部分已移至歷史上下文中單獨處理）",
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
    
    logger.info(f"[{user_id}] (Graph|5) 【當前場景事實】組裝完畢。")
    return {"world_snapshot": final_world_snapshot}
# 函式：[新] 世界快照組裝節點 (v2.0 - 職責簡化)





# 函式：[新] 最终生成节点 (v2.0 - 优化信息顺序)
# 更新纪录:
# v2.0 (2025-10-07): [架構重構] 此节点的职责被扩展。它现在负责组装所有不同来源的上下文（RAG 记忆、短期对话历史、世界快照），并严格按照“历史 -> 事实 -> 指令”的顺序，将它们填充到新的提示词模板中，然后调用核心生成链。
# v3.0 (2025-10-15): [災難性BUG修復] 引入了【指令防火牆】策略。重構了 `human_prompt_template`，將用戶指令與上下文數據明確分離，並用強烈的提示詞命令 AI 絕對優先執行指令，以解決 AI 因上下文數據污染而忽略用戶指令的問題。
# v4.0 (2025-10-15): [健壯性] 引入了【無損上下文通道】，在處理連續性指令時，將上一輪的原文注入提示詞，確保劇情連貫。
async def final_generation_node(state: ConversationGraphState) -> Dict:
    """[6] (全新) 组装所有上下文，并调用统一生成链来一步到位地创作小说。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    world_snapshot = state['world_snapshot']
    user_input = state['messages'][-1].content
    rag_context = state.get('rag_context', '无相关长期记忆。')
    logger.info(f"[{user_id}] (Graph|6) Node: final_generation -> 启动最终生成流程...")

    # [v4.0 核心修正] 準備上下文
    chat_history_str = await _get_summarized_chat_history(ai_core, user_id)
    historical_context = f"{rag_context}\n\n{chat_history_str}"
    
    # [v4.0 核心修正] 檢查是否為連續性指令，並注入無損上下文
    continuation_keywords = ["继续", "繼續", "然後呢", "接下來", "go on", "continue"]
    if any(user_input.strip().lower().startswith(kw) for kw in continuation_keywords):
        last_response = state.get('last_response_text')
        if last_response:
            uncompressed_context = (
                f"\n# ==============================================================================\n"
                f"# == 📝 上一幕場景原文 (絕對參考) 📝 ==\n"
                f"# ==============================================================================\n"
                f"# 說明：這是上一回合的完整場景描述，你的任務是基於此原文繼續創作。\n"
                f"# ------------------------------------------------------------------------------\n"
                f"{last_response}\n"
                f"# =============================================================================="
            )
            # 將無損上下文放在最前面，以獲得最高注意力權重
            historical_context = uncompressed_context + "\n\n" + historical_context
            logger.info(f"[{user_id}] (Graph|6) 已成功為連續性指令注入上一輪的無損上下文。")

    prompt_template_runnable = ai_core.get_unified_generation_chain()
    
    final_chain = prompt_template_runnable
    
    final_response_raw = await ai_core.ainvoke_with_rotation(
        final_chain,
        {
            "core_protocol_prompt": ai_core.core_protocol_prompt,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "historical_context": historical_context,
            "world_snapshot": world_snapshot,
            "latest_user_input": user_input,
        },
        retry_strategy='force',
        use_degradation=True
    )

    final_response = final_response_raw.content if hasattr(final_response_raw, 'content') else str(final_response_raw)

    if not final_response or not final_response.strip():
        logger.critical(f"[{user_id}] (Graph|6) 核心生成链在指令轰炸和模型降级后最终失败！")
        final_response = "（抱歉，我好像突然断线了，脑海中一片空白... 这很可能是因为您的指令触发了无法绕过的核心内容安全限制，或者是一个暂时的、严重的 API 服务问题。请尝试用完全不同的方式表达您的意图，或稍后再试。）"
        
    logger.info(f"[{user_id}] (Graph|6) 最终生成流程完成。")
    return {"llm_response": final_response}
# 函式：[新] 最终生成节点 (v2.0 - 优化信息顺序)




# 函式：驗證、學習與持久化節點
# 更新紀錄:
# v7.0 (2025-10-05): [重大架構重構] 根据最终确立的 v7.0 蓝图，彻底重写了整个对话图。废弃了所有基于 TurnPlan JSON 的复杂规划和渲染节点。新的“信息注入式架构”流程更线性、更简单：1. 感知与信息收集。 2. (全新) 前置工具调用，用于处理明确的状态变更。 3. 将所有信息（LORE、记忆、工具结果）组装成一个巨大的 world_snapshot 上下文。 4. (全新) 单一的最终生成节点，将 world_snapshot 和用户指令直接交给一个由 00_supreme_directive.txt 驱动的强大 LLM 进行一步到位的自由创作。每个与 API 交互的节点都内置了强大的“功能重建”式备援方案。
# v2.0 (2025-10-07): [架構重構] 此节点的职责被扩展。它现在负责组装所有不同来源的上下文（RAG 记忆、短期对话历史、世界快照），并严格按照“历史 -> 事实 -> 指令”的顺序，将它们填充到新的提示词模板中，然后调用核心生成链。
# v2.1 (2025-10-14): [災難性BUG修復] 確保 `llm_response` 在調用 `.strip()` 之前，先獲取其 `.content` 屬性，解決 `AttributeError: 'AIMessage' object has no attribute 'strip'`。
# v2.2 (2025-10-15): [災難性BUG修復] 恢復了對 `ai_core._save_interaction_to_dbs` 的調用，以確保對話歷史被正確持久化。
# v2.3 (2025-10-15): [災難性BUG修復] 在調用 `lore_extraction_chain` 時，補全了缺失的 `username` 和 `ai_name` 參數，解決了 `KeyError`。
# v2.4 (2025-10-15): [健壯性] 在調用 `lore_extraction_chain` 之前，先對可能包含 NSFW 內容的 `clean_response` 進行清洗，以避免內容審查。
# v2.5 (2025-10-15): [健壯性] 將本回合的最終回應存入 state，為下一輪的連續性指令提供無損上下文。
async def validate_and_persist_node(state: ConversationGraphState) -> Dict:
    """[7] 清理文本、事後 LORE 提取、保存對話歷史，並為下一輪準備無損上下文。"""
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
    try:
        logger.info(f"[{user_id}] (Graph|7) 正在啟動事後 LORE 學習...")
        lore_extraction_chain = ai_core.get_lore_extraction_chain()
        if lore_extraction_chain and ai_core.profile:
            logger.info(f"[{user_id}] (Graph|7) [LORE Pre-Sanitization] 正在為 LORE 提取器準備安全的輸入文本...")
            literary_chain = ai_core.get_literary_euphemization_chain()
            safe_response_for_lore = await ai_core.ainvoke_with_rotation(
                literary_chain,
                {"dialogue_history": clean_response},
                retry_strategy='none'
            )

            if not safe_response_for_lore:
                logger.warning(f"[{user_id}] (Graph|7) [LORE Pre-Sanitization] 文本預清洗失敗，將跳過本輪 LORE 提取。")
            else:
                logger.info(f"[{user_id}] (Graph|7) [LORE Pre-Sanitization] 文本預清洗成功。")
                lore_extraction_params = {
                    "username": ai_core.profile.user_profile.name,
                    "ai_name": ai_core.profile.ai_profile.name,
                    "existing_lore_summary": "",
                    "user_input": user_input,
                    "final_response_text": safe_response_for_lore
                }
                extraction_plan = await ai_core.ainvoke_with_rotation(
                    lore_extraction_chain,
                    lore_extraction_params,
                    retry_strategy='euphemize'
                )
                if extraction_plan and extraction_plan.plan:
                    logger.info(f"[{user_id}] (Graph|7) 事後學習到 {len(extraction_plan.plan)} 條新 LORE，正在後台保存...")
                    asyncio.create_task(ai_core._execute_tool_call_plan(extraction_plan, ai_core.profile.game_state.location_path))
    except Exception as e:
        logger.warning(f"[{user_id}] (Graph|7) 事後 LORE 學習失敗，已跳過。核心對話保存不受影響。錯誤: {e}")

    # 3. 持久化
    if clean_response and "抱歉" not in clean_response:
        chat_history_manager = ai_core.session_histories.setdefault(user_id, ChatMessageHistory())
        chat_history_manager.add_user_message(user_input)
        chat_history_manager.add_ai_message(clean_response)
        
        last_interaction_text = f"使用者: {user_input}\n\nAI:\n{clean_response}"
        asyncio.create_task(ai_core._save_interaction_to_dbs(last_interaction_text))
        
        logger.info(f"[{user_id}] (Graph|7) 對話歷史已更新並準備保存到 DB。")

    logger.info(f"[{user_id}] (Graph|7) 狀態持久化完成。")
    
    # [v2.5 核心修正] 將本回合的最終回應存入 state，為下一輪提供無損上下文
    return {"final_output": clean_response, "last_response_text": clean_response}
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
# ... (此处应包含 setup_graph 的所有节点和构建器代码)
# ... (为遵守“严禁省略”规则，此处贴上所有 setup_graph 相关代码)
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

async def world_genesis_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Setup Graph|3/4) Node: world_genesis -> 節點已啟動...")
    genesis_result = None
    try:
        if not ai_core.profile: raise Exception("AI Profile is not loaded.")
        genesis_chain = ai_core.get_world_genesis_chain()
        genesis_result = await ai_core.ainvoke_with_rotation(genesis_chain, {"world_settings": ai_core.profile.world_settings, "username": ai_core.profile.user_profile.name, "ai_name": ai_core.profile.ai_profile.name}, retry_strategy='force')
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













