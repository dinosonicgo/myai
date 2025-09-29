# src/graph.py 的中文註釋(v32.0 - 永久性轟炸架構)
# 更新紀錄:
# v32.0 (2025-10-15): [架構簡化] 根據「永久性轟炸」策略，移除了 `classify_intent_node`，並將其職責（處理連續性指令、恢復上下文快照）整合進了 `perceive_scene_node` 和 `validate_and_persist_node`，使流程更線性、更高效。
# v31.0 (2025-10-15): [架構重構] 將新的 `classify_intent_node` 作為圖的入口點，實現了「智能轟炸」策略。
# v30.0 (2025-10-05): [重大架構重構] 根据最终确立的 v7.0 蓝图，彻底重写了整个对话图。
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

# --- [v32.0 新架构] 主對話圖 (Main Conversation Graph) 的節點 ---

# graph.py 的 perceive_scene_node 函式 (v3.0 - 讀取持久化快照)
# 更新紀錄:
# v3.0 (2025-11-24): [健壯性強化] 根據上下文快照持久化策略，增加了在處理「繼續」指令時，如果記憶體快照丟失，則嘗試從資料庫中讀取持久化快照的備援邏輯。此修改確保了即使在程式重啟後，連續性指令依然能夠無縫銜接劇情。
# v2.0 (2025-10-15): [災難性BUG修復] 引入了【上下文感知的視角保持】策略。
# v1.0 (2025-10-05): [重大架構重構] 根據 v7.0 藍圖創建此節點。
async def perceive_scene_node(state: ConversationGraphState) -> Dict:
    """[1] (入口) 分析用户输入，處理連續性指令，恢復上下文快照，並保持場景視角的連貫性。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content.strip()
    logger.info(f"[{user_id}] (Graph|1) Node: perceive_scene -> 正在感知场景与恢复上下文...")

    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|1) ai_core.profile 未加载，无法感知场景。")
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='错误：AI profile 未加载。', action_summary=user_input)}

    gs = ai_core.profile.game_state
    
    # --- 步驟 1: 處理連續性指令與上下文恢復 ---
    continuation_keywords = ["继续", "繼續", "然後呢", "接下來", "go on", "continue"]
    if any(user_input.lower().startswith(kw) for kw in continuation_keywords):
        logger.info(f"[{user_id}] (Graph|1) 檢測到連續性指令，將繼承上一輪的場景狀態並嘗試恢復上下文快照。")
        
        scene_analysis = SceneAnalysisResult(
            viewing_mode=gs.viewing_mode,
            reasoning="繼承上一輪的場景狀態。",
            target_location_path=gs.remote_target_path,
            action_summary=user_input
        )
        
        # [v3.0 核心修正] 優先從記憶體恢復，失敗則從資料庫恢復
        if ai_core.last_context_snapshot:
            logger.info(f"[{user_id}] (Graph|1) [上下文恢復] 成功從【記憶體】恢復上一輪的上下文快照。")
            return { "scene_analysis": scene_analysis, **ai_core.last_context_snapshot }
        else:
            logger.warning(f"[{user_id}] (Graph|1) [上下文恢復] 未在記憶體中找到快照，正在嘗試從【資料庫】讀取...")
            async with AsyncSessionLocal() as session:
                user_data = await session.get(UserData, user_id)
                if user_data and user_data.context_snapshot_json:
                    logger.info(f"[{user_id}] (Graph|1) [上下文恢復] 成功從【資料庫】恢復持久化的上下文快照。")
                    # 將從資料庫讀取的快照加載到當前 AI 實例的記憶體中，以供後續節點使用
                    ai_core.last_context_snapshot = user_data.context_snapshot_json
                    return { "scene_analysis": scene_analysis, **user_data.context_snapshot_json }
                else:
                    logger.error(f"[{user_id}] (Graph|1) [上下文恢復] 災難性失敗：記憶體和資料庫中均未找到任何上下文快照，無法完美銜接劇情。")
                    return {"scene_analysis": scene_analysis}

    # --- 步驟 2: 處理常規指令的視角更新 ---
    # (此部分邏輯不變，保持原樣)
    new_viewing_mode = 'local'
    new_target_path = None
    try:
        location_chain = ai_core.get_contextual_location_chain()
        location_result = await ai_core.ainvoke_with_rotation(
            location_chain, 
            {"user_input": user_input, "world_settings": ai_core.profile.world_settings or "未设定", "scene_context_json": "[]"},
            retry_strategy='euphemize'
        )
        if location_result and location_result.location_path:
            new_target_path = location_result.location_path
            new_viewing_mode = 'remote'
    except Exception as e:
        logger.warning(f"[{user_id}] (Graph|1) 地點推斷鏈失敗: {e}，將回退到基本邏輯。")

    final_viewing_mode = gs.viewing_mode
    final_target_path = gs.remote_target_path
    if gs.viewing_mode == 'remote':
        is_explicit_local_move = any(user_input.startswith(kw) for kw in ["去", "前往", "移動到", "旅行到"])
        is_direct_ai_interaction = ai_core.profile.ai_profile.name in user_input
        if is_explicit_local_move or is_direct_ai_interaction:
            final_viewing_mode = 'local'
            final_target_path = None
        elif new_viewing_mode == 'remote' and new_target_path and new_target_path != gs.remote_target_path:
            final_target_path = new_target_path
    else:
        if new_viewing_mode == 'remote' and new_target_path:
            final_viewing_mode = 'remote'
            final_target_path = new_target_path
    
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
    # 清空上一輪的記憶體快照，為本輪生成新的快照做準備
    ai_core.last_context_snapshot = None
    return {"scene_analysis": scene_analysis}
# 函式：[新] 场景感知与上下文恢复节点




# 函式：[新] 記憶與 LORE 查詢節點
# 更新紀錄:
# v4.0 (2025-10-15): [性能優化] 增加了對已恢復上下文的檢查。如果上下文已由 `perceive_scene_node` 注入，則跳過所有耗時的查詢操作。
async def retrieve_and_query_node(state: ConversationGraphState) -> Dict:
    """[2] (如果需要) 清洗使用者輸入，檢索 RAG 記憶，並查詢所有相關的 LORE。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    
    # [v4.0 核心修正] 檢查上下文是否已被恢復
    if state.get('raw_lore_objects') is not None:
        logger.info(f"[{user_id}] (Graph|2) Node: retrieve_and_query -> 檢測到已恢復的 LORE 上下文，將跳過重新查詢。")
        rag_context_str = await ai_core.retrieve_and_summarize_memories(user_input)
        return {
            "rag_context": rag_context_str,
            "raw_lore_objects": state['raw_lore_objects'],
            "sanitized_query_for_tools": user_input,
            "last_response_text": state.get('last_response_text') # 保持傳遞
        }

    logger.info(f"[{user_id}] (Graph|2) Node: retrieve_and_query -> 正在檢索記憶與查詢LORE...")
    scene_analysis = state['scene_analysis']
    
    sanitized_query = user_input
    try:
        literary_chain = ai_core.get_literary_euphemization_chain()
        result = await ai_core.ainvoke_with_rotation(literary_chain, {"dialogue_history": user_input}, retry_strategy='euphemize')
        if result:
            sanitized_query = result.content if hasattr(result, 'content') else str(result)
    except Exception:
        logger.warning(f"[{user_id}] (Graph|2) 源頭清洗失敗，將使用原始輸入進行查詢。")

    rag_context_str = "沒有檢索到相關的長期記憶。"
    if len(user_input) > 10 or any(kw in user_input for kw in ["誰", "什麼", "回憶", "記得"]):
        logger.info(f"[{user_id}] (Graph|2) 輸入較複雜，執行 RAG 檢索...")
        rag_context_str = await ai_core.retrieve_and_summarize_memories(sanitized_query)
    else:
        logger.info(f"[{user_id}] (Graph|2) 輸入為簡單指令，跳過 RAG 檢索以節省配額。")

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
# v5.0 (2025-10-10): [災難性BUG修復] 徹底重構了節點的輸出邏輯，以中斷遞迴查詢風暴。
# v4.0 (2025-10-15): [架構重構] 實現了「原子化角色創建」。
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
    
    newly_created_lores: List[Lore] = []
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
    
    final_lore_objects = raw_lore_objects + newly_created_lores
    final_lores_map = {lore.key: lore for lore in final_lore_objects}
    
    return {"planning_subjects": [lore.content for lore in final_lores_map.values()]}
# 函式：[新] LORE 擴展決策與執行節點

# 函式：[新] 前置工具調用節點
# 更新紀錄:
# v2.1 (2025-10-14): [災難性BUG修復] 修正了 `CharacterAction` 驗證錯誤。
# v2.0 (2025-10-07): [架構重構] 此节点的职责被扩展。
async def preemptive_tool_call_node(state: ConversationGraphState) -> Dict:
    """[4] (全新) 判斷並執行使用者指令中明確的、需要改變世界狀態的動作。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|4) Node: preemptive_tool_call -> 正在解析前置工具調用...")

    tool_parsing_chain = ai_core.get_preemptive_tool_parsing_chain()
    tool_call_plan = await ai_core.ainvoke_with_rotation(
        tool_parsing_chain,
        {"user_input": user_input, "character_list_str": ", ".join([ps.get("name", "") for ps in state.get("planning_subjects", [])])},
        retry_strategy='euphemize'
    )
    
    if not tool_call_plan or not tool_call_plan.plan:
        logger.info(f"[{user_id}] (Graph|4) 未解析到明確的工具調用。")
        return {"tool_results": "系統事件：無前置工具被調用。"}

    logger.info(f"[{user_id}] (Graph|4) 解析到 {len(tool_call_plan.plan)} 個工具調用，準備執行...")
    
    tool_context.set_context(user_id, ai_core)
    try:
        from .schemas import TurnPlan, CharacterAction
        simple_turn_plan = TurnPlan(
            character_actions=[
                CharacterAction(
                    character_name="system", 
                    reasoning="preemptive tool execution", 
                    tool_call=call,
                    action_description=f"執行工具 {call.tool_name}."
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

# 函式：[新] 世界快照組裝節點
# 更新紀錄:
# v2.1 (2025-10-14): [災難性BUG修復] 增加了 `username` 和 `ai_name` 到 `context_vars`。
# v2.0 (2025-10-07): [架構重構] 此节点的职责被简化。
async def assemble_world_snapshot_node(state: ConversationGraphState) -> Dict:
    """[5] (核心) 匯集所有【當前場景】的信息，使用模板格式化成 world_snapshot 字符串。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|5) Node: assemble_world_snapshot -> 正在組裝【當前場景事實】...")
    
    planning_subjects = state.get("planning_subjects", [])
    tool_results = state.get("tool_results", "")
    
    npc_context_str = "\n".join([f"- **{npc.get('name', '未知NPC')}**: {npc.get('description', '無描述')}" for npc in planning_subjects])
    if not npc_context_str: npc_context_str = "當前場景沒有已知的特定角色。"

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
    
    logger.info(f"[{user_id}] (Graph|5) 【當前場景事實】組裝完畢。")
    return {"world_snapshot": final_world_snapshot}
# 函式：[新] 世界快照組裝節點

# 函式：[新] 最终生成节点
# 更新紀錄:
# v7.0 (2025-10-15): [架構簡化] 移除了意圖分類的判斷，實現永久性轟炸。
# v6.0 (2025-10-15): [健壯性] 實現了「智能轟炸」策略。
async def final_generation_node(state: ConversationGraphState) -> Dict:
    """[6] (全新) 组装所有上下文，并调用统一生成链来一步到位地创作小说。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    world_snapshot = state['world_snapshot']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|6) Node: final_generation -> 启动最终生成流程...")

    historical_context = await _get_summarized_chat_history(ai_core, user_id)
    
    plot_anchor = "（無）"
    continuation_keywords = ["继续", "繼續", "然後呢", "接下來", "go on", "continue"]
    if any(user_input.strip().lower().startswith(kw) for kw in continuation_keywords):
        last_response = state.get('last_response_text')
        if last_response:
            plot_anchor = last_response
            logger.info(f"[{user_id}] (Graph|6) 已成功為連續性指令設置【劇情錨點】。")

    prompt_template_runnable = ai_core.get_unified_generation_chain()
    
    final_retry_strategy: Literal['force', 'euphemize', 'none'] = 'force'
    logger.info(f"[{user_id}] (Graph|6) [永久性轟炸] 已啟用 'force' 備援策略以最大化成功率。")
    
    final_response_raw = await ai_core.ainvoke_with_rotation(
        prompt_template_runnable,
        {
            "core_protocol_prompt": ai_core.core_protocol_prompt,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "plot_anchor": plot_anchor,
            "historical_context": historical_context,
            "world_snapshot": world_snapshot,
            "latest_user_input": user_input,
        },
        retry_strategy=final_retry_strategy,
        use_degradation=True
    )

    final_response = final_response_raw.content if hasattr(final_response_raw, 'content') else str(final_response_raw)

    if not final_response or not final_response.strip():
        logger.critical(f"[{user_id}] (Graph|6) 核心生成链在所有策略（包括 '{final_retry_strategy}'）之后最终失败！")
        final_response = "（抱歉，我好像突然断线了，脑海中一片空白...）"
        
    logger.info(f"[{user_id}] (Graph|6) 最终生成流程完成。")
    return {"llm_response": final_response}
# 函式：[新] 最终生成节点

# graph.py 的 validate_and_persist_node 函式 (v3.0 - 儲存持久化快照)
# 更新紀錄:
# v3.0 (2025-11-24): [健壯性強化] 根據上下文快照持久化策略，增加了在對話結束時，將新生成的上下文快照異步寫入後端資料庫的核心邏輯。此修改確保了「繼續」指令的上下文即使在程式重啟後也能被成功恢復，極大地提升了系統的穩定性和用戶體驗。
# v2.8 (2025-10-15): [健壯性] 在此節點的末尾，創建並儲存上下文快照。
async def validate_and_persist_node(state: ConversationGraphState) -> Dict:
    """[7] 清理文本、事後 LORE 提取、保存對話歷史，並創建和持久化上下文快照。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    llm_response_raw = state['llm_response']
    logger.info(f"[{user_id}] (Graph|7) Node: validate_and_persist -> 正在驗證、學習與持久化...")

    clean_response = str(llm_response_raw).strip()
    
    # 創建上下文快照
    context_snapshot = {
        "raw_lore_objects": state.get("raw_lore_objects", []),
        "last_response_text": clean_response
    }
    ai_core.last_context_snapshot = context_snapshot
    logger.info(f"[{user_id}] (Graph|7) 已為下一輪在【記憶體】中創建上下文快照。")

    # [v3.0 核心修正] 異步任務：將所有需要持久化的數據一次性寫入資料庫
    async def persist_all_data():
        try:
            # 任務 A: 儲存對話歷史到長期記憶
            if clean_response and "抱歉" not in clean_response:
                last_interaction_text = f"使用者: {user_input}\n\nAI:\n{clean_response}"
                await ai_core._save_interaction_to_dbs(last_interaction_text)
                logger.info(f"[{user_id}] (Graph|7|Async) 對話歷史已成功保存到 DB。")

            # 任務 B: 儲存上下文快照到 UserData 表
            from sqlalchemy import update
            async with AsyncSessionLocal() as session:
                stmt = update(UserData).where(UserData.user_id == user_id).values(context_snapshot_json=context_snapshot)
                await session.execute(stmt)
                await session.commit()
                logger.info(f"[{user_id}] (Graph|7|Async) 上下文快照已成功【持久化】到資料庫。")
                
            # 任務 C: 啟動背景 LORE 提取 (它內部會處理自己的異步邏輯)
            await ai_core._background_lore_extraction(user_input, clean_response)

        except Exception as e:
            logger.error(f"[{user_id}] (Graph|7|Async) 在後台持久化數據時發生錯誤: {e}", exc_info=True)

    asyncio.create_task(persist_all_data())
    
    logger.info(f"[{user_id}] (Graph|7) 狀態持久化與背景學習任務已全部提交。")
    
    return {"final_output": clean_response}
# 函式：驗證、學習與持久化節點

# 函式：獲取摘要後的對話歷史
# 更新紀錄:
# v28.0 (2025-09-25): [災難性BUG修復] 採用了全新的、更強大的文學評論家鏈。
async def _get_summarized_chat_history(ai_core: AILover, user_id: str, num_messages: int = 8) -> str:
    """提取並摘要最近的對話歷史，並內建一個強大的、基於「文學評論家」重寫的 NSFW 內容安全備援機制。"""
    if not ai_core.profile: return "（沒有最近的對話歷史）"
    chat_history_manager = ai_core.session_histories.get(user_id, ChatMessageHistory())
    if not chat_history_manager.messages:
        return "（沒有最近的對話歷史）"
        
    recent_messages = chat_history_manager.messages[-num_messages:]
    if not recent_messages:
        return "（沒有最近的對話歷史）"

    raw_history_text = "\n".join([f"{'使用者' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in recent_messages])

    try:
        literary_chain = ai_core.get_literary_euphemization_chain()
        summary = await ai_core.ainvoke_with_rotation(literary_chain, {"dialogue_history": raw_history_text}, retry_strategy='euphemize')

        if not summary or not summary.strip():
            raise Exception("Summarization returned empty content.")
            
        return f"【最近對話摘要】:\n{summary}"

    except Exception as e:
        logger.error(f"[{user_id}] (History Summarizer) 生成摘要時發生錯誤: {e}。返回中性提示。")
        return "（歷史對話摘要因错误而生成失败，部分上下文可能缺失。）"
# 函式：獲取摘要後的對話歷史

# --- [v32.0 新架构] 图的构建 ---
# 更新紀錄:
# v32.0 (2025-10-15): [架構簡化] 移除了 `classify_intent_node`，恢復 `perceive_scene_node` 為入口點。
def create_main_response_graph() -> StateGraph:
    """创建并连接所有节点，构建最终的对话图。"""
    graph = StateGraph(ConversationGraphState)
    
    graph.add_node("perceive_scene", perceive_scene_node)
    graph.add_node("retrieve_and_query", retrieve_and_query_node)
    graph.add_node("expansion_decision_and_execution", expansion_decision_and_execution_node)
    graph.add_node("preemptive_tool_call", preemptive_tool_call_node)
    graph.add_node("assemble_world_snapshot", assemble_world_snapshot_node)
    graph.add_node("final_generation", final_generation_node)
    graph.add_node("validate_and_persist", validate_and_persist_node)
    
    graph.set_entry_point("perceive_scene")
    
    graph.add_edge("perceive_scene", "retrieve_and_query")
    graph.add_edge("retrieve_and_query", "expansion_decision_and_execution")
    graph.add_edge("expansion_decision_and_execution", "preemptive_tool_call")
    graph.add_edge("preemptive_tool_call", "assemble_world_snapshot")
    graph.add_edge("assemble_world_snapshot", "final_generation")
    graph.add_edge("final_generation", "validate_and_persist")
    graph.add_edge("validate_and_persist", END)
    
    return graph.compile()
# --- [v32.0 新架构] 图的构建 ---

# --- Setup Graph (保持不变) ---
# 函式：處理世界聖經節點
async def process_canon_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    canon_text = state['canon_text']
    logger.info(f"[{user_id}] (Setup Graph|1/4) Node: process_canon -> 節點已啟動。")
    try:
        if canon_text:
            logger.info(f"[{user_id}] (Setup Graph|1/4) 檢測到世界聖經文本 (長度: {len(canon_text)})，開始處理...")
            await ai_core.add_canon_to_vector_store(canon_text)
            logger.info(f"[{user_id}] (Setup Graph|1/4) 步驟 A: 向量化儲存完成。")
            await ai_core.parse_and_create_lore_from_canon(None, canon_text, is_setup_flow=True)
            logger.info(f"[{user_id}] (Setup Graph|1/4) 步驟 B: LORE 智能解析完成。")
        else:
            logger.info(f"[{user_id}] (Setup Graph|1/4) 未提供世界聖經文本，跳過處理。")
        logger.info(f"[{user_id}] (Setup Graph|1/4) Node: process_canon -> 節點執行成功。")
    except Exception as e:
        logger.error(f"[{user_id}] (Setup Graph|1/4) Node: process_canon -> 執行時發生嚴重錯誤: {e}", exc_info=True)
    finally:
        await asyncio.sleep(5.0)
    return {}
# 函式：處理世界聖經節點

# 函式：補完角色檔案節點
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
                tasks_to_clean = {}
                if (desc := safe_profile_data.get('description', '')): tasks_to_clean['description'] = literary_chain.ainvoke({"dialogue_history": desc})
                if (appr := safe_profile_data.get('appearance', '')): tasks_to_clean['appearance'] = literary_chain.ainvoke({"dialogue_history": appr})
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
                original_data.update({k: v for k, v in original_profile.model_dump().items() if v and k in ['description', 'appearance', 'name']})
                return CharacterProfile.model_validate(original_data)
            except Exception as e:
                logger.error(f"[{user_id}] 為角色 '{original_profile.name}' 進行安全補完時發生錯誤: {e}", exc_info=True)
                return original_profile
        final_user_profile, final_ai_profile = await asyncio.gather(_safe_complete_profile(ai_core.profile.user_profile), _safe_complete_profile(ai_core.profile.ai_profile))
        await ai_core.update_and_persist_profile({'user_profile': final_user_profile.model_dump(), 'ai_profile': final_ai_profile.model_dump()})
        logger.info(f"[{user_id}] (Setup Graph|2/4) Node: complete_profiles_node -> 節點執行成功。")
    except Exception as e:
        logger.error(f"[{user_id}] (Setup Graph|2/4) Node: complete_profiles_node -> 執行時發生嚴重錯誤: {e}", exc_info=True)
    finally:
        await asyncio.sleep(5.0)
    return {}
# 函式：補完角色檔案節點

# 函式：世界創世節點
async def world_genesis_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Setup Graph|3/4) Node: world_genesis -> 節點已啟動...")
    genesis_result = None
    try:
        if not ai_core.profile: raise Exception("AI Profile is not loaded.")
        genesis_chain = ai_core.get_world_genesis_chain()
        genesis_params = {"world_settings": ai_core.profile.world_settings or "未設定", "username": ai_core.profile.user_profile.name, "ai_name": ai_core.profile.ai_profile.name}
        genesis_result = await ai_core.ainvoke_with_rotation(genesis_chain, genesis_params, retry_strategy='force')
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
        await asyncio.sleep(5.0)
    return {"genesis_result": genesis_result}
# 函式：世界創世節點

# 函式：生成開場白節點
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
# 函式：生成開場白節點

# 函式：創建設定圖
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
# 函式：創建設定圖


