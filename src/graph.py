# src/graph.py 的中文註釋(v34.0 - 违规审查循环)
# 更新紀錄:
# v34.0 (2025-10-13): [灾难性BUG修复] 引入了`violation_check_node`和`should_regenerate`条件分支。在AI生成回应后，会额外进行一次“使用者主权”违规审查，如果发现AI扮演了使用者，则会强制驳回并重新生成，从根本上解决角色扮演混乱的问题。
# v33.0 (2025-10-03): [重大架構重構] 根據「永久性轟炸」與「直接RAG」策略，徹底重寫了主對話圖。
# v32.0 (2025-10-15): [架構簡化] 移除了 `classify_intent_node`。

import sys
import asyncio
import json
import re
from typing import Dict, List, Literal, Optional, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, END

from .ai_core import AILover
from .logger import logger
from .graph_state import ConversationGraphState, SetupGraphState
from . import lore_book, tools
from .schemas import (CharacterProfile, ExpansionDecision, 
                      UserInputAnalysis, SceneAnalysisResult, SceneCastingResult, 
                      WorldGenesisResult, IntentClassificationResult, StyleAnalysisResult,
                      ToolCall, ValidationResult)
from .tool_context import tool_context
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 主對話圖 (Main Conversation Graph) 的節點 ---

async def perceive_scene_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    if not state.get('messages'):
        logger.error(f"[{user_id}] (Graph|1) 狀態中缺少 'messages'，無法感知場景。")
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='错误：状态中缺少 messages。', action_summary="")}
    user_input = state['messages'][-1].content.strip() if state['messages'] else ""
    logger.info(f"[{user_id}] (Graph|1) Node: perceive_scene -> 正在感知场景与恢复上下文...")
    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|1) ai_core.profile 未加载，无法感知场景。")
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='错误：AI profile 未加载。', action_summary=user_input)}
    gs = ai_core.profile.game_state
    logger.info(f"[{user_id}] (Graph|1) 已從持久化存儲中恢復當前狀態：viewing_mode='{gs.viewing_mode}', remote_target='{gs.remote_target_path}'")
    continuation_keywords = ["继续", "繼續", "然後呢", "接下來", "go on", "continue"]
    if any(user_input.lower().startswith(kw) for kw in continuation_keywords):
        logger.info(f"[{user_id}] (Graph|1) 檢測到連續性指令，將繼承上一輪的場景狀態並恢復上下文快照。")
        scene_analysis = SceneAnalysisResult(
            viewing_mode=gs.viewing_mode,
            reasoning="繼承上一輪的場景狀態。",
            target_location_path=gs.remote_target_path,
            action_summary=user_input
        )
        if ai_core.last_context_snapshot:
            logger.info(f"[{user_id}] (Graph|1) [上下文恢復] 成功恢復上一輪的上下文快照。")
            return {
                "scene_analysis": scene_analysis,
                "raw_lore_objects": ai_core.last_context_snapshot.get("raw_lore_objects", []),
                "last_response_text": ai_core.last_context_snapshot.get("last_response_text", None)
            }
        else:
            logger.warning(f"[{user_id}] (Graph|1) [上下文恢復] 未找到上一輪的上下文快照，將重新查詢 LORE。")
            return {"scene_analysis": scene_analysis}
    new_viewing_mode = 'local'
    new_target_path = None
    final_reasoning = "場景感知完成。"
    try:
        location_chain_prompt = ai_core.get_location_extraction_prompt()
        full_prompt = ai_core._safe_format_prompt(location_chain_prompt, {"user_input": user_input})
        from .schemas import SceneLocationExtraction
        location_result = await ai_core.ainvoke_with_rotation(
            full_prompt, 
            output_schema=SceneLocationExtraction,
            retry_strategy='euphemize'
        )
        if location_result and location_result.has_explicit_location and location_result.location_path:
            final_reasoning = f"LLM 感知成功。推斷出的目標地點: {location_result.location_path}"
            logger.info(f"[{user_id}] (Graph|1) {final_reasoning}")
            new_target_path = location_result.location_path
            new_viewing_mode = 'remote'
    except Exception as e:
        final_reasoning = f"地點推斷鏈失敗: {e}，將回退到基本邏輯。"
        logger.warning(f"[{user_id}] (Graph|1) {final_reasoning}")
    final_viewing_mode = gs.viewing_mode
    final_target_path = gs.remote_target_path
    if gs.viewing_mode == 'remote':
        is_explicit_local_move = any(user_input.startswith(kw) for kw in ["去", "前往", "移動到", "旅行到", "我"])
        is_direct_ai_interaction = ai_core.profile.ai_profile.name in user_input
        if is_explicit_local_move or is_direct_ai_interaction:
            final_viewing_mode = 'local'
            final_target_path = None
            logger.info(f"[{user_id}] (Graph|1) [狀態切換] 檢測到明確的本地指令，導演視角從 'remote' 切換回 'local'。")
        elif new_viewing_mode == 'remote' and new_target_path and new_target_path != gs.remote_target_path:
            final_target_path = new_target_path
            logger.info(f"[{user_id}] (Graph|1) [狀態更新] 在遠程模式下，更新了觀察目標地點為: {final_target_path}")
        else:
            logger.info(f"[{user_id}] (Graph|1) [狀態保持] 未檢測到明確的本地切換信號，導演視角保持為 'remote'。")
    else: 
        if new_viewing_mode == 'remote' and new_target_path:
            final_viewing_mode = 'remote'
            final_target_path = new_target_path
            logger.info(f"[{user_id}] (Graph|1) [狀態切換] 檢測到遠程描述指令，導演視角從 'local' 切換到 'remote'。目標: {final_target_path}")
    if gs.viewing_mode != final_viewing_mode or gs.remote_target_path != final_target_path:
        logger.info(f"[{user_id}] (Graph|1) [持久化] 檢測到狀態變更，正在將新的 GameState 寫入資料庫...")
        gs.viewing_mode = final_viewing_mode
        gs.remote_target_path = final_target_path
        await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})
        logger.info(f"[{user_id}] (Graph|1) [持久化] 狀態已成功保存。")
    scene_analysis = SceneAnalysisResult(
        viewing_mode=gs.viewing_mode,
        reasoning=final_reasoning,
        target_location_path=gs.remote_target_path,
        action_summary=user_input
    )
    return {"scene_analysis": scene_analysis, "regeneration_count": 0}

async def retrieve_and_query_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    if state.get('raw_lore_objects') is not None:
        logger.info(f"[{user_id}] (Graph|2) Node: retrieve_and_query -> 檢測到已恢復的 LORE 上下文，將跳過重新查詢。")
        rag_context_dict = await ai_core.retrieve_and_summarize_memories(user_input)
        return {
            "rag_context": rag_context_dict.get("summary", "無相關長期記憶。"),
            "raw_lore_objects": state['raw_lore_objects'],
            "sanitized_query_for_tools": user_input,
            "last_response_text": state.get('last_response_text')
        }
    logger.info(f"[{user_id}] (Graph|2) Node: retrieve_and_query -> 正在執行 RAG 檢索...")
    sanitized_query = user_input
    try:
        literary_chain_prompt = ai_core.get_literary_euphemization_chain()
        full_prompt = ai_core._safe_format_prompt(literary_chain_prompt, {"dialogue_history": user_input})
        result = await ai_core.ainvoke_with_rotation(
            full_prompt, 
            retry_strategy='euphemize'
        )
        if result:
            sanitized_query = result
    except Exception as e:
        logger.warning(f"[{user_id}] (Graph|2) 源頭清洗失敗，將使用原始輸入進行查詢。詳細錯誤: {type(e).__name__}")
    rag_context_dict = await ai_core.retrieve_and_summarize_memories(sanitized_query)
    rag_context_str = rag_context_dict.get("summary", "沒有檢索到相關的長期記憶。")
    all_lores = await lore_book.get_all_lores_for_user(user_id)
    logger.info(f"[{user_id}] (Graph|2) RAG 檢索完成。")
    return {
        "rag_context": rag_context_str,
        "raw_lore_objects": [], 
        "sanitized_query_for_tools": sanitized_query
    }

async def expansion_decision_and_execution_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    safe_query_text = state['sanitized_query_for_tools']
    raw_lore_objects = state.get('raw_lore_objects', [])
    logger.info(f"[{user_id}] (Graph|3) Node: expansion_decision_and_execution -> 正在決策是否擴展LORE...")
    all_lores = await lore_book.get_all_lores_for_user(user_id)
    lightweight_lore_json = json.dumps(
        [{"name": lore.structured_content.get("name"), "description": (lore.narrative_content or "")[:50]} for lore in all_lores if lore.structured_content],
        ensure_ascii=False
    )
    decision_chain_prompt = ai_core.get_expansion_decision_chain()
    full_prompt = ai_core._safe_format_prompt(decision_chain_prompt, {
        "user_input": safe_query_text, 
        "existing_characters_json": lightweight_lore_json
    })
    decision = await ai_core.ainvoke_with_rotation(
        full_prompt, 
        output_schema=ExpansionDecision,
        retry_strategy='euphemize'
    )
    if not decision or not decision.should_expand:
        reason = decision.reasoning if decision else "決策鏈失敗"
        logger.info(f"[{user_id}] (Graph|3) 決策結果：無需擴展。理由: {reason}")
        return {"planning_subjects": [lore.structured_content for lore in all_lores if lore.structured_content]}
    logger.info(f"[{user_id}] (Graph|3) 決策結果：需要擴展。理由: {decision.reasoning}。正在執行LORE擴展...")
    newly_created_lores = []
    try:
        expansion_prompt_template = ai_core.get_lore_expansion_pipeline_prompt()
        existing_lore_names = [lore.structured_content.get("name") for lore in all_lores if lore.structured_content and lore.structured_content.get("name")]
        expansion_prompt = ai_core._safe_format_prompt(expansion_prompt_template, {
            "user_input": safe_query_text,
            "existing_lore_json": json.dumps(existing_lore_names, ensure_ascii=False)
        })
        from .schemas import CanonParsingResult
        expansion_result = await ai_core.ainvoke_with_rotation(
            expansion_prompt,
            output_schema=CanonParsingResult,
            retry_strategy='euphemize'
        )
        if expansion_result:
            newly_created_lores = await ai_core.parse_and_create_lore_from_canon_object(expansion_result)
            created_names = [lore.structured_content.get("name", "未知") for lore in newly_created_lores if lore.structured_content]
            logger.info(f"[{user_id}] (Graph|3) 擴展成功，創建了 {len(created_names)} 個新實體骨架: {created_names}")
    except Exception as e:
        logger.error(f"[{user_id}] (Graph|3) LORE擴展執行時發生錯誤: {e}", exc_info=True)
    final_lore_objects = all_lores + newly_created_lores
    final_lores_map = {lore.key: lore for lore in final_lore_objects}
    return {"planning_subjects": [lore.structured_content for lore in final_lores_map.values() if lore.structured_content]}

async def preemptive_tool_call_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|4) Node: preemptive_tool_call -> 正在解析前置工具調用...")
    tool_parsing_chain_prompt = ai_core.get_preemptive_tool_parsing_chain()
    character_list_str = ", ".join([ps.get("name", "") for ps in state.get("planning_subjects", []) if ps and ps.get("name")])
    full_prompt = ai_core._safe_format_prompt(tool_parsing_chain_prompt, {
        "user_input": user_input, 
        "character_list_str": character_list_str
    })
    from .schemas import ToolCallPlan
    tool_call_plan = await ai_core.ainvoke_with_rotation(
        full_prompt,
        output_schema=ToolCallPlan,
        retry_strategy='euphemize'
    )
    if not tool_call_plan or not tool_call_plan.plan:
        logger.info(f"[{user_id}] (Graph|4) 未解析到明確的工具調用。")
        return {"tool_results": "系統事件：無前置工具被調用。"}
    logger.info(f"[{user_id}] (Graph|4) 解析到 {len(tool_call_plan.plan)} 個工具調用，準備執行...")
    tool_context.set_context(user_id, ai_core)
    results_summary = ""
    try:
        if not ai_core.profile:
             raise Exception("AI Profile尚未初始化，無法獲取當前地點。")
        current_location = ai_core.profile.game_state.location_path
        results_summary, _ = await ai_core._execute_tool_call_plan(tool_call_plan, current_location)
    except Exception as e:
        logger.error(f"[{user_id}] (Graph|4) 前置工具執行時發生錯誤: {e}", exc_info=True)
        results_summary = f"系統事件：工具執行時發生嚴重錯誤: {e}"
    finally:
        tool_context.set_context(None, None)
    logger.info(f"[{user_id}] (Graph|4) 前置工具執行完畢。")
    return {"tool_results": results_summary}
    
async def assemble_world_snapshot_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|5) Node: assemble_world_snapshot -> 正在組裝【純淨版】上下文快照...")
    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|5) 致命錯誤: ai_core.profile 未加載！")
        return {"world_snapshot": "錯誤：AI Profile 丟失。"}
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
        'npc_context': "（上下文已由RAG提供）",
        'relevant_npc_context': "（上下文已由RAG提供）",
        'explicit_character_files_context': "（上下文已由RAG提供）",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': gs.viewing_mode,
        'remote_target_path_str': " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定",
        'scene_rules_context': "（上下文已由RAG提供）"
    }
    if not ai_core.world_snapshot_template:
        logger.error(f"[{user_id}] (Graph|5) 致命錯誤: world_snapshot_template 未加載！")
        return {"world_snapshot": "錯誤：世界快照模板丟失。"}
    final_world_snapshot = ai_core._safe_format_prompt(ai_core.world_snapshot_template, context_vars)
    logger.info(f"[{user_id}] (Graph|5) 【純淨版】上下文快照組裝完畢。")
    return {"world_snapshot": final_world_snapshot}

async def final_generation_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    rag_context = state.get('rag_context', '（無相關長期記憶。）')
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|6) Node: final_generation -> 启动【雙重強化風格注入】最终生成流程...")
    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|6) 致命錯誤: ai_core.profile 未加載！")
        return {"llm_response": "（錯誤：AI Profile 丟失，無法生成回應。）"}
    style_prompt = ai_core.profile.response_style_prompt or "非常具體詳細描述，豐富對話互動"
    top_level_mandate = f"# === 【【【✍️ 絕對風格強制令】】】 ===\n# 你的所有旁白和對話，其語言風格、詳細程度和語氣，都【必須】嚴格遵循以下指令：\n# \"{style_prompt}\""
    recency_reinforcement = f"# === 【🎬 最終指令：風格提醒】 ===\n# 請嚴格遵循您在Prompt頂層收到的【絕對風格強制令】（“{style_prompt}”）來生成您的回應。\n{ai_core.profile.ai_profile.name}:"
    historical_context = await _get_summarized_chat_history(ai_core, user_id)
    plot_anchor = "（無）"
    continuation_keywords = ["继续", "繼續", "然後呢", "接下來", "go on", "continue"]
    if any(user_input.strip().lower().startswith(kw) for kw in continuation_keywords):
        last_response = state.get('last_response_text')
        if last_response:
            plot_anchor = last_response
            logger.info(f"[{user_id}] (Graph|6) 已成功為連續性指令設置【劇情錨點】。")
    regeneration_warning = ""
    if state.get('regeneration_count', 0) > 0:
        regeneration_warning = "【🚨 警告！你上一轮的回应因严重违反‘使用者主权原则’而被驳回！你错误地扮演了使用者。请严格根据以下上下文重新生成你的回应，并确保你的所有描述都只针对AI角色自身。】\n\n"
    final_prompt_template = """{regeneration_warning}{core_protocol_prompt}
{top_level_style_mandate}

# === 情報簡報 (來自 RAG 檢索) ===
{rag_context}
# === 情報結束 ===

# === 劇情錨點 (上一幕的最後場景) ===
{plot_anchor}
# === 錨點結束 ===

# === 最近對話歷史 ===
{historical_context}
# === 歷史結束 ===

# === 本回合互動 ===
{username}: {latest_user_input}
{recency_style_reinforcement}"""
    full_prompt = ai_core._safe_format_prompt(
        final_prompt_template,
        {
            "regeneration_warning": regeneration_warning,
            "core_protocol_prompt": ai_core.core_protocol_prompt,
            "top_level_style_mandate": top_level_mandate,
            "rag_context": rag_context,
            "plot_anchor": plot_anchor,
            "historical_context": historical_context,
            "username": ai_core.profile.user_profile.name,
            "latest_user_input": user_input,
            "recency_style_reinforcement": recency_reinforcement
        }
    )
    logger.info(f"[{user_id}] (Graph|6) [永久性轟炸] 已啟用 'force' 備援策略以最大化成功率。")
    final_response = await ai_core.ainvoke_with_rotation(
        full_prompt,
        retry_strategy='force',
        use_degradation=True
    )
    if not final_response or not final_response.strip():
        logger.critical(f"[{user_id}] (Graph|6) 核心生成链在所有策略之後最終失敗！")
        final_response = "（抱歉，我好像突然断线了，脑海中一片空白...）"
    logger.info(f"[{user_id}] (Graph|6) 最终生成流程完成。")
    return {"llm_response": final_response, "regeneration_count": state.get('regeneration_count', 0) + 1}

async def violation_check_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    llm_response = state['llm_response']
    logger.info(f"[{user_id}] (Graph|7) Node: violation_check -> 正在对AI回应进行“使用者主权”违规审查...")
    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|7) 致命错误: ai_core.profile 未加载！")
        return {"llm_response": llm_response}
    try:
        check_prompt_template = ai_core.get_violation_check_prompt()
        full_prompt = ai_core._safe_format_prompt(
            check_prompt_template,
            {
                "username": ai_core.profile.user_profile.name,
                "llm_response": llm_response
            }
        )
        validation_result = await ai_core.ainvoke_with_rotation(
            full_prompt,
            output_schema=ValidationResult,
            models_to_try_override=[FUNCTIONAL_MODEL]
        )
        if validation_result and validation_result.is_violating:
            logger.warning(f"[{user_id}] (Graph|7) 🚨 审查发现违规！AI回应侵犯了使用者主权。将触发重生成。")
            new_messages = state['messages'] + [SystemMessage(content="VIOLATION_DETECTED")]
            return {"messages": new_messages}
        else:
            logger.info(f"[{user_id}] (Graph|7) ✅ 审查通过，AI回应合规。")
            new_messages = [m for m in state['messages'] if m.content != "VIOLATION_DETECTED"]
            return {"messages": new_messages, "llm_response": llm_response}
    except Exception as e:
        logger.error(f"[{user_id}] (Graph|7) 🔥 违规审查节点执行时发生错误: {e}", exc_info=True)
        return {"llm_response": llm_response}

async def validate_and_persist_node(state: ConversationGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    # [v34.1 核心修正] 確保 user_input 來源正確
    user_input_message = next((m for m in reversed(state['messages']) if isinstance(m, HumanMessage)), None)
    if not user_input_message:
        logger.error(f"[{user_id}] (Graph|8) 致命错误: 在消息列表中找不到 HumanMessage，无法持久化。")
        # 即使出錯，也要返回 final_output 以免前端卡住
        return {"final_output": state.get('llm_response', "錯誤：找不到用戶輸入。")}
    
    user_input = user_input_message.content
    llm_response = state['llm_response']
    logger.info(f"[{user_id}] (Graph|8) Node: validate_and_persist -> 正在驗證、學習與持久化...")
    clean_response = llm_response.strip()
    snapshot_for_analysis = {
        "user_input": user_input,
        "final_response": clean_response,
    }
    asyncio.create_task(ai_core._background_lore_extraction(snapshot_for_analysis))
    await ai_core._save_interaction_to_dbs(f"使用者: {user_input}\n\nAI:\n{clean_response}")
    logger.info(f"[{user_id}] (Graph|8) 對話歷史已更新並進行雙重持久化。")
    snapshot_for_next_turn = {
        "raw_lore_objects": state.get("raw_lore_objects", []),
        "last_response_text": clean_response
    }
    ai_core.last_context_snapshot = snapshot_for_next_turn
    logger.info(f"[{user_id}] (Graph|8) 已為下一輪創建上下文快照。")
    logger.info(f"[{user_id}] (Graph|8) 狀態持久化完成。")
    return {"final_output": clean_response}
    
async def _get_summarized_chat_history(ai_core: AILover, user_id: str, num_messages: int = 8) -> str:
    if not ai_core.profile: return "（沒有最近的對話歷史）"
    scene_key = ai_core._get_scene_key()
    chat_history_manager = ai_core.scene_histories.get(scene_key, ChatMessageHistory())
    if not chat_history_manager.messages:
        return "（沒有最近的對話歷史）"
    recent_messages = chat_history_manager.messages[-num_messages:]
    if not recent_messages:
        return "（沒有最近的對話歷史）"
    raw_history_text = "\n".join([f"{'使用者' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in recent_messages])
    try:
        literary_chain_prompt = ai_core.get_literary_euphemization_chain()
        summary = await ai_core.ainvoke_with_rotation(literary_chain_prompt, {"dialogue_history": raw_history_text}, retry_strategy='euphemize')
        if not summary or not summary.strip():
            raise Exception("Summarization returned empty content.")
        return f"【最近對話摘要】:\n{summary}"
    except Exception as e:
        logger.error(f"[{user_id}] (History Summarizer) 生成摘要時發生錯誤: {e}。返回中性提示。")
        return "（歷史對話摘要因错误而生成失败，部分上下文可能缺失。）"

def should_regenerate(state: ConversationGraphState) -> Literal["final_generation", "validate_and_persist"]:
    MAX_REGENERATIONS = 2
    user_id = state['user_id']
    if state['messages'] and state['messages'][-1].content == "VIOLATION_DETECTED":
        count = state.get('regeneration_count', 0)
        if count < MAX_REGENERATIONS:
            logger.warning(f"[{user_id}] [Graph Control] 触发重生成 (尝试 {count + 1}/{MAX_REGENERATIONS})。")
            return "final_generation"
        else:
            logger.error(f"[{user_id}] [Graph Control] 🔥 已达到最大重生成次数 ({MAX_REGENERATIONS})！将强制通过违规回应以避免死循环。")
            return "validate_and_persist"
    return "validate_and_persist"

def create_main_response_graph() -> StateGraph:
    """创建并连接所有节点，构建包含违规审查循环的最终对话图。"""
    graph = StateGraph(ConversationGraphState)
    
    graph.add_node("perceive_scene", perceive_scene_node)
    graph.add_node("retrieve_and_query", retrieve_and_query_node)
    graph.add_node("expansion_decision_and_execution", expansion_decision_and_execution_node)
    graph.add_node("preemptive_tool_call", preemptive_tool_call_node)
    graph.add_node("assemble_world_snapshot", assemble_world_snapshot_node)
    graph.add_node("final_generation", final_generation_node)
    graph.add_node("violation_check", violation_check_node)
    graph.add_node("validate_and_persist", validate_and_persist_node)
    
    graph.set_entry_point("perceive_scene")
    
    graph.add_edge("perceive_scene", "retrieve_and_query")
    graph.add_edge("retrieve_and_query", "expansion_decision_and_execution")
    graph.add_edge("expansion_decision_and_execution", "preemptive_tool_call")
    graph.add_edge("preemptive_tool_call", "assemble_world_snapshot")
    graph.add_edge("assemble_world_snapshot", "final_generation")
    
    graph.add_edge("final_generation", "violation_check")
    graph.add_conditional_edges(
        "violation_check",
        should_regenerate,
        {
            "final_generation": "final_generation",
            "validate_and_persist": "validate_and_persist"
        }
    )
    
    graph.add_edge("validate_and_persist", END)
    
    return graph.compile()

# --- Setup Graph (保持不變) ---
async def process_canon_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    canon_text = state['canon_text']
    logger.info(f"[{user_id}] (Setup Graph|1/4) Node: process_canon -> 節點已啟動。")
    try:
        if canon_text:
            logger.info(f"[{user_id}] (Setup Graph|1/4) 檢測到世界聖經文本 (長度: {len(canon_text)})，開始處理...")
            await ai_core.parse_and_create_lore_from_canon(canon_text)
            logger.info(f"[{user_id}] (Setup Graph|1/4) LORE 智能解析完成。")
        else:
            logger.info(f"[{user_id}] (Setup Graph|1/4) 未提供世界聖經文本，跳過處理。")
    except Exception as e:
        logger.error(f"[{user_id}] (Setup Graph|1/4) Node: process_canon -> 執行時發生嚴重錯誤: {e}", exc_info=True)
    return {}

async def complete_profiles_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Setup Graph|2/4) Node: complete_profiles_node -> 節點已啟動，準備補完角色檔案...")
    try:
        await ai_core.complete_character_profiles()
        logger.info(f"[{user_id}] (Setup Graph|2/4) Node: complete_profiles_node -> 節點執行成功。")
    except Exception as e:
        logger.error(f"[{user_id}] (Setup Graph|2/4) Node: complete_profiles_node -> 執行時發生嚴重錯誤: {e}", exc_info=True)
    return {}

async def world_genesis_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    canon_text = state.get('canon_text')
    logger.info(f"[{user_id}] (Setup Graph|3/4) Node: world_genesis -> 節點已啟動...")
    genesis_result = None
    try:
        await ai_core.generate_world_genesis(canon_text=canon_text)
        logger.info(f"[{user_id}] (Setup Graph|3/4) Node: world_genesis -> 節點執行成功。")
    except Exception as e:
        logger.error(f"[{user_id}] (Setup Graph|3/4) Node: world_genesis -> 執行時發生嚴重錯誤: {e}", exc_info=True)
    return {"genesis_result": genesis_result}

async def generate_opening_scene_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    canon_text = state.get('canon_text')
    opening_scene = ""
    logger.info(f"[{user_id}] (Setup Graph|4/4) Node: generate_opening_scene -> 節點已啟動...")
    try:
        opening_scene = await ai_core.generate_opening_scene(canon_text=canon_text)
        if not opening_scene or not opening_scene.strip():
            opening_scene = (f"在一片柔和的光芒中...")
        logger.info(f"[{user_id}] (Setup Graph|4/4) Node: generate_opening_scene -> 節點執行成功。")
    except Exception as e:
        logger.error(f"[{user_id}] (Setup Graph|4/4) Node: generate_opening_scene -> 執行時發生嚴重錯誤: {e}", exc_info=True)
        opening_scene = (f"在一片柔和的光芒中...")
    return {"opening_scene": opening_scene}

def create_setup_graph() -> StateGraph:
    graph = StateGraph(SetupGraphState)
    graph.add_node("complete_profiles", complete_profiles_node)
    graph.add_node("generate_opening_scene", generate_opening_scene_node)
    graph.set_entry_point("complete_profiles")
    graph.add_edge("complete_profiles", "generate_opening_scene")
    graph.add_edge("generate_opening_scene", END)
    return graph.compile()
