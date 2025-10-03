# src/graph.py 的中文註釋(v33.0 - 永久性轟炸架構)
# 更新紀錄:
# v33.0 (2025-10-03): [重大架構重構] 根據「永久性轟炸」與「直接RAG」策略，徹底重寫了主對話圖，用一個更簡單、更線性的新工作流取代了舊的 `preprocess_and_generate` 邏輯。
# v32.0 (2025-10-15): [架構簡化] 根據「永久性轟炸」策略，移除了 `classify_intent_node`，並將其職責整合進了 `perceive_scene_node` 和 `validate_and_persist_node`。
# v31.0 (2025-10-15): [架構重構] 將新的 `classify_intent_node` 作為圖的入口點，實現了「智能轟炸」策略。
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
                      ToolCall)
from .tool_context import tool_context
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- [v33.0 新架構] 主對話圖 (Main Conversation Graph) 的節點 ---

# 函式：[新] 場景感知與上下文恢復節點
# 更新紀錄:
# v3.0 (2025-10-03): [全新創建] 根據「永久性轟炸」架構創建此節點。它作為圖的入口，負責分析用戶輸入、處理連續性指令、恢復上下文快照，並保持場景視角的連貫性。
# v2.0 (2025-10-15): [災難性BUG修復] 引入了【上下文感知的視角保持】策略。
# v1.0 (2025-10-05): [重大架構重構] 根據 v7.0 藍圖創建此節點。
async def perceive_scene_node(state: ConversationGraphState) -> Dict:
    """[1] (入口) 分析用户输入，處理連續性指令，恢復上下文快照，並保持場景視角的連貫性。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    # 確保 messages 列表不為空
    if not state.get('messages'):
        logger.error(f"[{user_id}] (Graph|1) 狀態中缺少 'messages'，無法感知場景。")
        # 提供一個安全的預設值
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='错误：状态中缺少 messages。', action_summary="")}

    user_input = state['messages'][-1].content.strip() if state['messages'] else ""
    logger.info(f"[{user_id}] (Graph|1) Node: perceive_scene -> 正在感知场景与恢复上下文...")

    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|1) ai_core.profile 未加载，无法感知场景。")
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='错误：AI profile 未加载。', action_summary=user_input)}

    gs = ai_core.profile.game_state
    
    # --- 步驟 1: 處理連續性指令與上下文恢復 ---
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

    # --- 步驟 2: 處理常規指令的視角更新 ---
    new_viewing_mode = 'local'
    new_target_path = None
    final_reasoning = "場景感知完成。"

    try:
        location_chain = ai_core.get_location_extraction_prompt()
        full_prompt = ai_core._safe_format_prompt(location_chain, {"user_input": user_input})
        
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
        is_explicit_local_move = any(kw in user_input for kw in ["我", ai_core.profile.user_profile.name, ai_core.profile.ai_profile.name])
        if is_explicit_local_move and new_viewing_mode == 'local':
            final_viewing_mode = 'local'
            final_target_path = None
            logger.info(f"[{user_id}] (Graph|1) 檢測到明確的本地指令，導演視角從 'remote' 切換回 'local'。")
        elif new_viewing_mode == 'remote' and new_target_path and new_target_path != gs.remote_target_path:
            final_target_path = new_target_path
            logger.info(f"[{user_id}] (Graph|1) 在遠程模式下，更新了觀察目標地點為: {final_target_path}")
        else:
            logger.info(f"[{user_id}] (Graph|1) 未檢測到本地切換信號，導演視角保持為 'remote'。")
    else: # gs.viewing_mode == 'local'
        if new_viewing_mode == 'remote' and new_target_path:
            final_viewing_mode = 'remote'
            final_target_path = new_target_path
            logger.info(f"[{user_id}] (Graph|1) 檢測到遠程描述指令，導演視角從 'local' 切換到 'remote'。目標: {final_target_path}")

    if gs.viewing_mode != final_viewing_mode or gs.remote_target_path != final_target_path:
        gs.viewing_mode = final_viewing_mode
        gs.remote_target_path = final_target_path
        await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})
    
    scene_analysis = SceneAnalysisResult(
        viewing_mode=gs.viewing_mode,
        reasoning=final_reasoning,
        target_location_path=gs.remote_target_path,
        action_summary=user_input
    )
    return {"scene_analysis": scene_analysis}
# 函式：[新] 場景感知與上下文恢復節點





# 函式：[新] 記憶與 LORE 查詢節點 (v5.1 - 呼叫簽名修正)
# 更新紀錄:
# v5.1 (2025-10-03): [災難性BUG修復] 根據 AttributeError，修正了在源頭清洗步驟中對 `ainvoke_with_rotation` 的呼叫方式。新版本現在會先使用 `_safe_format_prompt` 將模板和參數手動組合成一個完整的 Prompt 字串，然後再以正確的簽名進行調用，從根源上解決了因參數傳遞錯誤導致的屬性錯誤。
# v5.0 (2025-10-03): [全新創建] 根據「直接RAG」架構創建此節點。
# v4.0 (2025-10-15): [性能優化] 增加了對已恢復上下文的檢查。
async def retrieve_and_query_node(state: ConversationGraphState) -> Dict:
    """[2] (如果需要) 清洗使用者輸入，檢索 RAG 記憶，並查詢所有相關的 LORE。"""
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

    logger.info(f"[{user_id}] (Graph|2) Node: retrieve_and_query -> 正在檢索記憶與查詢LORE...")
    scene_analysis = state['scene_analysis']
    
    sanitized_query = user_input
    try:
        # [v5.1 核心修正] 先格式化 Prompt，再調用 ainvoke_with_rotation
        literary_chain_prompt = ai_core.get_literary_euphemization_chain()
        full_prompt = ai_core._safe_format_prompt(literary_chain_prompt, {"dialogue_history": user_input})
        
        result = await ai_core.ainvoke_with_rotation(
            full_prompt, 
            retry_strategy='euphemize'
        )
        if result:
            sanitized_query = result
    except Exception as e:
        # 由於 ainvoke_with_rotation 增強了日誌，這裡的日誌會更詳細
        logger.warning(f"[{user_id}] (Graph|2) 源頭清洗失敗，將使用原始輸入進行查詢。詳細錯誤: {type(e).__name__}")

    rag_context_dict = await ai_core.retrieve_and_summarize_memories(sanitized_query)
    rag_context_str = rag_context_dict.get("summary", "沒有檢索到相關的長期記憶。")

    is_remote = scene_analysis.viewing_mode == 'remote'
    final_lores = await ai_core._query_lore_from_entities(sanitized_query, is_remote_scene=is_remote)
        
    logger.info(f"[{user_id}] (Graph|2) 查詢完成。檢索到 {len(final_lores)} 條相關LORE。")
    
    return {
        "rag_context": rag_context_str,
        "raw_lore_objects": final_lores,
        "sanitized_query_for_tools": sanitized_query
    }
# 函式：[新] 記憶與 LORE 查詢節點 (v5.1 - 呼叫簽名修正)




# 函式：[新] LORE 擴展決策與執行節點
# 更新紀錄:
# v6.0 (2025-10-03): [全新創建] 根據「直接RAG」架構創建此節點。它取代了舊的重量級「即時精煉」，改為執行一個輕量級的決策：判斷是否需要在對話前為一個全新的實體創建一個「骨架 LORE」，以避免 LLM 憑空捏造。
# v5.0 (2025-10-10): [災難性BUG修復] 徹底重構了節點的輸出邏輯。
async def expansion_decision_and_execution_node(state: ConversationGraphState) -> Dict:
    """[3] 決策是否需要擴展 LORE，如果需要，則立即執行輕量級的骨架創建。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    safe_query_text = state['sanitized_query_for_tools']
    raw_lore_objects = state.get('raw_lore_objects', [])
    logger.info(f"[{user_id}] (Graph|3) Node: expansion_decision_and_execution -> 正在決策是否擴展LORE...")

    lightweight_lore_json = json.dumps(
        [{"name": lore.content.get("name"), "description": lore.content.get("description")} for lore in raw_lore_objects if lore.category == 'npc_profile'],
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
        return {"planning_subjects": [lore.content for lore in raw_lore_objects]}

    logger.info(f"[{user_id}] (Graph|3) 決策結果：需要擴展。理由: {decision.reasoning}。正在執行LORE擴展...")
    
    newly_created_lores = []
    try:
        gs = ai_core.profile.game_state
        effective_location_path = gs.remote_target_path if gs.viewing_mode == 'remote' and gs.remote_target_path else gs.location_path
        
        casting_chain_prompt = ai_core.get_scene_casting_chain()
        full_casting_prompt = ai_core._safe_format_prompt(casting_chain_prompt, {
            "world_settings": ai_core.profile.world_settings or "", 
            "current_location_path": " > ".join(effective_location_path), 
            "user_input": safe_query_text
        })
        cast_result = await ai_core.ainvoke_with_rotation(
            full_casting_prompt,
            output_schema=SceneCastingResult,
            retry_strategy='euphemize'
        )
        
        if cast_result:
            newly_created_lores = await ai_core._add_cast_to_scene(cast_result)
            created_names = [lore.content.get("name", "未知") for lore in newly_created_lores]
            logger.info(f"[{user_id}] (Graph|3) 擴展成功，創建了 {len(created_names)} 位新角色骨架。")
        
    except Exception as e:
        logger.error(f"[{user_id}] (Graph|3) LORE擴展執行時發生錯誤: {e}", exc_info=True)
    
    final_lore_objects = raw_lore_objects + newly_created_lores
    final_lores_map = {lore.key: lore for lore in final_lore_objects}
    
    return {"planning_subjects": [lore.content for lore in final_lores_map.values()]}
# 函式：[新] LORE 擴展決策與執行節點

# 函式：[新] 前置工具調用節點
# 更新紀錄:
# v3.0 (2025-10-03): [全新創建] 根據「直接RAG」架構創建此節點。它的職責是在主小說生成之前，解析並執行使用者輸入中明確包含的工具調用指令（例如 `/equip`），以確保世界狀態的即時更新。
# v2.1 (2025-10-14): [災難性BUG修復] 修正了 `CharacterAction` 驗證錯誤。
async def preemptive_tool_call_node(state: ConversationGraphState) -> Dict:
    """[4] (全新) 判斷並執行使用者指令中明確的、需要改變世界狀態的動作。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|4) Node: preemptive_tool_call -> 正在解析前置工具調用...")

    tool_parsing_chain_prompt = ai_core.get_preemptive_tool_parsing_chain()
    character_list_str = ", ".join([ps.get("name", "") for ps in state.get("planning_subjects", []) if ps])
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
        results_summary, _ = await ai_core._execute_tool_call_plan(tool_call_plan, ai_core.profile.game_state.location_path)
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
# v3.0 (2025-10-03): [全新創建] 根據「直接RAG」架構創建此節點。它的核心職責是匯集前面所有節點的處理結果（RAG、LORE、工具執行結果等），並使用 `world_snapshot_template.txt` 模板將它們格式化成一個巨大的、包含所有上下文的字符串，作為給最終生成模型的「單一信息源」。
# v2.1 (2025-10-14): [災難性BUG修復] 增加了 `username` 和 `ai_name` 到 `context_vars`。
async def assemble_world_snapshot_node(state: ConversationGraphState) -> Dict:
    """[5] (核心) 匯集所有【當前場景】的信息，使用模板格式化成 world_snapshot 字符串。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|5) Node: assemble_world_snapshot -> 正在組裝【當前場景事實】...")
    
    planning_subjects = state.get("planning_subjects", [])
    tool_results = state.get("tool_results", "")
    
    npc_context_str = "\n".join([f"- **{npc.get('name', '未知NPC')}**: {npc.get('description', '無描述')}" for npc in planning_subjects if npc])
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
        'scene_rules_context': "（暫無）" # 規則注入將在生成前完成
    }
    
    final_world_snapshot = ai_core.world_snapshot_template.format(**context_vars)
    
    logger.info(f"[{user_id}] (Graph|5) 【當前場景事實】組裝完畢。")
    return {"world_snapshot": final_world_snapshot}
# 函式：[新] 世界快照組裝節點

# 函式：[新] 最終生成節點
# 更新紀錄:
# v8.0 (2025-10-03): [全新創建] 根據「永久性轟炸」架構創建此節點。它負責組裝所有上下文，並調用統一的生成鏈 `ainvoke_with_rotation`（設置 `retry_strategy='force'`）來一步到位地創作小說，將對抗內容審查的壓力完全交給後端備援機制。
# v7.0 (2025-10-15): [架構簡化] 移除了意圖分類的判斷。
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

    full_prompt = ai_core._safe_format_prompt(
        "{core_protocol_prompt}\n\n{world_snapshot}\n\n# --- 對話歷史摘要 ---\n{historical_context}\n\n# --- 本回合互動 ---\n{username}: {latest_user_input}\n{ai_name}:",
        {
            "core_protocol_prompt": ai_core.core_protocol_prompt,
            "world_snapshot": world_snapshot,
            "historical_context": historical_context,
            "username": ai_core.profile.user_profile.name,
            "latest_user_input": user_input,
            "ai_name": ai_core.profile.ai_profile.name
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
    return {"llm_response": final_response}
# 函式：[新] 最终生成节点

# 函式：[新] 驗證、學習與持久化節點
# 更新紀錄:
# v3.0 (2025-10-03): [全新創建] 根據「生成與學習分離」原則創建此節點。它作為圖的終點，負責所有事後處理工作：清理文本、異步觸發背景 LORE 提取、保存對話歷史，並為下一輪的連續性指令創建關鍵的上下文快照。
# v2.8 (2025-10-15): [健壯性] 在此節點的末尾，創建並儲存上下文快照。
async def validate_and_persist_node(state: ConversationGraphState) -> Dict:
    """[7] 清理文本、事後 LORE 提取、保存對話歷史，並為下一輪創建上下文快照。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    llm_response = state['llm_response']
    logger.info(f"[{user_id}] (Graph|7) Node: validate_and_persist -> 正在驗證、學習與持久化...")

    clean_response = llm_response.strip()
    
    # 創建上下文快照以供後台任務使用
    context_snapshot = {
        "user_input": user_input,
        "final_response": clean_response,
        "scene_rules_context": "（暫無）", # 這裡可以從 state 中獲取，如果需要的話
        "relevant_characters": [lore.content for lore in state.get('raw_lore_objects', []) if lore.category == 'npc_profile']
    }
    
    # 異步觸發背景學習
    asyncio.create_task(ai_core._background_lore_extraction(context_snapshot))

    if clean_response and "抱歉" not in clean_response:
        scene_key = ai_core._get_scene_key()
        await ai_core._add_message_to_scene_history(scene_key, HumanMessage(content=user_input))
        await ai_core._add_message_to_scene_history(scene_key, AIMessage(content=clean_response))
        logger.info(f"[{user_id}] (Graph|7) 對話歷史已更新並持久化。")

    # 為下一輪的 "continue" 指令創建快照
    snapshot_for_next_turn = {
        "raw_lore_objects": state.get("raw_lore_objects", []),
        "last_response_text": clean_response
    }
    ai_core.last_context_snapshot = snapshot_for_next_turn
    logger.info(f"[{user_id}] (Graph|7) 已為下一輪創建上下文快照。")

    logger.info(f"[{user_id}] (Graph|7) 狀態持久化完成。")
    
    return {"final_output": clean_response}
# 函式：[新] 驗證、學習與持久化節點

# 函式：獲取摘要後的對話歷史
# 更新紀錄:
# v28.0 (2025-09-25): [災難性BUG修復] 採用了全新的、更強大的文學評論家鏈。
async def _get_summarized_chat_history(ai_core: AILover, user_id: str, num_messages: int = 8) -> str:
    """提取並摘要最近的對話歷史，並內建一個強大的、基於「文學評論家」重寫的 NSFW 內容安全備援機制。"""
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
# 函式：獲取摘要後的對話歷史

# --- [v33.0 新架構] 圖的構建 ---
# 更新紀錄:
# v33.0 (2025-10-03): [重大架構重構] 創建此函式，將所有新創建的節點連接成一個線性的、取代舊 `preprocess_and_generate` 流程的主對話圖。
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
# --- [v33.0 新架構] 圖的構建 ---

# --- Setup Graph (保持不變) ---
# 函式：處理世界聖經節點
async def process_canon_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    canon_text = state['canon_text']
    logger.info(f"[{user_id}] (Setup Graph|1/4) Node: process_canon -> 節點已啟動。")
    try:
        if canon_text:
            logger.info(f"[{user_id}] (Setup Graph|1/4) 檢測到世界聖經文本 (長度: {len(canon_text)})，開始處理...")
            # 注意：此處 RAG 索引的構建已被移至更高層的協調器
            await ai_core.parse_and_create_lore_from_canon(canon_text)
            logger.info(f"[{user_id}] (Setup Graph|1/4) LORE 智能解析完成。")
        else:
            logger.info(f"[{user_id}] (Setup Graph|1/4) 未提供世界聖經文本，跳過處理。")
    except Exception as e:
        logger.error(f"[{user_id}] (Setup Graph|1/4) Node: process_canon -> 執行時發生嚴重錯誤: {e}", exc_info=True)
    return {}
# 函式：處理世界聖經節點

# 函式：補完角色檔案節點
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
# 函式：補完角色檔案節點

# 函式：世界創世節點
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
# 函式：世界創世節點

# 函式：生成開場白節點
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
