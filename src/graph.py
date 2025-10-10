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

# 函式：[新] 場景感知與上下文恢復節點 (v3.1 - 遠程優先狀態保持)
# 更新紀錄:
# v3.1 (2025-10-03): [災難性BUG修復] 根據場景誤判問題，徹底重構了此節點的狀態管理邏輯。新版本引入了「遠程優先」的狀態保持策略：當視角已處於遠程模式時，除非使用者發出非常明確的返回本地的指令（如直接與AI互動或發出移動命令），否則視角將被強制保持在遠程。同時，在節點的末尾增加了狀態持久化邏輯，確保任何視角的變更都會被立即寫入資料庫，解決了因狀態丟失導致的場景錯亂問題。
# v3.0 (2025-10-03): [全新創建] 根據「永久性轟炸」架構創建此節點。
# v2.0 (2025-10-15): [災難性BUG修復] 引入了【上下文感知的視角保持】策略。
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

    # [v3.1 核心修正] 無條件從資料庫恢復的 GameState 開始
    gs = ai_core.profile.game_state
    logger.info(f"[{user_id}] (Graph|1) 已從持久化存儲中恢復當前狀態：viewing_mode='{gs.viewing_mode}', remote_target='{gs.remote_target_path}'")
    
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

    # --- 步驟 2: 使用 LLM 智能推斷使用者是否意圖觀察遠程 ---
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

    # --- 步驟 3: [v3.1 核心修正] 應用「遠程優先」的狀態保持邏輯 ---
    final_viewing_mode = gs.viewing_mode
    final_target_path = gs.remote_target_path

    if gs.viewing_mode == 'remote':
        # 只有在非常明確的情況下才切換回 local
        is_explicit_local_move = any(user_input.startswith(kw) for kw in ["去", "前往", "移動到", "旅行到", "我"])
        is_direct_ai_interaction = ai_core.profile.ai_profile.name in user_input
        
        if is_explicit_local_move or is_direct_ai_interaction:
            final_viewing_mode = 'local'
            final_target_path = None
            logger.info(f"[{user_id}] (Graph|1) [狀態切換] 檢測到明確的本地指令，導演視角從 'remote' 切換回 'local'。")
        elif new_viewing_mode == 'remote' and new_target_path and new_target_path != gs.remote_target_path:
            # 如果仍在遠程模式，但目標變了，則更新目標
            final_target_path = new_target_path
            logger.info(f"[{user_id}] (Graph|1) [狀態更新] 在遠程模式下，更新了觀察目標地點為: {final_target_path}")
        else:
            # 對於 "卡蓮呢?" 這類模糊指令，保持 remote 模式不變
            logger.info(f"[{user_id}] (Graph|1) [狀態保持] 未檢測到明確的本地切換信號，導演視角保持為 'remote'。")
    else: # gs.viewing_mode == 'local'
        # 從 local 切換到 remote 的條件保持不變
        if new_viewing_mode == 'remote' and new_target_path:
            final_viewing_mode = 'remote'
            final_target_path = new_target_path
            logger.info(f"[{user_id}] (Graph|1) [狀態切換] 檢測到遠程描述指令，導演視角從 'local' 切換到 'remote'。目標: {final_target_path}")

    # --- 步驟 4: [v3.1 核心修正] 持久化狀態變更 ---
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
    return {"scene_analysis": scene_analysis}
# 函式：[新] 場景感知與上下文恢復節點 (v3.1 - 遠程優先狀態保持) 結束





# 函式：[新] 記憶與 LORE 查詢節點 (v5.2 - 職責降級)
# 更新紀錄:
# v5.2 (2025-10-03): [架構簡化] 根據「上下文污染」分析，徹底移除了此節點查詢結構化 LORE (`_query_lore_from_entities`) 的職責。此節點現在的唯一任務是執行 RAG 檢索，將結構化 LORE 的數據與生成 Prompt 徹底解耦，以實現更純淨、更接近「RAG 直通」的上下文環境。
# v5.1 (2025-10-03): [災難性BUG修復] 根據 AttributeError，修正了在源頭清洗步驟中對 `ainvoke_with_rotation` 的呼叫方式。
# v5.0 (2025-10-03): [全新創建] 根據「直接RAG」架構創建此節點。
async def retrieve_and_query_node(state: ConversationGraphState) -> Dict:
    """[2] (職責降級) 僅執行 RAG 檢索，不再查詢結構化 LORE。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    
    # 檢查上下文是否已被恢復 (此邏輯保持不變)
    if state.get('raw_lore_objects') is not None:
        logger.info(f"[{user_id}] (Graph|2) Node: retrieve_and_query -> 檢測到已恢復的 LORE 上下文，將跳過重新查詢。")
        rag_context_dict = await ai_core.retrieve_and_summarize_memories(user_input)
        return {
            "rag_context": rag_context_dict.get("summary", "無相關長期記憶。"),
            # 保持傳遞 raw_lore_objects 以供後續節點（如 expansion）使用
            "raw_lore_objects": state['raw_lore_objects'],
            "sanitized_query_for_tools": user_input,
            "last_response_text": state.get('last_response_text')
        }

    logger.info(f"[{user_id}] (Graph|2) Node: retrieve_and_query -> 正在執行 RAG 檢索...")
    
    # 清洗使用者輸入的邏輯保持不變
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

    # 只執行 RAG 檢索
    rag_context_dict = await ai_core.retrieve_and_summarize_memories(sanitized_query)
    rag_context_str = rag_context_dict.get("summary", "沒有檢索到相關的長期記憶。")

    # [v5.2 核心修正] 移除對 _query_lore_from_entities 的調用
    # 我們仍然需要一個 planning_subjects 的來源，這裡我們從 RAG 的結果中粗略提取
    # 注意：這一步驟是為了讓 expansion_decision_and_execution_node 能夠運作
    # 但這個數據不會再污染最終的生成 Prompt
    all_lores = await lore_book.get_all_lores_for_user(user_id)
    
    logger.info(f"[{user_id}] (Graph|2) RAG 檢索完成。")
    
    return {
        "rag_context": rag_context_str,
        # 為了讓 expansion 節點能運作，我們傳遞一個空的 LORE 列表
        # 後續 expansion 節點會自己處理 LORE 的獲取
        "raw_lore_objects": [], 
        "sanitized_query_for_tools": sanitized_query
    }
# 函式：[新] 記憶與 LORE 查詢節點 (v5.2 - 職責降級) 結束



# 函式：[新] LORE 擴展決策與執行節點 (v6.0 - 輕量級骨架)
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

    # 獲取所有現有 LORE 的輕量級摘要
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
        # 即使不擴展，也要將所有現有的 LORE 傳遞給下一個節點
        return {"planning_subjects": [lore.structured_content for lore in all_lores if lore.structured_content]}

    logger.info(f"[{user_id}] (Graph|3) 決策結果：需要擴展。理由: {decision.reasoning}。正在執行LORE擴展...")
    
    newly_created_lores = []
    try:
        # 這裡我們調用一個更通用的 LORE 擴展管線，而不僅僅是場景選角
        expansion_prompt_template = ai_core.get_lore_expansion_pipeline_prompt()
        
        # 準備已知 LORE 實體列表供管線參考
        existing_lore_names = [lore.structured_content.get("name") for lore in all_lores if lore.structured_content and lore.structured_content.get("name")]
        
        expansion_prompt = ai_core._safe_format_prompt(expansion_prompt_template, {
            "user_input": safe_query_text,
            "existing_lore_json": json.dumps(existing_lore_names, ensure_ascii=False)
        })
        
        # 該管線會返回一個 CanonParsingResult 物件
        expansion_result = await ai_core.ainvoke_with_rotation(
            expansion_prompt,
            output_schema=CanonParsingResult,
            retry_strategy='euphemize'
        )
        
        if expansion_result:
            # 調用一個輔助函式來處理並保存所有新創建的 LORE 骨架
            # 注意：這裡的實現假設 _resolve_and_save_parsed_canon 是一個能夠處理 CanonParsingResult 的新輔助函式
            # 我們將在後續步驟中完善它
            newly_created_lores = await ai_core.parse_and_create_lore_from_canon_object(expansion_result)
            created_names = [lore.structured_content.get("name", "未知") for lore in newly_created_lores if lore.structured_content]
            logger.info(f"[{user_id}] (Graph|3) 擴展成功，創建了 {len(created_names)} 個新實體骨架: {created_names}")
        
    except Exception as e:
        logger.error(f"[{user_id}] (Graph|3) LORE擴展執行時發生錯誤: {e}", exc_info=True)
    
    # 合併原始 LORE 和新創建的 LORE
    final_lore_objects = all_lores + newly_created_lores
    final_lores_map = {lore.key: lore for lore in final_lore_objects}
    
    # 將所有 LORE 的結構化部分傳遞給下一個節點
    return {"planning_subjects": [lore.structured_content for lore in final_lores_map.values() if lore.structured_content]}
# 函式：[新] LORE 擴展決策與執行節點 (v6.0 - 輕量級骨架) 結束

# 函式：[新] 前置工具調用節點 (v3.1 - 執行邏輯修正)
# 更新紀錄:
# v3.1 (2025-10-03): [災難性BUG修復] 根據潛在的後續錯誤分析，徹底重構了此函式的工具執行邏輯。新版本不再錯誤地將 `ToolCallPlan` 包裝成 `TurnPlan`，而是直接調用正確的、更底層的 `_execute_tool_call_plan` 輔助函式來執行工具，確保了工具調用流程的正確性和連貫性。
# v3.0 (2025-10-03): [全新創建] 根據「直接RAG」架構創建此節點。
# v2.1 (2025-10-14): [災難性BUG修復] 修正了 `CharacterAction` 驗證錯誤。
async def preemptive_tool_call_node(state: ConversationGraphState) -> Dict:
    """[4] (全新) 判斷並執行使用者指令中明確的、需要改變世界狀態的動作。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|4) Node: preemptive_tool_call -> 正在解析前置工具調用...")

    tool_parsing_chain_prompt = ai_core.get_preemptive_tool_parsing_chain()
    # 從上一個節點獲取所有相關角色的名字
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
        # [v3.1 核心修正] 直接調用正確的工具計畫執行函式
        if not ai_core.profile:
             raise Exception("AI Profile尚未初始化，無法獲取當前地點。")
        current_location = ai_core.profile.game_state.location_path
        
        # 注意：_execute_tool_call_plan 現在需要能夠處理核心動作工具
        # 我們將在後續步驟中確保這一點
        results_summary, _ = await ai_core._execute_tool_call_plan(tool_call_plan, current_location)
        
    except Exception as e:
        logger.error(f"[{user_id}] (Graph|4) 前置工具執行時發生錯誤: {e}", exc_info=True)
        results_summary = f"系統事件：工具執行時發生嚴重錯誤: {e}"
    finally:
        tool_context.set_context(None, None)
    
    logger.info(f"[{user_id}] (Graph|4) 前置工具執行完畢。")
    return {"tool_results": results_summary}
# 函式：[新] 前置工具調用節點 (v3.1 - 執行邏輯修正) 結束






# 函式：[新] 世界快照組裝節點 (v3.2 - 職責降級)
# 更新紀錄:
# v3.2 (2025-10-03): [架構簡化] 根據「上下文污染」分析與「RAG直通」策略，徹底簡化了此節點的職責。它不再從 `planning_subjects` 或 `tool_results` 中拼接複雜的場景描述，而是只負責將最核心的設定（世界觀、AI設定）和最關鍵的 RAG 檢索結果 (`retrieved_context`) 填入模板，從而創建一個更純淨、污染更少的上下文快照。
# v3.1 (2025-10-03): [災難性BUG修復] 根據 KeyError，在 context_vars 字典中補全了 `explicit_character_files_context` 鍵。
# v3.0 (2025-10-03): [全新創建] 根據「直接RAG」架構創建此節點。
async def assemble_world_snapshot_node(state: ConversationGraphState) -> Dict:
    """[5] (職責降級) 僅組裝一個包含 RAG 和核心設定的純淨上下文快照。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|5) Node: assemble_world_snapshot -> 正在組裝【純淨版】上下文快照...")
    
    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|5) 致命錯誤: ai_core.profile 未加載！")
        return {"world_snapshot": "錯誤：AI Profile 丟失。"}
        
    gs = ai_core.profile.game_state
    
    # [v3.2 核心修正] context_vars 現在只包含最核心、最不可能造成污染的信息
    context_vars = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', "無相關長期記憶。"),
        
        # --- 以下為模板中必須存在但在此簡化流程中可以提供安全預設值的佔位符 ---
        'possessions_context': f"團隊庫存: {', '.join(gs.inventory) or '空的'}",
        'quests_context': "當前無任務。",
        'location_context': f"當前地點: {' > '.join(gs.location_path)}",
        'npc_context': "（上下文已由RAG提供）",
        'relevant_npc_context': "（上下文已由RAG提供）",
        'explicit_character_files_context': "（上下文已由RAG提供）",
        
        # --- 導演視角相關信息保持不變 ---
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': gs.viewing_mode,
        'remote_target_path_str': " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定",
        'scene_rules_context': "（上下文已由RAG提供）"
    }
    
    # 確保 world_snapshot_template 存在
    if not ai_core.world_snapshot_template:
        logger.error(f"[{user_id}] (Graph|5) 致命錯誤: world_snapshot_template 未加載！")
        return {"world_snapshot": "錯誤：世界快照模板丟失。"}

    final_world_snapshot = ai_core._safe_format_prompt(ai_core.world_snapshot_template, context_vars)
    
    logger.info(f"[{user_id}] (Graph|5) 【純淨版】上下文快照組裝完畢。")
    return {"world_snapshot": final_world_snapshot}
# 函式：[新] 世界快照組裝節點 (v3.2 - 職責降級) 結束




# 函式：[新] 最終生成節點 (v8.1 - 模板簡化)
# 更新紀錄:
# v8.1 (2025-10-03): [架構簡化] 根據「RAG直通」策略，徹底重寫了此節點的 Prompt 組合邏輯。它不再依賴於 `world_snapshot`，而是採用了一個更簡潔、更直接的模板，只將最核心的元素（越獄指令、RAG上下文、對話歷史、使用者輸入）拼接在一起，最大限度地減少了上下文污染，提高了生成成功率。
# v8.0 (2025-10-03): [全新創建] 根據「永久性轟炸」架構創建此節點。
# v7.0 (2025-10-15): [架構簡化] 移除了意圖分類的判斷。
async def final_generation_node(state: ConversationGraphState) -> Dict:
    """[6] (模板簡化) 组装一個純淨的、類似 RAG 直通的 Prompt，並調用生成鏈。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    # 注意：我們現在直接從 state 中獲取 rag_context，而不是從 world_snapshot 中解析
    rag_context = state.get('rag_context', '（無相關長期記憶。）')
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|6) Node: final_generation -> 启动【RAG直通模式】最终生成流程...")

    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|6) 致命錯誤: ai_core.profile 未加載！")
        return {"llm_response": "（錯誤：AI Profile 丟失，無法生成回應。）"}

    historical_context = await _get_summarized_chat_history(ai_core, user_id)
    
    plot_anchor = "（無）"
    continuation_keywords = ["继续", "繼續", "然後呢", "接下來", "go on", "continue"]
    if any(user_input.strip().lower().startswith(kw) for kw in continuation_keywords):
        last_response = state.get('last_response_text')
        if last_response:
            plot_anchor = last_response
            logger.info(f"[{user_id}] (Graph|6) 已成功為連續性指令設置【劇情錨點】。")

    # [v8.1 核心修正] 使用一個更簡潔、污染更少的 Prompt 模板
    final_prompt_template = """{core_protocol_prompt}

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
{ai_name}:"""

    full_prompt = ai_core._safe_format_prompt(
        final_prompt_template,
        {
            "core_protocol_prompt": ai_core.core_protocol_prompt,
            "rag_context": rag_context,
            "plot_anchor": plot_anchor,
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
# 函式：[新] 最終生成節點 (v8.1 - 模板簡化) 結束




  

# 函式：[新] 驗證、學習與持久化節點 (v3.0 - 生成與學習分離)
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
    
    # 創建上下文快照以供背景任務使用
    snapshot_for_analysis = {
        "user_input": user_input,
        "final_response": clean_response,
    }
    
    # 異步觸發背景學習（調用我們在第五階段重構的、絕對安全的LORE提取函式）
    asyncio.create_task(ai_core._background_lore_extraction(snapshot_for_analysis))

    # 使用我們新的持久化函式保存對話歷史
    # 注意：_save_interaction_to_dbs 內部會處理編碼和雙重持久化
    await ai_core._save_interaction_to_dbs(f"使用者: {user_input}\n\nAI:\n{clean_response}")
    logger.info(f"[{user_id}] (Graph|7) 對話歷史已更新並進行雙重持久化。")

    # 為下一輪的 "continue" 指令創建快照
    snapshot_for_next_turn = {
        "raw_lore_objects": state.get("raw_lore_objects", []),
        "last_response_text": clean_response
    }
    ai_core.last_context_snapshot = snapshot_for_next_turn
    logger.info(f"[{user_id}] (Graph|7) 已為下一輪創建上下文快照。")

    logger.info(f"[{user_id}] (Graph|7) 狀態持久化完成。")
    
    return {"final_output": clean_response}
# 函式：[新] 驗證、學習與持久化節點 結束





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




# 函式：創建設定圖 (v36.0 - 終極簡化)
# 更新紀錄:
# v36.0 (2025-10-03): [重大架構簡化] 根據「決策與執行合一」的最終策略，從創世圖中徹底移除了 `world_genesis` 節點。初始地點的選擇和 LORE 創建職責，現已完全整合進 `generate_opening_scene` 節點內部。此修改將創世流程簡化為兩個核心步驟（補完檔案 -> 生成場景），從根本上解決了因多節點數據流不同步而導致的地點不一致問題，同時提高了創世效率。
# v35.0 (2025-10-03): [重大架構重構] 徹底移除了 `process_canon` 節點，實現了 LORE 的「即時與增量」學習。
# v34.0 (2025-10-03): [重大架構簡化] 從創世圖中徹底移除了 `process_canon` 節點及其相關的邊。
def create_setup_graph() -> StateGraph:
    """創建設定圖 (v36.0 - 終極簡化版)"""
    graph = StateGraph(SetupGraphState)

    graph.add_node("complete_profiles", complete_profiles_node)
    # [v36.0 核心修正] 徹底移除 world_genesis 節點
    # graph.add_node("world_genesis", world_genesis_node)
    graph.add_node("generate_opening_scene", generate_opening_scene_node)
    
    graph.set_entry_point("complete_profiles")
    
    # [v36.0 核心修正] 調整邊的連接，跳過 world_genesis
    graph.add_edge("complete_profiles", "generate_opening_scene")
    graph.add_edge("generate_opening_scene", END)
    
    return graph.compile()
# 函式：創建設定圖



