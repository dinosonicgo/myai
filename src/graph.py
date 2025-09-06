# src/graph.py 的中文註釋(v21.1 - 完整性修正與拓撲健壯化)
# 更新紀錄:
# v21.1 (2025-09-10): [災難性BUG修復] 恢復了所有被先前版本錯誤省略的 `SetupGraph` 相關節點的完整程式碼，解決了 NameError 問題。同時，對主圖的路由匯合點 (Junction) 進行了標準化重構，提高了圖拓撲的健壯性。
# v21.0 (2025-09-09): [重大架構重構] 根據“一功能一節點”藍圖，對圖的拓撲結構進行了根本性的精細化重構。
# v20.1 (2025-09-06): [災難性BUG修復] 徹底修正了圖的拓撲定義。

import sys
print(f"[DEBUG] graph.py loaded from: {__file__}", file=sys.stderr)
import asyncio
import json
import re
from typing import Dict, List, Literal, Optional, Any

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

# --- 主對話圖 (Main Conversation Graph) 的節點 v21.1 ---


# 函式：場景與動作分析節點
# 更新紀錄:
# v1.0 (2025-09-13): [恢復] 恢复在重构中被遗漏的 SFW 路径核心节点，用于判断本地/远程视角。
async def scene_and_action_analysis_node(state: ConversationGraphState) -> Dict:
    """[SFW Path] 專用節點，分析 SFW 場景的視角（本地 vs 遠程）。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: scene_and_action_analysis [SFW Path] -> 正在進行場景視角分析...")

    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph) 在 scene_and_action_analysis 中 ai_core.profile 未加載。")
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='錯誤：AI profile 未加載。', action_summary=user_input)}

    current_location_path = ai_core.profile.game_state.location_path
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
    return {"scene_analysis": scene_analysis}
# 函式：場景與動作分析節點


# 函式：視角模式路由器
# 更新紀錄:
# v1.0 (2025-09-13): [恢復] 恢复在重构中被遗漏的 SFW 路径核心路由器，用于分发本地/远程流量。
def route_viewing_mode(state: ConversationGraphState) -> Literal["remote_scene", "local_scene"]:
    """[SFW Path] 根據視角分析結果，決定是生成遠程場景還是繼續本地流程。"""
    user_id = state['user_id']
    scene_analysis = state.get("scene_analysis")
    if scene_analysis and scene_analysis.viewing_mode == 'remote':
        logger.info(f"[{user_id}] (Graph) Router: SFW 視角分析為遠程，進入 remote_sfw_planning。")
        return "remote_scene"
    else:
        logger.info(f"[{user_id}] (Graph) Router: SFW 視角分析為本地，繼續本地主流程。")
        return "local_scene"
# 函式：視角模式路由器

# --- 階段一：感知 (Perception) ---

# 函式：[新建] 導演視角狀態管理節點 (v1.0)
# 更新紀錄:
# v1.0 (2025-09-06): [災難性BUG修復] 創建此新節點，專門用於調用 ai_core._update_viewing_mode。將其插入圖的早期流程中，確保在任何規劃開始之前，系統的“導演視角”（本地或遠程）狀態都已被明確設定和持久化。
async def update_viewing_mode_node(state: ConversationGraphState) -> None:
    """調用 ai_core 中的輔助函式來更新並持久化導演視角模式。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph) Node: update_viewing_mode -> 正在更新並持久化導演視角...")
    
    await ai_core._update_viewing_mode(state)
    
    # 這個節點不直接返回更新到 state，因為 ai_core 的方法會直接修改資料庫
    # 和 ai_core 實例內的 GameState。後續節點將讀取到這個最新的狀態。
    return {}
# 函式：[新建] 導演視角狀態管理節點 (v1.0)


async def classify_intent_node(state: ConversationGraphState) -> Dict:
    """[1] 圖的入口點，唯一職責是對原始輸入進行意圖分類。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|1) Node: classify_intent -> 正在對 '{user_input[:30]}...' 進行意圖分類...")
    
    classification_chain = ai_core.get_intent_classification_chain()
    classification_result = await ai_core.ainvoke_with_rotation(
        classification_chain,
        {"user_input": user_input},
        retry_strategy='none'
    )
    
    if not classification_result:
        logger.warning(f"[{user_id}] (Graph|1) 意圖分類鏈失敗，啟動安全備援，預設為 SFW。")
        classification_result = IntentClassificationResult(intent_type='sfw', reasoning="安全備援：分類鏈失敗。")
        
    return {"intent_classification": classification_result}

async def retrieve_memories_node(state: ConversationGraphState) -> Dict:
    """[2] 專用記憶檢索節點，執行RAG操作。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|2) Node: retrieve_memories -> 正在檢索相關長期記憶...")
    
    # 這是 ai_core.py 中為新節點準備的輔助函式
    rag_context_str = await ai_core.retrieve_and_summarize_memories(user_input)
    return {"rag_context": rag_context_str}

async def query_lore_node(state: ConversationGraphState) -> Dict:
    """[3] 專用LORE查詢節點，從資料庫獲取原始LORE對象。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    intent_type = state['intent_classification'].intent_type
    logger.info(f"[{user_id}] (Graph|3) Node: query_lore -> 正在查詢相關LORE實體...")
    
    is_remote = intent_type == 'nsfw_descriptive'
    # 這是 ai_core.py 中為新節點準備的輔助函式
    raw_lore_objects = await ai_core._query_lore_from_entities(user_input, is_remote_scene=is_remote)
    return {"raw_lore_objects": raw_lore_objects}

# 函式：專用上下文組裝節點 (v1.1 - 傳遞原始LORE)
# 更新紀錄:
# v1.1 (2025-09-06): [災難性BUG修復] 修改此節點的返回值，使其除了生成格式化的 `structured_context` 外，還將未經修改的 `raw_lore_objects` 直接透傳下去。這是實現“LORE事實鎖定”機制的關鍵一步，確保後續的規劃節點能夠訪問到完整的、未經摘要的原始 LORE 數據。
# v1.0 (2025-09-12): [架構重構] 創建此專用函式，將上下文格式化邏輯分離。
async def assemble_context_node(state: ConversationGraphState) -> Dict:
    """[4] 專用上下文組裝節點，將原始LORE格式化為LLM可讀的字符串，並透傳原始LORE對象。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    raw_lore = state['raw_lore_objects']
    intent_type = state['intent_classification'].intent_type
    logger.info(f"[{user_id}] (Graph|4) Node: assemble_context -> 正在組裝最終上下文簡報並透傳原始LORE...")
    
    is_remote = intent_type == 'nsfw_descriptive'
    structured_context = ai_core._assemble_context_from_lore(raw_lore, is_remote_scene=is_remote)
    
    # [v1.1 核心修正] 將原始 LORE 對象也加入返回字典中，以便後續節點使用
    return {
        "structured_context": structured_context,
        "raw_lore_objects": raw_lore 
    }
# 函式：專用上下文組裝節點 (v1.1 - 傳遞原始LORE)
    
# 函式：LORE擴展決策節點 (v2.0 - 飽和度分析)
# 更新紀錄:
# v2.0 (2025-09-06): [災難性BUG修復] 徹底重構了此節點的邏輯。現在，它會在調用決策鏈之前，先對當前場景的 LORE 數據（特別是在場 NPC 數量）進行量化分析，得到“飽和度”指標。然後將此飽和度信息作為強力約束注入到 Prompt 中，指示 LLM 只有在場景確實“空曠”時才進行擴展。此修改旨在從根本上解決 AI 在細節豐富的場景中無限創造新 LORE 的問題。
# v1.0 (2025-09-09): [架構重構] 創建此專用節點。
async def expansion_decision_node(state: ConversationGraphState) -> Dict:
    """[5] LORE擴展決策節點，引入飽和度分析以避免無限擴展。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    raw_lore_objects = state.get('raw_lore_objects', [])
    logger.info(f"[{user_id}] (Graph|5) Node: expansion_decision -> 正在結合LORE飽和度分析，判斷是否擴展...")
    
    # [v2.0 核心修正] LORE 飽和度分析
    npc_count = 0
    location_has_description = False
    if isinstance(raw_lore_objects, list):
        for lore in raw_lore_objects:
            if lore.category == 'npc_profile':
                npc_count += 1
            elif lore.category == 'location_info':
                if lore.content and lore.content.get('description'):
                    location_has_description = True

    saturation_analysis = (
        f"- **當前在場NPC數量**: {npc_count}\n"
        f"- **當前地點是否有詳細描述**: {'是' if location_has_description else '否'}"
    )
    logger.info(f"[{user_id}] (Graph|5) LORE飽和度分析結果:\n{saturation_analysis}")

    chat_history_manager = ai_core.session_histories.get(user_id, ChatMessageHistory())
    recent_dialogue = "\n".join([f"{'使用者' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in chat_history_manager.messages[-6:]])
    
    decision_chain = ai_core.get_expansion_decision_chain()
    decision = await ai_core.ainvoke_with_rotation(
        decision_chain, 
        {
            "user_input": user_input, 
            "recent_dialogue": recent_dialogue,
            "saturation_analysis": saturation_analysis # 將飽和度分析注入 Prompt
        },
        retry_strategy='euphemize'
    )

    if not decision:
        logger.warning(f"[{user_id}] (Graph|5) LORE擴展決策鏈失敗，安全備援為不擴展。")
        decision = ExpansionDecision(should_expand=False, reasoning="安全備援：決策鏈失敗。")
    
    logger.info(f"[{user_id}] (Graph|5) LORE擴展決策: {decision.should_expand}。理由: {decision.reasoning}")
    return {"expansion_decision": decision}
# 函式：LORE擴展決策節點 (v2.0 - 飽和度分析)





# 函式：專用的LORE擴展執行節點 (v2.1 - 架構遷移適配)
# 更新紀錄:
# v2.1 (2025-09-06): [重大架構重構] 更新了此節點的實現，使其調用 ai_core._add_cast_to_scene 而不是依賴於 discord_bot.py。此修改是為了配合 _add_cast_to_scene 函式向 ai_core.py 的遷移，理順了數據流並消除了不健康的模組間依賴。
# v2.0 (2025-09-06): [災難性BUG修復] 引入了輸入委婉化處理，以解決 LORE 創建被內容審查攔截的問題。
# v1.0 (2025-09-09): [架構重構] 創建此專用節點。
async def lore_expansion_node(state: ConversationGraphState) -> Dict:
    """[6A] 專用的LORE擴展執行節點，執行選角並刷新上下文。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|6A) Node: lore_expansion -> 正在執行場景選角與LORE擴展...")
    
    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|6A) ai_core.profile 未加載，跳過 LORE 擴展。")
        return {}

    current_location_path = ai_core.profile.game_state.location_path
    game_context_for_casting = json.dumps(state.get('structured_context', {}), ensure_ascii=False, indent=2)
    
    euphemization_chain = ai_core.get_euphemization_chain()
    
    entity_extraction_chain = ai_core.get_entity_extraction_chain()
    entity_result = await ai_core.ainvoke_with_rotation(entity_extraction_chain, {"text_input": user_input})
    keywords_for_euphemization = entity_result.names if entity_result and entity_result.names else user_input.split()
    
    safe_dialogue_context = await ai_core.ainvoke_with_rotation(
        euphemization_chain,
        {"keywords": keywords_for_euphemization},
        retry_strategy='none'
    )
    
    if not safe_dialogue_context:
        logger.warning(f"[{user_id}] (Graph|6A) 委婉化對話上下文失敗，將使用原始輸入，這可能存在風險。")
        safe_dialogue_context = user_input

    logger.info(f"[{user_id}] (Graph|6A) 已生成用於選角鏈的安全上下文: '{safe_dialogue_context}'")

    cast_result = await ai_core.ainvoke_with_rotation(
        ai_core.get_scene_casting_chain(),
        {
            "world_settings": ai_core.profile.world_settings or "", 
            "current_location_path": current_location_path, 
            "game_context": game_context_for_casting, 
            "recent_dialogue": safe_dialogue_context
        },
        retry_strategy='euphemize'
    )
    
    updates: Dict[str, Any] = {}
    if cast_result and (cast_result.newly_created_npcs or cast_result.supporting_cast):
        # [v2.1 核心修正] 直接調用 ai_core 上的方法
        await ai_core._add_cast_to_scene(cast_result)
        
        logger.info(f"[{user_id}] (Graph|6A) 選角完成，正在刷新LORE和上下文...")
        intent_type = state['intent_classification'].intent_type
        is_remote = intent_type == 'nsfw_descriptive'
        refreshed_lore = await ai_core._query_lore_from_entities(user_input, is_remote_scene=is_remote)
        refreshed_context = ai_core._assemble_context_from_lore(refreshed_lore, is_remote_scene=is_remote)
        updates = {"raw_lore_objects": refreshed_lore, "structured_context": refreshed_context}
    else:
         logger.info(f"[{user_id}] (Graph|6A) 場景選角鏈未返回新角色（可能因內容審查或無創造必要），無需刷新。")

    return updates
# 函式：專用的LORE擴展執行節點 (v2.1 - 架構遷移適配)

# --- 階段二：規劃 (Planning) ---




# 函式：NSFW 初步規劃節點
# 更新紀錄:
# v1.0 (2025-09-15): [重大架構重構] 创建此新节点，作为 NSFW 思维链的第一步。
async def nsfw_initial_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7B.1] NSFW思维链-步骤1: 生成初步的行动计划草稿。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|7B.1) Node: nsfw_initial_planning -> 正在生成NSFW初步行动计划...")
    
    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}

    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': state.get('structured_context', {}).get('possessions_context', ''),
        'quests_context': state.get('structured_context', {}).get('quests_context', ''),
        'location_context': state.get('structured_context', {}).get('location_context', ''),
        'npc_context': state.get('structured_context', {}).get('npc_context', ''),
        'relevant_npc_context': state.get('structured_context', {}).get('relevant_npc_context', ''),
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)

    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_nsfw_initial_planning_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "world_snapshot": world_snapshot, 
            "user_input": state['messages'][-1].content
        },
        retry_strategy='force'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：NSFW初步規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}
# 函式：NSFW 初步規劃節點



# 函式：NSFW 词汇注入節點 (v1.2 - 數據流修正)
# 更新紀錄:
# v1.2 (2025-09-05): [災難性BUG修復] 修正了調用鏈時的參數傳遞，補上了缺失的 `system_prompt`。這是實現“指令淹沒”策略、解決內容審查問題的關鍵一步。
# v1.1 (2025-09-15): [災難性BUG修復] 修正了 full_context_dict 的构建逻辑，从一个过于简化的版本恢复为包含所有模板所需键的完整版本，解决了因缺少 `world_settings` 等键而导致的 KeyError。
# v1.0 (2025-09-15): [重大架構重構] 创建此新节点，作为 NSFW 思维链的第二步。
async def nsfw_lexicon_injection_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7B.2] NSFW思维链-步骤2: 强制修正计划中的词汇为露骨术语。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph|7B.2) Node: nsfw_lexicon_injection -> 正在注入NSFW露骨词汇...")

    if not ai_core.profile or not turn_plan:
        return {}

    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': state.get('structured_context', {}).get('possessions_context', ''),
        'quests_context': state.get('structured_context', {}).get('quests_context', ''),
        'location_context': state.get('structured_context', {}).get('location_context', ''),
        'npc_context': state.get('structured_context', {}).get('npc_context', ''),
        'relevant_npc_context': state.get('structured_context', {}).get('relevant_npc_context', ''),
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)
    
    corrected_plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_nsfw_lexicon_injection_chain(),
        {
            # [v1.2 核心修正] 傳入完整的系統指令
            "system_prompt": ai_core.profile.one_instruction,
            "action_sexual_content_prompt": ai_core.modular_prompts.get("action_sexual_content", "警告:性愛模組未加載"),
            "world_snapshot": world_snapshot,
            "turn_plan_json": turn_plan.model_dump_json(indent=2)
        },
        retry_strategy='force'
    )
    if not corrected_plan:
        logger.warning(f"[{user_id}] (Graph|7B.2) NSFW词汇注入鏈返回空值，保留原始计划。")
        return {}
        
    return {"turn_plan": corrected_plan}
# 函式：NSFW 词汇注入節點 (v1.2 - 數據流修正)

# 函式：SFW規劃節點 (v21.4 - 事實鎖定)
# 更新紀錄:
# v21.4 (2025-09-06): [災難性BUG修復] 增加了 `full_lore_records_json` 參數，並將其注入到 `world_snapshot` 模板中。這是實現“LORE事實鎖定”機制的關鍵一步，它將完整的、未經摘要的 LORE 數據以 JSON 形式提供給 LLM，並通過模板中的指令強制 LLM 遵循這些既定事實，從根本上解決 AI 篡改或遺忘 LORE（如將“鐵匠”變為“獵人”）的問題。
# v21.3 (2025-09-06): [災難性BUG修復] 更新了 `full_context_dict` 的構建邏輯以適配新的導演視角。
# v21.2 (2025-09-15): [架構重構] 移除了對 style_analysis 的依賴。
async def sfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7A] SFW路徑專用規劃器，生成結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|7A) Node: sfw_planning -> 正在生成SFW行動計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}

    gs = ai_core.profile.game_state
    
    # [v21.4 核心修正] 準備完整的 LORE JSON 數據
    raw_lore_objects = state.get('raw_lore_objects', [])
    full_lore_records_json = json.dumps([lore.model_dump() for lore in raw_lore_objects], ensure_ascii=False, indent=2)

    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': state.get('structured_context', {}).get('possessions_context', ''),
        'quests_context': state.get('structured_context', {}).get('quests_context', ''),
        'location_context': state.get('structured_context', {}).get('location_context', ''),
        'npc_context': state.get('structured_context', {}).get('npc_context', ''),
        'relevant_npc_context': state.get('structured_context', {}).get('relevant_npc_context', ''),
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': gs.viewing_mode,
        'remote_target_path_str': " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定",
        # [v21.4 新增]
        'full_lore_records_json': full_lore_records_json
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)
    
    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_sfw_planning_chain(), 
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot, 
            "user_input": state['messages'][-1].content, 
        },
        retry_strategy='euphemize'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：SFW規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}
# 函式：SFW規劃節點 (v21.4 - 事實鎖定)




# 函式：NSFW 風格合規節點 (v1.2 - 數據流修正)
# 更新紀錄:
# v1.2 (2025-09-05): [災難性BUG修復] 修正了調用鏈時的參數傳遞，補上了缺失的 `system_prompt`。這是實現“指令淹沒”策略、解決內容審查問題的關鍵一步。
# v1.1 (2025-09-15): [災難性BUG修復] 与 nsfw_lexicon_injection_node 同步，修正了 full_context_dict 的构建逻辑，解决了 KeyError。
# v1.0 (2025-09-15): [重大架構重構] 创建此新节点，作为 NSFW 思维链的最后一步。
async def nsfw_style_compliance_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7B.3] NSFW思维链-步骤3: 检查并补充对话，确保计划符合用户风格。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph|7B.3) Node: nsfw_style_compliance -> 正在进行NSFW风格合规检查...")

    if not ai_core.profile or not turn_plan:
        return {}

    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': state.get('structured_context', {}).get('possessions_context', ''),
        'quests_context': state.get('structured_context', {}).get('quests_context', ''),
        'location_context': state.get('structured_context', {}).get('location_context', ''),
        'npc_context': state.get('structured_context', {}).get('npc_context', ''),
        'relevant_npc_context': state.get('structured_context', {}).get('relevant_npc_context', ''),
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)

    final_plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_nsfw_style_compliance_chain(),
        {
            # [v1.2 核心修正] 傳入完整的系統指令
            "system_prompt": ai_core.profile.one_instruction,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot,
            "turn_plan_json": turn_plan.model_dump_json(indent=2)
        },
        retry_strategy='force'
    )
    if not final_plan:
        logger.warning(f"[{user_id}] (Graph|7B.3) NSFW风格合规鏈返回空值，保留修正前计划。")
        return {}

    return {"turn_plan": final_plan}
# 函式：NSFW 風格合規節點 (v1.2 - 數據流修正)


# 函式：遠程 SFW 規劃節點 (v1.3 - 適配新世界快照)
# 更新紀錄:
# v1.3 (2025-09-06): [災難性BUG修復] 更新了 `full_context_dict` 的構建邏輯，以傳遞新的 `player_location`, `viewing_mode`, `remote_target_path_str` 等變數給 `world_snapshot` 模板，解決了因模板更新而導致的 KeyError。
# v1.2 (2025-09-15): [災難性BUG修復] 在 ainvoke 的參數中補上了缺失的 `response_style_prompt`。
# v1.1 (2025-09-14): [架構重構] 修改此節點，使其能接收並使用上游 `scene_and_action_analysis_node` 解析出的 `target_location_path`。
async def remote_sfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7D] SFW 描述路徑專用規劃器，生成遠景場景的結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|7D) Node: remote_sfw_planning -> 正在生成遠程SFW場景計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}

    gs = ai_core.profile.game_state
    scene_analysis = state.get('scene_analysis')
    
    if not scene_analysis or not scene_analysis.target_location_path:
        logger.error(f"[{user_id}] (Graph|7D) 錯誤：進入 remote_sfw_planning_node 但未找到 target_location_path。")
        return {"turn_plan": TurnPlan(thought="錯誤：未能解析出遠程觀察的目標地點。", character_actions=[])}
    
    target_location_path_str = " > ".join(scene_analysis.target_location_path)

    # [v1.3 新增] 為 world_snapshot 準備完整的上下文
    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': "(遠程觀察模式)",
        'quests_context': "(遠程觀察模式)",
        'location_context': f"遠程觀察地點: {target_location_path_str}",
        'npc_context': state.get('structured_context', {}).get('npc_context', ''),
        'relevant_npc_context': state.get('structured_context', {}).get('relevant_npc_context', ''),
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': 'remote', # 強制為 remote
        'remote_target_path_str': target_location_path_str
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)

    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_remote_sfw_planning_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_settings": ai_core.profile.world_settings, 
            "target_location_path_str": target_location_path_str,
            "remote_scene_context": json.dumps(state['structured_context'], ensure_ascii=False), 
            "user_input": state['messages'][-1].content,
            "username": ai_core.profile.user_profile.name,
            "ai_name": ai_core.profile.ai_profile.name
        },
        retry_strategy='euphemize'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：遠程SFW規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}
# 函式：遠程 SFW 規劃節點 (v1.3 - 適配新世界快照)




# 函式：遠程NSFW規劃節點 (v21.3 - 適配新世界快照)
# 更新紀錄:
# v21.3 (2025-09-06): [災難性BUG修復] 更新了 `full_context_dict` 的構建邏輯，以傳遞新的 `player_location`, `viewing_mode`, `remote_target_path_str` 等變數給 `world_snapshot` 模板，解決了因模板更新而導致的 KeyError。
# v21.2 (2025-09-15): [災難性BUG修復] 在 ainvoke 的參數中補上了缺失的 `response_style_prompt`。
# v21.1 (2025-09-14): [架構重構] 修改此節點，使其能接收並使用上游 `scene_and_action_analysis_node` 解析出的 `target_location_path`。
async def remote_nsfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7C] NSFW描述路徑專用規劃器，生成遠景場景的結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|7C) Node: remote_nsfw_planning -> 正在生成遠程NSFW場景計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}
    
    gs = ai_core.profile.game_state
    scene_analysis = state.get('scene_analysis')

    if not scene_analysis or not scene_analysis.target_location_path:
        logger.error(f"[{user_id}] (Graph|7C) 錯誤：進入 remote_nsfw_planning_node 但未找到 target_location_path。")
        return {"turn_plan": TurnPlan(thought="錯誤：未能解析出遠程觀察的目標地點。", character_actions=[])}

    target_location_path_str = " > ".join(scene_analysis.target_location_path)

    # [v21.3 新增] 為 world_snapshot 準備完整的上下文
    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': "(遠程觀察模式)",
        'quests_context': "(遠程觀察模式)",
        'location_context': f"遠程觀察地點: {target_location_path_str}",
        'npc_context': state.get('structured_context', {}).get('npc_context', ''),
        'relevant_npc_context': state.get('structured_context', {}).get('relevant_npc_context', ''),
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': 'remote', # 強制為 remote
        'remote_target_path_str': target_location_path_str
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)

    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_remote_nsfw_planning_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_settings": ai_core.profile.world_settings,
            "target_location_path_str": target_location_path_str,
            "remote_scene_context": json.dumps(state['structured_context'], ensure_ascii=False), 
            "user_input": state['messages'][-1].content,
            "username": ai_core.profile.user_profile.name,
            "ai_name": ai_core.profile.ai_profile.name
        },
        retry_strategy='force'
    )
    if not plan:
        plan = TurnPlan(thought="安全備援：遠程NSFW規劃鏈失敗。", character_actions=[])
    return {"turn_plan": plan}
# 函式：遠程NSFW規劃節點 (v21.3 - 適配新世界快照)

# --- 階段三：執行與渲染 (Execution & Rendering) ---

async def tool_execution_node(state: ConversationGraphState) -> Dict[str, str]:
    """[8] 統一的工具執行節點。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph|8) Node: tool_execution -> 正在執行行動計劃中的工具...")
    
    if not plan or not plan.character_actions:
        return {"tool_results": "系統事件：無任何工具被調用。"}
    try:
        results_summary = await ai_core._execute_planned_actions(plan)
    except Exception as e:
        logger.error(f"[{user_id}] (Graph|8) 工具執行時發生未捕獲的異常: {e}", exc_info=True)
        results_summary = f"系統事件：工具執行時發生嚴重錯誤: {e}"
    finally:
        tool_context.set_context(None, None)
    
    return {"tool_results": results_summary}

# 函式：統一的敘事渲染節點 (v22.0 - 數據流修正)
# 更新紀錄:
# v22.0 (2025-09-05): [災難性BUG修復] 修正了調用鏈時的參數傳遞，補上了缺失的 `system_prompt` 和 `response_style_prompt`。這是實現“指令淹沒”策略、解決最終渲染環節被內容審查攔截的關鍵一步。
# v21.0 (2025-09-12): [架構重構] 強化此鏈，使其成為能夠處理所有類型 TurnPlan (SFW, NSFW, 遠景) 的統一“小說家”節點。
# v204.0 (2025-09-06): [重大功能修正] 賦予敘事鏈在計畫對話不足時，根據風格指令補充對話的權力。
async def narrative_rendering_node(state: ConversationGraphState) -> Dict[str, str]:
    """[9] 統一的敘事渲染節點，將行動計劃轉化為小說文本。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph|9) Node: narrative_rendering -> 正在將行動計劃渲染為小說...")

    if not turn_plan:
        return {"llm_response": "（系統錯誤：未能生成有效的行動計劃。）"}
    if turn_plan.execution_rejection_reason:
        return {"llm_response": turn_plan.execution_rejection_reason}
        
    narrative_text = await ai_core.ainvoke_with_rotation(
        ai_core.get_narrative_chain(),
        {
            # [v22.0 核心修正] 傳入完整的系統指令和風格指令
            "system_prompt": ai_core.profile.one_instruction if ai_core.profile else "預設系統指令",
            "response_style_prompt": ai_core.profile.response_style_prompt if ai_core.profile else "預設風格",
            "turn_plan": turn_plan
        },
        retry_strategy='force'
    )
    if not narrative_text:
        narrative_text = "（AI 在將計劃轉化為故事時遭遇了內容安全限制。）"
    return {"llm_response": narrative_text}
# 函式：統一的敘事渲染節點 (v22.0 - 數據流修正)

# --- 階段四：收尾 (Finalization) ---

async def validate_and_rewrite_node(state: ConversationGraphState) -> Dict:
    """[10] 統一的輸出驗證與淨化節點。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    initial_response = state['llm_response']
    logger.info(f"[{user_id}] (Graph|10) Node: validate_and_rewrite -> 正在對 LLM 原始輸出進行內容保全式淨化...")
    
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
    """[11] 統一的狀態持久化節點。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    clean_response = state['final_output']
    logger.info(f"[{user_id}] (Graph|11) Node: persist_state -> 正在持久化狀態與記憶...")
    
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

# --- 主對話圖的路由 v21.1 ---

def route_expansion_decision(state: ConversationGraphState) -> Literal["expand_lore", "continue_to_planner"]:
    """根據LORE擴展決策，決定是否進入擴展節點。"""
    if state.get("expansion_decision") and state["expansion_decision"].should_expand:
        return "expand_lore"
    else:
        return "continue_to_planner"



# --- 主對話圖的建構器 v21.1 ---








# 函式：創建主回應圖 (v24.1 - 路由修正)
# 更新紀錄:
# v24.1 (2025-09-06): [災難性BUG修復] 修正了 `planner_junction` 的路由邏輯。`nsfw_interactive` 意圖現在被正確地路由到 `nsfw_planning_node`，而不是錯誤地流向遠程規劃器。此修改解決了在連續的遠程觀察後發出本地互動指令時，系統仍然嘗試生成遠程場景的錯誤。
# v24.0 (2025-09-06): [災難性BUG修復] 引入了全新的 `update_viewing_mode_node` 節點以解決場景連續性混亂的問題。
# v23.0 (2025-09-05): [災難性BUG修復] 根據 ai_core.py v4.0 的重構，徹底移除了舊的 NSFW 思維鏈三節點。
def create_main_response_graph() -> StateGraph:
    graph = StateGraph(ConversationGraphState)
    
    # --- 1. 註冊所有節點 ---
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_memories", retrieve_memories_node)
    graph.add_node("query_lore", query_lore_node)
    graph.add_node("assemble_context", assemble_context_node)
    graph.add_node("expansion_decision", expansion_decision_node)
    graph.add_node("lore_expansion", lore_expansion_node)
    graph.add_node("scene_and_action_analysis", scene_and_action_analysis_node)
    graph.add_node("update_viewing_mode", update_viewing_mode_node)
    
    # SFW & 远景规划
    graph.add_node("sfw_planning", sfw_planning_node)
    graph.add_node("remote_nsfw_planning", remote_nsfw_planning_node)
    graph.add_node("remote_sfw_planning", remote_sfw_planning_node)
    graph.add_node("nsfw_planning", sfw_planning_node)

    # 统一出口节点
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("narrative_rendering", narrative_rendering_node)
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)
    
    graph.add_node("after_perception_junction", lambda state: {})
    graph.add_node("planner_junction", lambda state: {})

    # --- 2. 定義圖的拓撲結構 ---
    graph.set_entry_point("classify_intent")
    
    graph.add_edge("classify_intent", "scene_and_action_analysis")
    graph.add_edge("scene_and_action_analysis", "update_viewing_mode")
    graph.add_edge("update_viewing_mode", "retrieve_memories")
    
    graph.add_edge("retrieve_memories", "query_lore")
    graph.add_edge("query_lore", "assemble_context")
    graph.add_edge("assemble_context", "expansion_decision")
    
    graph.add_conditional_edges(
        "expansion_decision",
        route_expansion_decision,
        { "expand_lore": "lore_expansion", "continue_to_planner": "planner_junction" }
    )
    graph.add_edge("lore_expansion", "planner_junction")

    def route_to_planner(state: ConversationGraphState) -> str:
        """根據意圖分類和導演視角將流量路由到不同的規劃器。"""
        user_id = state['user_id']
        ai_core = state['ai_core']
        intent = state['intent_classification'].intent_type
        viewing_mode = ai_core.profile.game_state.viewing_mode if ai_core.profile else 'local'
        
        logger.info(f"[{user_id}] (Router) Routing to planner. Intent: '{intent}', Viewing Mode: '{viewing_mode}'")
        
        # [v24.1 核心路由修正]
        # 即使意圖是互動式的，如果導演視角仍然是 remote，則繼續遠程規劃
        if viewing_mode == 'remote':
            if 'nsfw' in intent:
                logger.info(f"[{user_id}] (Router) Mode is remote, intent is NSFW-like -> remote_nsfw_planning")
                return "remote_nsfw_planner"
            else:
                logger.info(f"[{user_id}] (Router) Mode is remote, intent is SFW-like -> remote_sfw_planning")
                return "remote_sfw_planning"
        else: # viewing_mode == 'local'
            if intent == 'nsfw_interactive':
                logger.info(f"[{user_id}] (Router) Mode is local, intent is nsfw_interactive -> nsfw_planning")
                return "nsfw_planner"
            else: # sfw_interactive or any other local intent
                logger.info(f"[{user_id}] (Router) Mode is local, intent is SFW-like -> sfw_planning")
                return "sfw_planner"


    graph.add_conditional_edges(
        "planner_junction",
        route_to_planner,
        { 
            "sfw_planner": "sfw_planning", 
            "nsfw_planner": "nsfw_planning",
            "remote_sfw_planner": "remote_sfw_planning",
            "remote_nsfw_planner": "remote_nsfw_planning"
        }
    )
    
    # 所有规划器都汇合到工具执行
    graph.add_edge("sfw_planning", "tool_execution")
    graph.add_edge("nsfw_planning", "tool_execution")
    graph.add_edge("remote_nsfw_planning", "tool_execution")
    graph.add_edge("remote_sfw_planning", "tool_execution")
    
    # 统一出口流程
    graph.add_edge("tool_execution", "narrative_rendering")
    graph.add_edge("narrative_rendering", "validate_and_rewrite")
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", END)
    
    return graph.compile()
# 函式：創建主回應圖 (v24.1 - 路由修正)



# --- 設定圖 (Setup Graph) 的節點與建構器 (完整版) ---

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
    
    completed_user_profile_task = ai_core.ainvoke_with_rotation(completion_chain, {"profile_json": ai_core.profile.user_profile.model_dump_json()}, retry_strategy='euphemize')
    completed_ai_profile_task = ai_core.ainvoke_with_rotation(completion_chain, {"profile_json": ai_core.profile.ai_profile.model_dump_json()}, retry_strategy='euphemize')
    
    completed_user_profile, completed_ai_profile = await asyncio.gather(completed_user_profile_task, completed_ai_profile_task)

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
    
    if not ai_core.profile:
        raise Exception("AI Profile is not loaded for world genesis.")

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
    """
    創建設定圖
    """
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
