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

# 函式：獲取格式化的聊天歷史 (v2.0 - 健壯性修正)
# 更新紀錄:
# v2.0 (2025-09-28): [健壯性] 在獲取 AI 名字時，增加了對 `ai_core.profile` 是否存在的檢查，並提供了一個 “AI” 作為備援名稱，以防止在罕見的初始化失敗場景下發生 AttributeError。
# v1.0 (2025-09-26): [全新創建] 創建此輔助函式，用於從 AI 核心實例中提取並格式化最近的對話歷史，供圖中的各個節點注入到提示詞中。
def _get_formatted_chat_history(ai_core: AILover, user_id: str, num_messages: int = 10) -> str:
    """從 AI 核心實例中提取並格式化最近的對話歷史。"""
    chat_history_manager = ai_core.session_histories.get(user_id, ChatMessageHistory())
    if not chat_history_manager.messages:
        return "（沒有最近的對話歷史）"
    
    recent_messages = chat_history_manager.messages[-num_messages:]
    
    formatted_history = []
    ai_name = ai_core.profile.ai_profile.name if ai_core.profile and ai_core.profile.ai_profile and ai_core.profile.ai_profile.name else "AI"
    
    for msg in recent_messages:
        role = "使用者" if isinstance(msg, HumanMessage) else ai_name
        formatted_history.append(f"{role}: {msg.content}")
        
    return "\n".join(formatted_history)
# 函式：獲取格式化的聊天歷史 (v2.0 - 健壯性修正)

# --- 主對話圖 (Main Conversation Graph) 的節點 v41.0 ---

# 函式：[全新] 確定性視角與模組節點 (v2.0 - 健壯檢測與模組加載)
# 更新紀錄:
# v2.0 (2025-09-29): [災難性BUG修復 & 功能擴展]
#    1. [健壯檢測] 徹底重寫了視角檢測邏輯，改為使用更強大的正則表達式在整個指令中搜索關鍵詞，不再依賴脆弱的 `startswith`，確保了“重新描述”等指令能被準確識別。
#    2. [模組加載] 新增了第二個核心職責：分析使用者輸入，判斷是否需要為當前回合加載特定的戰術指令模組（如性愛協議），並將其內容寫入圖狀態，供下游所有節點使用。
# v1.1 (2025-09-29): [災難性BUG修復] 將此節點修改為異步函式以解決 RuntimeError。
async def determine_viewing_mode_node(state: ConversationGraphState) -> Dict:
    """[1] 圖的新入口點。通過確定性分析，決定視角模式並為回合加載必要的戰術指令模組。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|1) Node: determine_viewing_mode -> 正在確定性地分析視角與指令模組...")

    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|1) ai_core.profile 未加載，無法繼續。")
        return {}

    # --- 步驟 1: 視角模式分析 ---
    gs = ai_core.profile.game_state
    descriptive_keywords = ["描述", "描寫", "看看", "觀察", "重新描述"]
    descriptive_pattern = re.compile(f"({'|'.join(descriptive_keywords)})")
    is_descriptive = bool(descriptive_pattern.search(user_input))

    needs_update = False
    if is_descriptive:
        if gs.viewing_mode != 'remote':
            gs.viewing_mode = 'remote'
            needs_update = True
            logger.info(f"[{user_id}] (Graph|1) 檢測到描述性指令，視角模式設定為: remote")
    else:
        if gs.viewing_mode != 'local':
            gs.viewing_mode = 'local'
            gs.remote_target_path = None
            needs_update = True
            logger.info(f"[{user_id}] (Graph|1) 未檢測到描述性指令，視角模式設定為: local")
            
    if needs_update:
        await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})

    # --- 步驟 2: 戰術指令模組分析 ---
    sexual_keywords = ["輪姦", "強姦", "口交", "舔", "吸吮", "肉棒", "肉穴", "插入", "交合", "做愛", "性交", "肛交", "抽插", "射精", "淫穴", "淫水", "調教", "自慰", "上我", "幹我", "操我", "騎上來", "含住", "脫光", "裸體", "高潮"]
    active_module_content = None
    if any(keyword in user_input for keyword in sexual_keywords):
        module_name = "action_sexual_content"
        active_module_content = ai_core.modular_prompts.get(module_name)
        if active_module_content:
            logger.info(f"[{user_id}] (Graph|1) 檢測到NSFW關鍵詞，已加載戰術指令模組: {module_name}")
        else:
            logger.warning(f"[{user_id}] (Graph|1) 檢測到NSFW關鍵詞但未能加載模組: {module_name}")

    return {"active_action_module_content": active_module_content}
# 函式：[全新] 確定性視角與模組節點 (v2.0 - 健壯檢測與模組加載)

# 函式：記憶檢索節點 (v5.0 - 返璞歸真)
# 更新紀錄:
# v5.0 (2025-09-28): [架構重構] 根據“誘餌與酬載”策略的實施，此節點的職責回歸到最簡單的模式：直接使用原始使用者輸入調用 RAG 流程。所有繞過審查的複雜邏輯都已下沉到 `ai_core` 的輔助函式和鏈的定義中。
async def retrieve_memories_node(state: ConversationGraphState) -> Dict:
    """[2] 專用記憶檢索節點，執行RAG操作。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|2) Node: retrieve_memories -> 正在基於原始查詢 '{user_input[:30]}...' 檢索相關長期記憶...")
    rag_context_str = await ai_core.retrieve_and_summarize_memories(user_input)
    return {"rag_context": rag_context_str}
# 函式：記憶檢索節點 (v5.0 - 返璞歸真)

# 函式：專用LORE查詢節點 (v6.0 - 返璞歸真)
# 更新紀錄:
# v6.0 (2025-09-28): [架構重構] 根據“誘餌與酬載”策略的實施，此節點回歸到直接使用原始使用者輸入來提取實體。
async def query_lore_node(state: ConversationGraphState) -> Dict:
    """[3] 專用LORE查詢節點，從資料庫獲取與當前輸入和【整個場景】相關的所有【非主角】LORE對象。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|3) Node: query_lore -> 正在執行【上下文優先】的LORE查詢...")

    if not ai_core.profile:
        logger.error(f"[{user_id}] (LORE Querier) ai_core.profile 未加載，無法查詢LORE。")
        return {"raw_lore_objects": []}

    gs = ai_core.profile.game_state
    effective_location_path: List[str]
    if gs.viewing_mode == 'remote' and gs.remote_target_path:
        effective_location_path = gs.remote_target_path
    else:
        effective_location_path = gs.location_path
    
    logger.info(f"[{user_id}] (LORE Querier) 已鎖定有效場景: {' > '.join(effective_location_path)}")

    lores_in_scene = await lore_book.get_lores_by_category_and_filter(
        user_id, 'npc_profile', lambda c: c.get('location_path') == effective_location_path
    )
    logger.info(f"[{user_id}] (LORE Querier) 在有效場景中找到 {len(lores_in_scene)} 位常駐NPC。")

    is_remote = gs.viewing_mode == 'remote'
    lores_from_input = await ai_core._query_lore_from_entities(user_input, is_remote_scene=is_remote)
    logger.info(f"[{user_id}] (LORE Querier) 從使用者輸入中提取並查詢到 {len(lores_from_input)} 條相關LORE。")

    final_lores_map = {lore.key: lore for lore in lores_in_scene}
    for lore in lores_from_input:
        if lore.key not in final_lores_map:
            final_lores_map[lore.key] = lore
            
    protected_names = {
        ai_core.profile.user_profile.name.lower(),
        ai_core.profile.ai_profile.name.lower()
    }
    
    filtered_lores_list = [lore for lore in final_lores_map.values() if lore.content.get('name', '').lower() not in protected_names]
    
    logger.info(f"[{user_id}] (LORE Querier) 經過上下文優先合併與過濾後，共鎖定 {len(filtered_lores_list)} 條LORE作為本回合上下文。")
    return {"raw_lore_objects": filtered_lores_list}
# 函式：專用LORE查詢節點 (v6.0 - 返璞歸真)

# 函式：專用上下文組裝節點 (v2.0 - 數據源修正)
# 更新紀錄:
# v2.0 (2025-09-29): [災難性BUG修復] 根據 KeyError Log，徹底移除了此節點對已被廢除的 `intent_classification` 狀態的依賴。現在，它直接從 `ai_core.profile.game_state.viewing_mode` 這個唯一、可靠的真實數據源來判斷當前是否為遠程場景，從而解決了因數據流斷裂導致的崩潰。
async def assemble_context_node(state: ConversationGraphState) -> Dict:
    """[4] 專用上下文組裝節點，將原始LORE格式化為LLM可讀的字符串，並透傳原始LORE對象。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    raw_lore = state['raw_lore_objects']
    
    viewing_mode = ai_core.profile.game_state.viewing_mode if ai_core.profile else 'local'
    is_remote = viewing_mode == 'remote'
    
    logger.info(f"[{user_id}] (Graph|4) Node: assemble_context -> 正在基於視角 '{viewing_mode}' 組裝最終上下文簡報...")
    
    structured_context = ai_core._assemble_context_from_lore(raw_lore, is_remote_scene=is_remote)
    
    return {
        "structured_context": structured_context,
        "raw_lore_objects": raw_lore 
    }
# 函式：專用上下文組裝節點 (v2.0 - 數據源修正)

# 函式：LORE擴展決策節點 (v6.0 - 返璞歸真)
# 更新紀錄:
# v6.0 (2025-09-28): [架構重構] 根據“誘餌與酬載”策略的實施，此節點回歸到直接使用原始使用者輸入進行決策。
async def expansion_decision_node(state: ConversationGraphState) -> Dict:
    """[5] LORE擴展決策節點，基於場景中是否已有合適角色來做決定。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    raw_lore_objects = state.get('raw_lore_objects', [])
    logger.info(f"[{user_id}] (Graph|5) Node: expansion_decision -> 正在基於語意匹配，判斷是否擴展...")
    
    lore_for_decision_making = [lore.content for lore in raw_lore_objects if lore.category == 'npc_profile']
    lore_json_str = json.dumps(lore_for_decision_making, ensure_ascii=False, indent=2)
    
    examples_str = """
- **情境 1**: 
    - 現有角色JSON: `[{"name": "海妖吟", "description": "一位販賣活魚的女性性神教徒..."}]`
    - 劇本酬載: `繼續描述那個賣魚的女人`
    - **你的決策**: `should_expand: false`
- **情境 2**:
    - 現有角色JSON: `[{"name": "海妖吟", "description": "一位女性性神教徒..."}]`
    - 劇本酬載: `這時一個衛兵走了過來`
    - **你的決策**: `should_expand: true`
"""

    decision_chain = ai_core.get_expansion_decision_chain()
    decision = await ai_core.ainvoke_with_rotation(
        decision_chain, 
        {"user_payload": user_input, "existing_characters_json": lore_json_str, "examples": examples_str},
        retry_strategy='euphemize'
    )

    if not decision:
        decision = ExpansionDecision(should_expand=False, reasoning="安全備援：決策鏈失敗。")
    
    logger.info(f"[{user_id}] (Graph|5) LORE擴展決策: {decision.should_expand}。理由: {decision.reasoning}")
    return {"expansion_decision": decision}
# 函式：LORE擴展決策節點 (v6.0 - 返璞歸真)

# 函式：專用的LORE擴展執行節點 (v10.0 - 返璞歸真)
# 更新紀錄:
# v10.0 (2025-09-28): [架構重構] 根據“誘餌與酬載”策略的實施，此節點回歸到直接使用原始使用者輸入進行選角。
async def lore_expansion_node(state: ConversationGraphState) -> Dict:
    """[6A] 專用的LORE擴展執行節點，執行選角，錨定場景，並將所有角色綁定為規劃主體。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    existing_lores = state.get('raw_lore_objects', [])
    
    logger.info(f"[{user_id}] (Graph|6A) Node: lore_expansion -> 正在執行場景選角與LORE擴展...")
    
    if not ai_core.profile:
        return {}

    gs = ai_core.profile.game_state
    effective_location_path: List[str]
    if gs.viewing_mode == 'remote' and gs.remote_target_path:
        effective_location_path = gs.remote_target_path
    else:
        effective_location_path = gs.location_path
        
    game_context_for_casting = json.dumps(state.get('structured_context', {}), ensure_ascii=False, indent=2)

    cast_result = await ai_core.ainvoke_with_rotation(
        ai_core.get_scene_casting_chain(),
        {"world_settings": ai_core.profile.world_settings or "", "current_location_path": effective_location_path,
         "game_context": game_context_for_casting, "dialogue_payload": user_input},
        retry_strategy='euphemize'
    )
    
    if cast_result and cast_result.implied_location:
        location_info = cast_result.implied_location
        base_path = [gs.location_path[0]] if gs.location_path else ["未知區域"]
        new_location_path = base_path + [location_info.name]
        lore_key = " > ".join(new_location_path)
        
        await lore_book.add_or_update_lore(user_id, 'location_info', lore_key, location_info.model_dump())
        gs.viewing_mode = 'remote'
        gs.remote_target_path = new_location_path
        await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})

    planning_subjects = [lore.content for lore in existing_lores if lore.category == 'npc_profile']
    
    if cast_result and (cast_result.newly_created_npcs or cast_result.supporting_cast):
        created_names = await ai_core._add_cast_to_scene(cast_result)
        if created_names:
            newly_created_lores = await lore_book.get_lores_by_category_and_filter(user_id, 'npc_profile', lambda c: c.get('name') in created_names)
            if newly_created_lores:
                planning_subjects.extend([lore.content for lore in newly_created_lores])
    
    return {"planning_subjects": planning_subjects}
# 函式：專用的LORE擴展執行節點 (v10.0 - 返璞歸真)




# 函式：NSFW 計畫潤色節點 (v1.0 - 備援創意生成)
# 更新紀錄:
# v1.0 (2025-09-29): [重大架構升級] 創建此全新的潤色節點。它接收一個由確定性節點生成的、缺乏創意的模板計畫，然後呼叫專門的潤色鏈，為其增加豐富的對話、呻吟和細節，從而在保證穩定性的前提下為劇情重新注入創意。
async def nsfw_refinement_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[NSFW Path] 接收模板計畫並呼叫 LLM 進行創意潤色。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    template_plan = state['turn_plan'] # 從上一步獲取模板計畫
    
    logger.info(f"[{user_id}] (Graph|NSFW Refinement) Node: nsfw_refinement -> 正在對模板計畫進行創意潤色...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(execution_rejection_reason="錯誤：AI profile 未加載，無法潤色。")}
    
    if not template_plan or template_plan.execution_rejection_reason:
        logger.warning(f"[{user_id}] (Refiner) 傳入的模板計畫為空或包含拒絕理由，跳過潤色。")
        return {"turn_plan": template_plan}

    gs = ai_core.profile.game_state
    chat_history_str = _get_formatted_chat_history(ai_core, user_id)
    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': state.get('structured_context', {}).get('possessions_context', ''),
        'quests_context': state.get('structured_context', {}).get('quests_context', ''),
        'location_context': state.get('structured_context', {}).get('location_context', ''),
        'npc_context': "(已棄用)",
        'relevant_npc_context': "(已棄用)",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': gs.viewing_mode,
        'remote_target_path_str': " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定",
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)

    refinement_chain = ai_core.get_nsfw_refinement_chain()
    
    action_prompt = state.get('active_action_module_content') or "（本回合無特定戰術指令）"

    refined_plan = await ai_core.ainvoke_with_rotation(
        refinement_chain,
        {
            "one_instruction": ai_core.profile.one_instruction,
            "action_sexual_content_prompt": action_prompt,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str,
            "turn_plan_json": template_plan.model_dump_json(indent=2)
        },
        retry_strategy='force'
    )
    
    if not refined_plan:
        logger.warning(f"[{user_id}] (Refiner) 計畫潤色鏈失敗，將回退到使用原始的、未經潤色的模板計畫。")
        return {"turn_plan": template_plan}
        
    return {"turn_plan": refined_plan}
# 函式：NSFW 計畫潤色節點 (v1.0 - 備援創意生成)




# 函式：統一規劃節點 (v3.0 - 混合創意模式)
# 更新紀錄:
# v3.0 (2025-09-29): [重大架構重構] 此節點的邏輯被徹底重寫，以支持“備援創意生成”管線。
#    - 對於 SFW 或遠程場景，它會直接調用 LLM 進行【完全創意規劃】。
#    - 對於【本地 NSFW】場景，它會【確定性地】生成一個無創意的【模板計畫骨架】，並為其行動添加 `template_id`，以便下游路由到新的潤色節點。
# v2.0 (2025-09-29): [架構重構] 節點的指令源被修改為動態讀取。
async def unified_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7] 單一硬化管線的核心規劃器，採用混合創意模式處理所有場景。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|7) Node: unified_planning -> 正在啟動統一規劃器...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(execution_rejection_reason="錯誤：AI profile 未加載，無法規劃。")}

    gs = ai_core.profile.game_state
    viewing_mode = gs.viewing_mode
    action_prompt = state.get('active_action_module_content')

    # [v3.0 核心邏輯] 判斷是否進入確定性模板路徑
    # 條件：本地模式 且 已加載性愛模組
    if viewing_mode == 'local' and action_prompt:
        logger.info(f"[{user_id}] (Planner)檢測到本地NSFW指令，進入【確定性模板骨架】生成模式。")
        
        planning_subjects_raw = state.get('planning_subjects', [])
        subjects_map = {p['name']: p for p in planning_subjects_raw}
        user_char = ai_core.profile.user_profile.model_dump()
        ai_char = ai_core.profile.ai_profile.model_dump()
        subjects_map[user_char['name']] = user_char
        subjects_map[ai_char['name']] = ai_char
        planning_subjects = list(subjects_map.values())

        # --- 確定性模板生成邏輯 ---
        templates = {
            "FUCK_TEMPLATE": {
                "thought": "本地NSFW指令觸發了確定性模板生成。這是一個基礎的性交動作骨架，將交由下游潤色。",
                "character_actions": [
                    {"character_name": "{角色A}", "reasoning": "響應指令，開始執行核心性愛動作。", "action_description": "{角色A} 將自己的[MALE_GENITALIA]對準 {角色B} 的[FEMALE_GENITALIA]口。", "template_id": "FUCK_TEMPLATE_STEP1"},
                    {"character_name": "{角色B}", "reasoning": "配合對方的動作，引導插入。", "action_description": "{角色B} 挺起腰肢，引導著 {角色A} 的[MALE_GENITALIA]完全插入自己的體內。", "template_id": "FUCK_TEMPLATE_STEP2"},
                    {"character_name": "{角色A}", "reasoning": "開始核心的性交動作。", "action_description": "{角色A} 開始在 {角色B} 溫暖濕滑的[FEMALE_GENITALIA]中用力抽插。", "template_id": "FUCK_TEMPLATE_STEP3"}
                ]
            }
        }
        selected_template_key = None
        if any(kw in user_input for kw in ["幹", "操", "性交", "插入"]):
            selected_template_key = "FUCK_TEMPLATE"
        
        if not selected_template_key:
            return {"turn_plan": TurnPlan(thought="本地NSFW指令未能匹配到任何確定性模板，將直接交由潤色節點進行純創意生成。")}

        # 簡易的角色分配邏輯
        actor_a = ai_core.profile.user_profile.name
        actor_b = ai_core.profile.ai_profile.name
        
        template_str = json.dumps(templates[selected_template_key])
        filled_str = template_str.replace("{角色A}", actor_a).replace("{角色B}", actor_b)
        
        try:
            plan = TurnPlan.model_validate(json.loads(filled_str))
            logger.info(f"[{user_id}] (Planner)已成功生成並填充確定性模板: {selected_template_key}")
            return {"turn_plan": plan}
        except Exception as e:
            return {"turn_plan": TurnPlan(execution_rejection_reason=f"模板填充失敗: {e}")}

    # --- 如果不是本地NSFW，則進入標準的創意規劃流程 ---
    logger.info(f"[{user_id}] (Planner)檢測到 SFW 或遠程場景，進入【完全創意規劃】模式。")
    
    planning_subjects_raw = state.get('planning_subjects', [])
    subjects_map = {p['name']: p for p in planning_subjects_raw}
    if viewing_mode == 'local':
        user_char = ai_core.profile.user_profile.model_dump()
        ai_char = ai_core.profile.ai_profile.model_dump()
        subjects_map[user_char['name']] = user_char
        subjects_map[ai_char['name']] = ai_char
    final_planning_subjects = list(subjects_map.values())
    planning_subjects_json = json.dumps(final_planning_subjects, ensure_ascii=False, indent=2)

    chat_history_str = _get_formatted_chat_history(ai_core, user_id)
    remote_target_path_str = " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定"
    
    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': state.get('structured_context', {}).get('possessions_context', '(遠程模式無此資訊)'),
        'quests_context': state.get('structured_context', {}).get('quests_context', '(遠程模式無此資訊)'),
        'location_context': state.get('structured_context', {}).get('location_context', ''),
        'npc_context': "(已棄用)", 'relevant_npc_context': "(已棄用)",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': viewing_mode, 'remote_target_path_str': remote_target_path_str,
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)

    planner_chain = ai_core.get_remote_nsfw_planning_chain() if viewing_mode == 'remote' else ai_core.get_nsfw_planning_chain()
    
    chain_input = {
        "one_instruction": ai_core.profile.one_instruction,
        "action_sexual_content_prompt": action_prompt or "（本回合無特定戰術指令）",
        "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
        "world_snapshot": world_snapshot,
        "chat_history": chat_history_str,
        "planning_subjects_json": planning_subjects_json,
        "user_input": user_input,
    }
    if viewing_mode == 'remote':
        chain_input["target_location_path_str"] = remote_target_path_str

    plan = await ai_core.ainvoke_with_rotation(planner_chain, chain_input, retry_strategy='force')
    if not plan:
        plan = TurnPlan(execution_rejection_reason="安全備援：統一規劃鏈最終失敗。")
    return {"turn_plan": plan}
# 函式：統一規劃節點 (v3.0 - 混合創意模式)

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
        results_summary = f"系統事件：工具執行時發生嚴重錯誤: {e}"
    finally:
        tool_context.set_context(None, None)
    
    return {"tool_results": results_summary}

async def narrative_rendering_node(state: ConversationGraphState) -> Dict[str, str]:
    """[9] 統一的敘事渲染節點，將行動計劃轉化為小說文本。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph|9) Node: narrative_rendering -> 正在將最終行動計劃渲染為小說...")

    if not turn_plan:
        return {"llm_response": "（系統錯誤：未能生成有效的行動計劃。）"}
        
    action_prompt = state.get('active_action_module_content') or "（本回合無特定戰術指令）"
    chain_input = {
        "one_instruction": ai_core.profile.one_instruction if ai_core.profile else "預設系統指令",
        "response_style_prompt": ai_core.profile.response_style_prompt if ai_core.profile else "預設風格",
        "action_sexual_content_prompt": action_prompt,
        "turn_plan": turn_plan
    }
        
    narrative_text = await ai_core.ainvoke_with_rotation(
        ai_core.get_narrative_chain(), chain_input, retry_strategy='force'
    )
    if not narrative_text:
        narrative_text = "（AI 在將最終計劃轉化為故事時遭遇了無法恢復的內容安全限制。）"
    return {"llm_response": narrative_text}

# 函式：統一的輸出淨化與解碼節點 (v5.0 - 動態解碼)
# 更新紀錄:
# v5.0 (2025-09-29): [災難性BUG修復] 為了從根本上解決代碼替換失敗的問題，此節點的解碼邏輯被徹底重構。它不再依賴於訪問 `ai_core`，而是直接從圖狀態 `state['active_action_module_content']` 中讀取當前回合激活的指令模組，並從中動態解析出正確的「密碼本」，確保了解碼規則與生成規則的絕對同步。
async def validate_and_rewrite_node(state: ConversationGraphState) -> Dict:
    """[10] 統一的輸出淨化與解碼節點，並執行最終的代碼還原。"""
    user_id = state['user_id']
    initial_response = state['llm_response']
    logger.info(f"[{user_id}] (Graph|10) Node: validate_and_rewrite -> 正在對 LLM 原始輸出進行淨化與最終解碼...")
    
    if not initial_response or not initial_response.strip():
        return {"final_output": "（...）"}
    
    clean_response = initial_response.strip()
    
    final_response = clean_response
    action_prompt = state.get('active_action_module_content')
    
    if action_prompt:
        try:
            replacement_map = {}
            matches = re.findall(r'其\s*(.*?)\s*【必須且只能】被稱為\s*[:：]\s*「(.*?)」。', action_prompt)
            term_map_from_prompt = {desc.strip(): term.strip() for desc, term in matches}
            
            code_to_desc = {
                "[MALE_GENITALIA]": "性器官", "[FEMALE_GENITALIA]": "性器官",
                "[MALE_FLUID]": "精液", "[FEMALE_FLUID]": "產生的愛液",
                "[CLITORIS]": "陰蒂", "[ANUS]": "肛門/後庭"
            }
            
            all_terms = re.findall(r'「(.*?)」', action_prompt)
            male_term_found = False
            for term in all_terms:
                if "MALE_GENITALIA" in action_prompt.split(f"「{term}」")[0] and not male_term_found:
                     replacement_map["[MALE_GENITALIA]"] = term; male_term_found = True
                elif "FEMALE_GENITALIA" in action_prompt.split(f"「{term}」")[0]:
                     replacement_map["[FEMALE_GENITALIA]"] = term

            for code, desc in code_to_desc.items():
                if desc in term_map_from_prompt and code not in replacement_map:
                     replacement_map[code] = term_map_from_prompt[desc]
            
            if "[ANUS]" not in replacement_map: replacement_map["[ANUS]"] = "後庭"

            if replacement_map:
                logger.info(f"[{user_id}] (Decoder) 已成功從 state 動態解析替換規則: {replacement_map}")
                for code, word in replacement_map.items():
                    final_response = final_response.replace(code, word)
            else:
                 logger.warning(f"[{user_id}] (Decoder) 未能從 state 的指令模組中解析出任何替換規則。")

        except Exception as e:
            logger.error(f"[{user_id}] (Decoder) 從 state 解析替換規則時發生嚴重錯誤: {e}。", exc_info=True)
    else:
        logger.info(f"[{user_id}] (Decoder) 本回合無激活的戰術指令模組，跳過代碼替換。")
        
    return {"final_output": final_response}
# 函式：統一的輸出淨化與解碼節點 (v5.0 - 動態解碼)

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
        tasks = [ai_core._generate_and_save_personal_memory(last_interaction_text)]
        if ai_core.vector_store:
            tasks.append(asyncio.to_thread(ai_core.vector_store.add_texts, [last_interaction_text], metadatas=[{"source": "history"}]))
        
        async def save_to_sql():
            from .database import AsyncSessionLocal, MemoryData
            import time
            async with AsyncSessionLocal() as session:
                session.add(MemoryData(user_id=user_id, content=last_interaction_text, timestamp=time.time(), importance=1))
                await session.commit()
        
        tasks.append(save_to_sql())
        await asyncio.gather(*tasks, return_exceptions=True)
        
    return {}

# --- 主對話圖的路由與建構器 v41.0 ---

def route_expansion_decision(state: ConversationGraphState) -> Literal["expand_lore", "continue_to_planner"]:
    """根據LORE擴展決策，決定是否進入擴展節點。"""
    if state.get("expansion_decision") and state["expansion_decision"].should_expand:
        return "expand_lore"
    else:
        return "continue_to_planner"

# 函式：創建主回應圖 (v41.0 - 備援創意生成)
# 更新紀錄:
# v41.0 (2025-09-29): [重大架構重構] 為了在保證穩定性的前提下恢復創意，對 NSFW 路徑進行了重構。
#    1. [廢除] `nsfw_template_assembly_node` 已被廢除。
#    2. [新增] 引入了全新的 `nsfw_refinement_node`，負責對計畫骨架進行創意擴寫。
#    3. [改造] `unified_planning_node` 現在的職責是：如果是 SFW 或遠程場景，則直接調用 LLM 進行創意規劃；如果是本地 NSFW 場景，則【確定性地】生成一個模板計畫骨架。
#    4. [重連] 重新設計了路由邏輯，以支持這種新的混合創意模式。
# v40.0 (2025-09-29): [重大架構重構] 引入了“單一硬化管線”策略。
def create_main_response_graph() -> StateGraph:
    graph = StateGraph(ConversationGraphState)
    
    graph.add_node("determine_viewing_mode", determine_viewing_mode_node)
    graph.add_node("retrieve_memories", retrieve_memories_node)
    graph.add_node("query_lore", query_lore_node)
    graph.add_node("assemble_context", assemble_context_node)
    graph.add_node("expansion_decision", expansion_decision_node)
    graph.add_node("lore_expansion", lore_expansion_node)
    graph.add_node("unified_planning", unified_planning_node)
    graph.add_node("nsfw_refinement", nsfw_refinement_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("narrative_rendering", narrative_rendering_node)
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)
    
    def prepare_existing_subjects_node(state: ConversationGraphState) -> Dict:
        lore_objects = state.get('raw_lore_objects', [])
        planning_subjects = [lore.content for lore in lore_objects if lore.category == 'npc_profile']
        return {"planning_subjects": planning_subjects}
        
    graph.add_node("prepare_existing_subjects", prepare_existing_subjects_node)

    graph.set_entry_point("determine_viewing_mode")
    graph.add_edge("determine_viewing_mode", "retrieve_memories")
    graph.add_edge("retrieve_memories", "query_lore")
    graph.add_edge("query_lore", "assemble_context")
    graph.add_edge("assemble_context", "expansion_decision")
    
    graph.add_conditional_edges(
        "expansion_decision", route_expansion_decision,
        {"expand_lore": "lore_expansion", "continue_to_planner": "prepare_existing_subjects"}
    )
    
    graph.add_edge("lore_expansion", "unified_planning")
    graph.add_edge("prepare_existing_subjects", "unified_planning")
    
    def route_after_planning(state: ConversationGraphState) -> str:
        turn_plan = state.get('turn_plan')
        is_templated_plan = False
        if turn_plan and turn_plan.character_actions:
            if any(action.template_id for action in turn_plan.character_actions):
                is_templated_plan = True

        if is_templated_plan:
            return "needs_refinement"
        else:
            return "execute_directly"

    graph.add_conditional_edges(
        "unified_planning", route_after_planning,
        {"needs_refinement": "nsfw_refinement", "execute_directly": "tool_execution"}
    )
    
    graph.add_edge("nsfw_refinement", "tool_execution")
    graph.add_edge("tool_execution", "narrative_rendering")
    graph.add_edge("narrative_rendering", "validate_and_rewrite")
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", END)
    
    return graph.compile()
# 函式：創建主回應圖 (v41.0 - 備援創意生成)

# --- 設定圖 (Setup Graph) 的節點與建構器 (完整版) ---

async def process_canon_node(state: SetupGraphState) -> Dict:
    ai_core = state['ai_core']
    canon_text = state['canon_text']
    if canon_text:
        await ai_core.add_canon_to_vector_store(canon_text)
        await ai_core.parse_and_create_lore_from_canon(None, canon_text, is_setup_flow=True)
    return {}

async def complete_profiles_node(state: SetupGraphState) -> Dict:
    ai_core = state['ai_core']
    completion_chain = ai_core.get_profile_completion_chain()
    if not ai_core.profile: return {}
    
    tasks = [
        ai_core.ainvoke_with_rotation(completion_chain, {"profile_json": ai_core.profile.user_profile.model_dump_json()}, retry_strategy='euphemize'),
        ai_core.ainvoke_with_rotation(completion_chain, {"profile_json": ai_core.profile.ai_profile.model_dump_json()}, retry_strategy='euphemize')
    ]
    completed_user_profile, completed_ai_profile = await asyncio.gather(*tasks)

    update_payload = {}
    if completed_user_profile: update_payload['user_profile'] = completed_user_profile.model_dump()
    if completed_ai_profile: update_payload['ai_profile'] = completed_ai_profile.model_dump()
    if update_payload: await ai_core.update_and_persist_profile(update_payload)
    return {}

async def world_genesis_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    if not ai_core.profile: raise Exception("AI Profile not loaded.")

    genesis_chain = ai_core.get_world_genesis_chain()
    genesis_result = await ai_core.ainvoke_with_rotation(genesis_chain, {"world_settings": ai_core.profile.world_settings, "username": ai_core.profile.user_profile.name, "ai_name": ai_core.profile.ai_profile.name}, retry_strategy='force')
    if not genesis_result: raise Exception("World genesis chain returned empty.")
        
    gs = ai_core.profile.game_state
    gs.location_path = genesis_result.location_path
    await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})
    
    await lore_book.add_or_update_lore(user_id, 'location_info', " > ".join(genesis_result.location_path), genesis_result.location_info.model_dump())
    for npc in genesis_result.initial_npcs:
        npc_key = " > ".join(genesis_result.location_path) + f" > {npc.name}"
        await lore_book.add_or_update_lore(user_id, 'npc_profile', npc_key, npc.model_dump())
    return {"genesis_result": genesis_result}

async def generate_opening_scene_node(state: SetupGraphState) -> Dict:
    ai_core = state['ai_core']
    opening_scene = await ai_core.generate_opening_scene()
    if not opening_scene or not opening_scene.strip():
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
# 函式：創建設定圖
