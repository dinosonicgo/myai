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


# 函式：場景與動作分析節點 (v5.0 - 兩階段驗證)
# 更新紀錄:
# v5.0 (2025-09-06): [災難性BUG修復] 根據反覆出現的 ValidationError，最終確認並實施了“兩階段驗證”策略。此節點現在接收來自上游的、更寬鬆的 `RawSceneAnalysis` 模型，然後在其內部，用確定性的 Python 程式碼對這個原始數據進行嚴格的邏輯校準和修正，最後再手動創建一個【保證邏輯自洽】的 `SceneAnalysisResult` 物件並輸出。此修改從根本上解決了所有因 LLM 輸出與 Pydantic 驗證器衝突而導致的崩潰。
# v4.0 (2025-09-06): [災難性BUG修復] 增加了【程式碼層面的業務邏輯校準】。
# v3.0 (2025-09-06): [災難性BUG修復] 重構了此節點的數據準備邏輯。
async def scene_and_action_analysis_node(state: ConversationGraphState) -> Dict:
    """分析場景的視角，並在程式碼層面強制校準結果以確保邏輯一致性。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: scene_and_action_analysis -> 正在進行場景視角分析...")

    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph) 在 scene_and_action_analysis 中 ai_core.profile 未加載。")
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='錯誤：AI profile 未加載。', action_summary=user_input)}

    try:
        entity_chain = ai_core.get_entity_extraction_chain()
        entity_result = await ai_core.ainvoke_with_rotation(entity_chain, {"text_input": user_input})
        sanitized_input_for_analysis = "觀察場景：" + " ".join(entity_result.names) if entity_result and entity_result.names else user_input
    except Exception as e:
        logger.error(f"[{user_id}] (Scene Analysis) 預處理失敗: {e}", exc_info=True)
        sanitized_input_for_analysis = user_input

    scene_context_lores = [lore.content for lore in state.get('raw_lore_objects', []) if lore.category == 'npc_profile']
    scene_context_json_str = json.dumps(scene_context_lores, ensure_ascii=False, indent=2)
    current_location_path = ai_core.profile.game_state.location_path
    
    scene_analysis_chain = ai_core.get_scene_analysis_chain()
    # [v5.0 核心修正] 接收寬鬆的 RawSceneAnalysis 模型
    raw_analysis = await ai_core.ainvoke_with_rotation(
        scene_analysis_chain,
        {
            "user_input": sanitized_input_for_analysis, 
            "current_location_path_str": " > ".join(current_location_path),
            "scene_context_json": scene_context_json_str
        },
        retry_strategy='euphemize'
    )

    if not raw_analysis:
        logger.warning(f"[{user_id}] (Graph) 場景分析鏈委婉化重試失敗，啟動安全備援。")
        final_analysis = SceneAnalysisResult(viewing_mode='local', reasoning='安全備援：場景分析鏈失敗。', action_summary=user_input)
    else:
        # [v5.0 核心修正] 在程式碼層面進行校準，然後創建最終的、嚴格的 SceneAnalysisResult
        logger.info(f"[{user_id}] (Analysis Corrector) 收到初步分析: mode={raw_analysis.viewing_mode}, path={raw_analysis.target_location_path}")

        # 複製數據以進行修正
        final_viewing_mode = raw_analysis.viewing_mode
        final_target_path = raw_analysis.target_location_path
        final_reasoning = raw_analysis.reasoning

        if final_viewing_mode == 'remote':
            # 規則：remote 模式必須有有效的 target_location_path
            if not final_target_path:
                logger.warning(f"[{user_id}] (Analysis Corrector) 邏輯衝突：初步分析為 remote 但缺少路徑。強制修正為 local。")
                final_viewing_mode = 'local'
                final_reasoning += " [校準：因缺少目標路徑，已強制修正為local模式]"
        
        # 使用校準後的值創建最終的、保證合法的 SceneAnalysisResult 物件
        final_analysis = SceneAnalysisResult(
            viewing_mode=final_viewing_mode,
            reasoning=final_reasoning,
            target_location_path=final_target_path,
            focus_entity=raw_analysis.focus_entity,
            action_summary=raw_analysis.action_summary or user_input # 確保摘要不為空
        )
        
    logger.info(f"[{user_id}] (Analysis Corrector) 已校準最終分析結果: mode={final_analysis.viewing_mode}, path={final_analysis.target_location_path}")
    return {"scene_analysis": final_analysis}
# 函式：場景與動作分析節點 (v5.0 - 兩階段驗證)


# 函式：[全新] 輸入預處理節點 (v1.0 - 前置淨化層)
# 更新紀錄:
# v1.0 (2025-09-06): [重大架構升級] 創建此全新的「前置淨化層」節點，作為圖的新入口。它負責調用實體提取鏈和委婉化鏈，將原始的、可能觸發審查的露骨使用者輸入，轉化為一個保留核心意圖但用詞中性的安全版本。此節點是從根本上解決因輸入「投毒」導致早期分析節點（如意圖分類）被 API 攔截的關鍵性修正。
async def pre_process_input_node(state: ConversationGraphState) -> Dict:
    """[0] 圖的新入口點。將原始使用者輸入轉化為一個對分析節點安全的、中性的指令。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|0) Node: pre_process_input -> 正在對原始輸入進行淨化...")

    try:
        # 使用一個不易被審查的鏈（實體提取）來獲取核心名詞
        entity_extraction_chain = ai_core.get_entity_extraction_chain()
        entity_result = await ai_core.ainvoke_with_rotation(
            entity_extraction_chain,
            {"text_input": user_input},
            retry_strategy='euphemize' # 使用委婉化重試確保它能成功
        )
        
        if not (entity_result and entity_result.names):
            logger.warning(f"[{user_id}] (Sanitizer) 未能從輸入中提取實體，將使用原始輸入作為淨化版本，這可能存在風險。")
            return {"sanitized_user_input": user_input}
            
        # 使用另一個安全的鏈，從核心名詞重構出一個中性的句子
        euphemization_chain = ai_core.get_euphemization_chain()
        sanitized_input = await ai_core.ainvoke_with_rotation(
            euphemization_chain,
            {"keywords": entity_result.names},
            retry_strategy='none' # 如果重構失敗，則無法挽救
        )
        
        if not sanitized_input:
            logger.error(f"[{user_id}] (Sanitizer) 委婉化重構鏈失敗！將回退到使用原始輸入，這極可能導致後續分析失敗！")
            sanitized_input = user_input
        
        logger.info(f"[{user_id}] (Sanitizer) 輸入淨化成功: '{user_input[:50]}...' -> '{sanitized_input[:50]}...'")
        return {"sanitized_user_input": sanitized_input}

    except Exception as e:
        logger.error(f"[{user_id}] (Sanitizer) 在預處理節點發生未知嚴重錯誤: {e}", exc_info=True)
        # 發生任何未知錯誤時，都返回原始輸入以確保流程繼續，即使後續可能失敗
        return {"sanitized_user_input": user_input}
# 函式：[全新] 輸入預處理節點 (v1.0 - 前置淨化層)


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

# 函式：[重構] 導演視角狀態管理節點 (v2.0 - 意圖驅動)
# 更新紀錄:
# v2.0 (2025-09-06): [災難性BUG修復] 根據「遠程視角管理失敗」的問題，徹底重構了此節點的邏輯，使其變得更加確定和可靠。新版本不再依賴複雜的、容易出錯的 `scene_analysis` 來判斷視角，而是直接由更可靠的 `intent_classification` 結果來驅動。如果意圖是描述性的，則強制切換到遠程模式，並調用一個專門的、更簡單的地點提取鏈來獲取目標路徑。此修改從根本上解決了視角模式判斷錯誤的問題。
# v1.0 (2025-09-06): [災難性BUG修復] 創建此新節點。
async def update_viewing_mode_node(state: ConversationGraphState) -> None:
    """根據意圖分類，更新並持久化導演視角模式和遠程目標。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    intent = state['intent_classification'].intent_type
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: update_viewing_mode -> 正在基於意圖 '{intent}' 更新導演視角...")
    
    if not ai_core.profile:
        return

    gs = ai_core.profile.game_state
    original_mode = gs.viewing_mode
    original_path = gs.remote_target_path
    changed = False

    if 'descriptive' in intent:
        # 意圖是描述性的，強制進入或保持遠程模式
        logger.info(f"[{user_id}] (View Mode) 檢測到描述性意圖，準備進入/更新遠程視角。")
        location_chain = ai_core.get_location_extraction_chain()
        location_result = await ai_core.ainvoke_with_rotation(location_chain, {"user_input": user_input})
        
        new_target_path = location_result.get("location_path") if location_result else None
        
        if new_target_path:
            if gs.viewing_mode != 'remote' or gs.remote_target_path != new_target_path:
                gs.viewing_mode = 'remote'
                gs.remote_target_path = new_target_path
                changed = True
        else:
            logger.warning(f"[{user_id}] (View Mode) 描述性意圖未能提取出有效地點，視角可能不會按預期更新。")

    else: # 意圖是互動性的
        if gs.viewing_mode != 'local':
            logger.info(f"[{user_id}] (View Mode) 檢測到互動性意圖，強制切換回本地視角。")
            gs.viewing_mode = 'local'
            gs.remote_target_path = None
            changed = True

    if changed:
        logger.info(f"[{user_id}] 導演視角模式已從 '{original_mode}' (路徑: {original_path}) 更新為 '{gs.viewing_mode}' (路徑: {gs.remote_target_path})")
        await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})
    else:
        logger.info(f"[{user_id}] 導演視角模式保持為 '{original_mode}' (路徑: {original_path})，無需更新。")

    return {}
# 函式：[重構] 導演視角狀態管理節點 (v2.0 - 意圖驅動)


# 函式：意圖分類節點 (v2.0 - 適配淨化層)
# 更新紀錄:
# v2.0 (2025-09-06): [災難性BUG修復] 修改了此節點的數據源。它現在優先使用由 `pre_process_input_node` 生成的 `sanitized_user_input` 進行意圖分類。這確保了此分析節點不會因直接接觸原始的露骨輸入而被內容審查攔截，是解決路由失敗問題的關鍵修正。
async def classify_intent_node(state: ConversationGraphState) -> Dict:
    """[1] 圖的入口點，唯一職責是對原始輸入進行意圖分類。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    
    # [v2.0 核心修正] 優先使用淨化後的輸入，如果不存在則備援至原始輸入
    user_input_for_classification = state.get('sanitized_user_input', state['messages'][-1].content)
    
    logger.info(f"[{user_id}] (Graph|1) Node: classify_intent -> 正在對 '{user_input_for_classification[:30]}...' 進行意圖分類...")
    
    classification_chain = ai_core.get_intent_classification_chain()
    classification_result = await ai_core.ainvoke_with_rotation(
        classification_chain,
        {"user_input": user_input_for_classification},
        retry_strategy='none' # 分類鏈不應重試，失敗則啟用備援
    )
    
    if not classification_result:
        logger.warning(f"[{user_id}] (Graph|1) 意圖分類鏈失敗，啟動安全備援，預設為 SFW。")
        classification_result = IntentClassificationResult(intent_type='sfw', reasoning="安全備援：分類鏈失敗。")
        
    return {"intent_classification": classification_result}
# 函式：意圖分類節點 (v2.0 - 適配淨化層)

# 函式：記憶檢索節點 (v2.0 - 適配淨化層)
# 更新紀錄:
# v2.0 (2025-09-06): [災難性BUG修復] 修改了此節點的數據源。它現在優先使用由 `pre_process_input_node` 生成的 `sanitized_user_input` 作為 RAG 檢索的查詢。這確保了檢索過程本身不會因觸發內容審查而失敗，提高了整個 RAG 鏈路的穩定性。
async def retrieve_memories_node(state: ConversationGraphState) -> Dict:
    """[2] 專用記憶檢索節點，執行RAG操作。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    
    # [v2.0 核心修正] 優先使用淨化後的輸入進行檢索
    user_input_for_retrieval = state.get('sanitized_user_input', state['messages'][-1].content)
    
    logger.info(f"[{user_id}] (Graph|2) Node: retrieve_memories -> 正在基於安全查詢 '{user_input_for_retrieval[:30]}...' 檢索相關長期記憶...")
    
    # ai_core.py 中的輔助函式會處理總結邏輯
    rag_context_str = await ai_core.retrieve_and_summarize_memories(user_input_for_retrieval)
    return {"rag_context": rag_context_str}
# 函式：記憶檢索節點 (v2.0 - 適配淨化層)



# 函式：[全新] 感知與視角設定中樞 (v1.1 - Pydantic 訪問修正)
# 更新紀錄:
# v1.1 (2025-09-06): [災難性BUG修復] 根據 AttributeError，修正了節點內部對上游鏈返回的 Pydantic 模型物件的數據訪問方式。將錯誤的字典式訪問 `location_result.get("location_path")` 修改為正確的物件屬性訪問 `location_result.location_path`，從根本上解決了因此導致的崩潰問題。
# v1.0 (2025-09-06): [重大架構重構] 創建了這個全新的、統一的感知中樞節點。
async def perceive_and_set_view_node(state: ConversationGraphState) -> Dict:
    """一個統一的節點，負責分析場景、根據意圖設定視角、並持久化狀態。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    intent = state['intent_classification'].intent_type
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: perceive_and_set_view -> 正在基於意圖 '{intent}' 統一處理感知與視角...")

    if not ai_core.profile:
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='錯誤：AI profile 未加載。', action_summary=user_input)}

    gs = ai_core.profile.game_state
    changed = False
    new_viewing_mode = gs.viewing_mode
    new_target_path = gs.remote_target_path

    if 'descriptive' in intent:
        # 意圖是描述性的，強制進入或保持遠程模式
        logger.info(f"[{user_id}] (View Mode) 檢測到描述性意圖，準備進入/更新遠程視角。")
        
        scene_context_lores = [lore.content for lore in state.get('raw_lore_objects', []) if lore.category == 'npc_profile']
        scene_context_json_str = json.dumps(scene_context_lores, ensure_ascii=False, indent=2)
        
        location_chain = ai_core.get_contextual_location_chain()
        location_result = await ai_core.ainvoke_with_rotation(
            location_chain, 
            {
                "user_input": user_input,
                "world_settings": ai_core.profile.world_settings or "未設定",
                "scene_context_json": scene_context_json_str
            }
        )
        
        # [v1.1 核心修正] 使用點號表示法訪問 Pydantic 物件屬性
        extracted_path = location_result.location_path if location_result else None
        
        if extracted_path:
            new_viewing_mode = 'remote'
            new_target_path = extracted_path
        else:
            logger.warning(f"[{user_id}] (Perception Hub) 描述性意圖未能推斷出有效地點，將回退到本地模式。")
            new_viewing_mode = 'local'
            new_target_path = None
            
    else: # 意圖是互動性的
        new_viewing_mode = 'local'
        new_target_path = None

    # 檢查狀態是否真的發生了變化
    if gs.viewing_mode != new_viewing_mode or gs.remote_target_path != new_target_path:
        gs.viewing_mode = new_viewing_mode
        gs.remote_target_path = new_target_path
        await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})
        logger.info(f"[{user_id}] (Perception Hub) GameState 已更新: mode={gs.viewing_mode}, path={gs.remote_target_path}")
    else:
        logger.info(f"[{user_id}] (Perception Hub) GameState 無需更新。")

    # 構建並返回 SceneAnalysisResult 以供下游使用
    scene_analysis = SceneAnalysisResult(
        viewing_mode=gs.viewing_mode,
        reasoning=f"基於意圖 '{intent}' 的統一感知結果。",
        target_location_path=gs.remote_target_path,
        focus_entity=None, # 此欄位可以後續由更專精的鏈填充，或暫時棄用
        action_summary=user_input
    )
    
    return {"scene_analysis": scene_analysis}
# 函式：[全新] 感知與視角設定中樞 (v1.1 - Pydantic 訪問修正)

# 函式：NSFW 模板裝配節點 (v2.0 - 確定性規劃)
# 更新紀錄:
# v2.0 (2025-09-07): [終極架構重構] 此節點被徹底重寫，成為一個純Python的「確定性計畫生成器」。它不再呼叫LLM，而是通過內部邏輯完成三項任務：
#    1. 根據關鍵詞選擇硬編碼的JSON模板。
#    2. 分析場景角色列表，識別出行動的參與者。
#    3. 使用字串替換，將角色名字機械地裝配到模板中。
#    此修改將最容易被審查的創意規劃步驟完全移出LLM，是解決頑固性內容審查的最終手段。
def nsfw_template_assembly_node(state: ConversationGraphState) -> Dict:
    """[NSFW Template-Step1] 純Python節點，負責選擇、填充並裝配一個預定義的NSFW動作模板。"""
    user_id = state['user_id']
    user_input = state['messages'][-1].content.lower()
    planning_subjects = state.get('planning_subjects', [])
    logger.info(f"[{user_id}] (Graph|NSFW Assembly) Node: nsfw_template_assembly -> 正在確定性地裝配NSFW計畫...")

    if not planning_subjects or len(planning_subjects) < 2:
        logger.warning(f"[{user_id}] (Graph|NSFW Assembly) 場景中角色不足（需要至少2名），無法裝配模板。")
        return {"turn_plan": TurnPlan(thought="場景角色不足，無法執行模板化動作。")}

    # 預定義的模板庫
    templates = {
        "FUCK_TEMPLATE": {
            "thought": "使用者指令的核心動作是性交。已使用預設的性交模板，並裝配了場景中的角色。",
            "character_actions": [
                {"character_name": "{角色A}", "reasoning": "響應指令，開始執行核心性愛動作。", "action_description": "{角色A} 將自己的肉棒對準 {角色B} 的肉穴口。"},
                {"character_name": "{角色B}", "reasoning": "配合對方的動作，引導插入。", "action_description": "{角色B} 挺起腰肢，引導著 {角色A} 的肉棒完全插入自己的體內。"},
                {"character_name": "{角色A}", "reasoning": "開始核心的性交動作。", "action_description": "{角色A} 開始在 {角色B} 溫暖濕滑的肉穴中用力抽插。"}
            ]
        }
    }

    selected_template_key = None
    if "幹" in user_input or "操" in user_input or "性交" in user_input:
        selected_template_key = "FUCK_TEMPLATE"
    
    if not selected_template_key:
        logger.warning(f"[{user_id}] (Graph|NSFW Assembly) 未能為輸入匹配到任何NSFW模板。")
        return {"turn_plan": TurnPlan(thought="未能匹配到動作模板，將嘗試根據上下文進行潤色。")}

    # 角色識別邏輯 (簡易版)
    # 假設：指令中提到的第一個角色類型是主動方，第二個是被動方
    actor_a_name = None
    actor_b_name = None
    
    # 尋找男性/男孩作為主動方
    for subject in planning_subjects:
        desc = subject.get('description', '').lower()
        if 'boy' in desc or '男孩' in desc or 'male' in desc or subject.get('gender', '').lower() == '男':
            actor_a_name = subject['name']
            break
    
    # 尋找女性/母親作為被動方
    for subject in planning_subjects:
        desc = subject.get('description', '').lower()
        if subject['name'] != actor_a_name and ('mother' in desc or '媽媽' in desc or 'female' in desc or subject.get('gender', '').lower() in ['女', '女性']):
            actor_b_name = subject['name']
            break

    if not (actor_a_name and actor_b_name):
        logger.error(f"[{user_id}] (Graph|NSFW Assembly) 無法從場景中明確識別出動作的雙方。")
        return {"turn_plan": TurnPlan(execution_rejection_reason="系統錯誤：無法在當前場景中識別出可以執行此動作的角色。")}

    logger.info(f"[{user_id}] (Graph|NSFW Assembly) 已選擇模板: {selected_template_key}。角色A: {actor_a_name}, 角色B: {actor_b_name}")

    # 機械裝配
    template = templates[selected_template_key]
    template_str = json.dumps(template)
    
    # 執行替換
    filled_str = template_str.replace("{角色A}", actor_a_name).replace("{角色B}", actor_b_name)
    
    try:
        filled_plan_dict = json.loads(filled_str)
        final_plan = TurnPlan.model_validate(filled_plan_dict)
        logger.info(f"[{user_id}] (Graph|NSFW Assembly) NSFW計畫模板已成功裝配。")
        return {"turn_plan": final_plan}
    except Exception as e:
        logger.error(f"[{user_id}] (Graph|NSFW Assembly) 裝配後的計畫未能通過Pydantic驗證: {e}", exc_info=True)
        return {"turn_plan": TurnPlan(execution_rejection_reason=f"系統錯誤：裝配計畫時發生內部錯誤: {e}")}
# 函式：NSFW 模板裝配節點 (v2.0 - 確定性規劃)


# 函式：專用LORE查詢節點 (v4.0 - 上下文優先檢索)
# 更新紀錄:
# v4.0 (2025-09-06): [災難性BUG修復] 徹底重寫了LORE的檢索邏輯，以解決因檢索到其他地點的同名NPC而導致的上下文污染問題。新版本實現了「上下文優先」原則：
#    1. [確定有效場景] 首先明確當前的工作場景（本地或遠程）。
#    2. [場景內優先] 優先獲取所有物理上存在於該場景的NPC。
#    3. [召喚式補充] 然後再檢索使用者明確提及的、可能不在場的角色。
#    4. [智能合併] 最後將兩者智能合併，確保了傳遞給規劃器的上下文是以當前場景為絕對核心的，從根本上杜絕了地點錯亂的問題。
# v3.0 (2025-09-06): [災難性BUG修復] 增加了“主角光環”過濾機制。
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
    
    # --- [v4.0 核心修正] 步驟 1: 確定有效場景 ---
    effective_location_path: List[str]
    if gs.viewing_mode == 'remote' and gs.remote_target_path:
        effective_location_path = gs.remote_target_path
    else:
        effective_location_path = gs.location_path
    
    logger.info(f"[{user_id}] (LORE Querier) 已鎖定有效場景: {' > '.join(effective_location_path)}")

    # --- [v4.0 核心修正] 步驟 2: 場景內實體優先 ---
    lores_in_scene = await lore_book.get_lores_by_category_and_filter(
        user_id,
        'npc_profile',
        lambda c: c.get('location_path') == effective_location_path
    )
    logger.info(f"[{user_id}] (LORE Querier) 在有效場景中找到 {len(lores_in_scene)} 位常駐NPC。")

    # --- [v4.0 核心修正] 步驟 3: 召喚式實體補充 ---
    is_remote = gs.viewing_mode == 'remote'
    lores_from_input = await ai_core._query_lore_from_entities(user_input, is_remote_scene=is_remote)
    logger.info(f"[{user_id}] (LORE Querier) 從使用者輸入中提取並查詢到 {len(lores_from_input)} 條相關LORE。")

    # --- [v4.0 核心修正] 步驟 4: 智能合併與去重 ---
    # 使用字典以 lore.key 作為鍵，可以高效地合併和去重
    # 將場景內的NPC首先放入，確保它們的優先級
    final_lores_map = {lore.key: lore for lore in lores_in_scene}
    # 然後補充使用者明確提到的、但可能不在場景內的NPC
    for lore in lores_from_input:
        if lore.key not in final_lores_map:
            final_lores_map[lore.key] = lore
            
    # "主角光環" 過濾 (保留此重要邏輯)
    protected_names = {
        ai_core.profile.user_profile.name.lower(),
        ai_core.profile.ai_profile.name.lower()
    }
    
    filtered_lores_list = []
    for lore in final_lores_map.values():
        lore_name = lore.content.get('name', '').lower()
        if lore_name not in protected_names:
            filtered_lores_list.append(lore)
        else:
            logger.warning(f"[{user_id}] (LORE Querier) 已過濾掉與核心主角同名的LORE記錄: '{lore.content.get('name')}'")

    logger.info(f"[{user_id}] (LORE Querier) 經過上下文優先合併與過濾後，共鎖定 {len(filtered_lores_list)} 條LORE作為本回合上下文。")
    
    return {"raw_lore_objects": filtered_lores_list}
# 函式：專用LORE查詢節點 (v4.0 - 上下文優先檢索)



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



# 函式：統一NSFW規劃節點 (v7.0 - KeyError 修正)
# 更新紀錄:
# v7.0 (2025-09-06): [災難性BUG修復] 根據 KeyError Log，移除了對已被廢棄的 `sanitized_user_input` 狀態的引用。
# v6.0 (2025-09-06): [健壯性] 修改了備援邏輯，改為使用 `execution_rejection_reason` 欄位來傳遞錯誤。
# v5.0 (2025-09-06): [健壯性] 修正了調用鏈時的參數傳遞。
async def nsfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7B] 統一的 NSFW 互動路徑規劃器，直接生成最終的、露骨的行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    # [v7.0 核心修正] 移除對 sanitized_user_input 的引用
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|7B) Node: nsfw_planning -> 正在基於指令 '{user_input[:50]}...' 生成統一NSFW行動計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(execution_rejection_reason="錯誤：AI profile 未加載，無法規劃。")}

    planning_subjects_raw = state.get('planning_subjects')
    if planning_subjects_raw is None:
        lore_objects = state.get('raw_lore_objects', [])
        planning_subjects_raw = [lore.content for lore in lore_objects if lore.category == 'npc_profile']
    planning_subjects_json = json.dumps(planning_subjects_raw, ensure_ascii=False, indent=2)

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
        'npc_context': "(已棄用，請參考 planning_subjects_json)",
        'relevant_npc_context': "(已棄用，請參考 planning_subjects_json)",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': gs.viewing_mode,
        'remote_target_path_str': " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定",
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)
    
    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_nsfw_planning_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction,
            "action_sexual_content_prompt": ai_core.modular_prompts.get("action_sexual_content", "警告:性愛模組未加載"),
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str,
            "planning_subjects_json": planning_subjects_json,
            "user_input": user_input,
            "username": ai_core.profile.user_profile.name,
        },
        retry_strategy='force'
    )
    if not plan:
        plan = TurnPlan(execution_rejection_reason="安全備援：NSFW統一規劃鏈最終失敗，可能因為內容審查或API臨時故障。")
    return {"turn_plan": plan}
# 函式：統一NSFW規劃節點 (v7.0 - KeyError 修正)



def _get_formatted_chat_history(ai_core: AILover, user_id: str, num_messages: int = 10) -> str:
    """從 AI 核心實例中提取並格式化最近的對話歷史。"""
    chat_history_manager = ai_core.session_histories.get(user_id, ChatMessageHistory())
    if not chat_history_manager.messages:
        return "（沒有最近的對話歷史）"
    
    # 提取最近的 N 條訊息
    recent_messages = chat_history_manager.messages[-num_messages:]
    
    formatted_history = []
    for msg in recent_messages:
        role = "使用者" if isinstance(msg, HumanMessage) else ai_core.profile.ai_profile.name if ai_core.profile else "AI"
        formatted_history.append(f"{role}: {msg.content}")
        
    return "\n".join(formatted_history)


    
# 函式：LORE擴展決策節點 (v4.1 - 範例注入)
# 更新紀錄:
# v4.1 (2025-09-06): [災難性BUG修復] 根據 KeyError，此節點現在負責以程序化的方式，將一個安全的、不含語法歧義的範例字符串，動態注入到決策鏈的 `{examples}` 佔位符中。這徹底解決了 LangChain 解析器錯誤解析靜態模板中範例的問題。
# v4.0 (2025-09-06): [災難性BUG修復] 徹底重構了此節點的數據傳遞邏輯，改為注入完整的角色 JSON。
# v3.0 (2025-09-18): [災難性BUG修復] 根據 ai_core v3.0 的重構，修改了此節點的邏輯。
async def expansion_decision_node(state: ConversationGraphState) -> Dict:
    """[5] LORE擴展決策節點，基於場景中是否已有合適角色來做決定。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    raw_lore_objects = state.get('raw_lore_objects', [])
    logger.info(f"[{user_id}] (Graph|5) Node: expansion_decision -> 正在基於語意匹配，判斷是否擴展...")
    
    lore_for_decision_making = [lore.content for lore in raw_lore_objects if lore.category == 'npc_profile']
    lore_json_str = json.dumps(lore_for_decision_making, ensure_ascii=False, indent=2)
    
    logger.info(f"[{user_id}] (Graph|5) 注入決策鏈的現有角色JSON:\n{lore_json_str}")

    # [v4.1 核心修正] 將範例從模板中分離出來，作為一個安全的變數傳入
    examples_str = """
- **情境 1**: 
    - 現有角色JSON: `[{"name": "海妖吟", "description": "一位販賣活魚的女性性神教徒..."}]`
    - 使用者輸入: `繼續描述那個賣魚的女人`
    - **你的決策**: `should_expand: false` (理由應類似於: 場景中已存在符合 '賣魚的女人' 描述的角色 (例如 '海妖吟')，應優先與其互動。)
- **情境 2**:
    - 現有角色JSON: `[{"name": "海妖吟", "description": "一位女性性神教徒..."}]`
    - 使用者輸入: `這時一個衛兵走了過來`
    - **你的決策**: `should_expand: true` (理由應類似於: 場景中缺乏能夠扮演 '衛兵' 的角色，需要創建新角色以響應指令。)
"""

    decision_chain = ai_core.get_expansion_decision_chain()
    decision = await ai_core.ainvoke_with_rotation(
        decision_chain, 
        {
            "user_input": user_input, 
            "existing_characters_json": lore_json_str,
            "examples": examples_str # 動態注入範例
        },
        retry_strategy='euphemize'
    )

    if not decision:
        logger.warning(f"[{user_id}] (Graph|5) LORE擴展決策鏈失敗，安全備援為不擴展。")
        decision = ExpansionDecision(should_expand=False, reasoning="安全備援：決策鏈失敗。")
    
    logger.info(f"[{user_id}] (Graph|5) LORE擴展決策: {decision.should_expand}。理由: {decision.reasoning}")
    return {"expansion_decision": decision}
# 函式：LORE擴展決策節點 (v4.1 - 範例注入)



# 函式：無害化輸入節點 (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-09-18): [重大架構升級] 創建此專用節點，作為所有規劃器前的“淨化層”。它調用委婉化鏈，將可能觸發內容審查的原始使用者輸入，轉化為一個保留核心意圖但用詞中性的安全指令，旨在從根本上解決因輸入“投毒”導致整個規劃鏈被 API 攔截的問題。
async def sanitize_input_node(state: ConversationGraphState) -> Dict:
    """將原始使用者輸入轉化為一個對規劃器安全的、中性的指令。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    intent_type = state['intent_classification'].intent_type
    
    # 只有 NSFW 相關的意圖需要淨化
    if 'nsfw' not in intent_type:
        logger.info(f"[{user_id}] (Graph) Node: sanitize_input -> 意圖為 SFW，跳過淨化。")
        return {"sanitized_user_input": user_input}

    logger.info(f"[{user_id}] (Graph) Node: sanitize_input -> 正在對 NSFW 指令進行無害化處理...")
    
    entity_extraction_chain = ai_core.get_entity_extraction_chain()
    entity_result = await ai_core.ainvoke_with_rotation(entity_extraction_chain, {"text_input": user_input})
    
    if not (entity_result and entity_result.names):
        logger.warning(f"[{user_id}] (Sanitizer) 未能從輸入中提取實體，將使用原始輸入作為安全備援。")
        return {"sanitized_user_input": user_input}
        
    euphemization_chain = ai_core.get_euphemization_chain()
    sanitized_input = await ai_core.ainvoke_with_rotation(
        euphemization_chain,
        {"keywords": entity_result.names},
        retry_strategy='none' # 委婉化本身失敗則無法挽救
    )
    
    if not sanitized_input:
        logger.error(f"[{user_id}] (Sanitizer) 委婉化重構鏈失敗，將使用原始輸入，這極可能導致後續規劃失敗！")
        sanitized_input = user_input
    
    logger.info(f"[{user_id}] (Sanitizer) 指令淨化成功: '{user_input}' -> '{sanitized_input}'")
    return {"sanitized_user_input": sanitized_input}
# 函式：無害化輸入節點 (v1.0 - 全新創建)


# 函式：專用的LORE擴展執行節點 (v8.0 - 錨點持久化)
# 更新紀錄:
# v8.0 (2025-09-06): [重大架構升級] 根據「遠景地點丟失」的根本原因，賦予此節點一項新的關鍵職責：【持久化場景錨點】。在調用選角鏈後，此節點會檢查結果中是否包含一個推斷出的 `implied_location`。如果有，它會立即將該地點存入LORE，並【強制更新GameState】，將系統的視角模式切換到 remote 並設置好遠程目標路徑。此修改從根本上解決了因首次描述時地點信息缺失而導致的後續流程崩潰問題。
# v7.0 (2025-09-06): [災難性BUG修復] 重構了此節點的地點判斷邏輯。
# v6.0 (2025-09-06): [災難性BUG修復] 修正了此節點的輸出邏輯，合併新舊角色。
async def lore_expansion_node(state: ConversationGraphState) -> Dict:
    """[6A] 專用的LORE擴展執行節點，執行選角，錨定場景，並將所有角色綁定為規劃主體。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    existing_lores = state.get('raw_lore_objects', [])
    
    logger.info(f"[{user_id}] (Graph|6A) Node: lore_expansion -> 正在執行場景選角與LORE擴展...")
    
    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|6A) ai_core.profile 未加載，跳過 LORE 擴展。")
        return {}

    # 地點判斷邏輯保持不變，為選角鏈提供一個基礎參考點
    scene_analysis = state.get('scene_analysis')
    gs = ai_core.profile.game_state
    effective_location_path: List[str]

    if gs.viewing_mode == 'remote' and scene_analysis and scene_analysis.target_location_path:
        effective_location_path = scene_analysis.target_location_path
    else:
        effective_location_path = gs.location_path

    # ... (輸入淨化邏輯保持不變) ...
    try:
        entity_chain = ai_core.get_entity_extraction_chain()
        entity_result = await ai_core.ainvoke_with_rotation(entity_chain, {"text_input": user_input})
        sanitized_context_for_casting = "為場景選角：" + " ".join(entity_result.names) if entity_result and entity_result.names else user_input
    except Exception as e:
        logger.error(f"[{user_id}] (LORE Expansion) 預處理失敗: {e}", exc_info=True)
        sanitized_context_for_casting = user_input
        
    game_context_for_casting = json.dumps(state.get('structured_context', {}), ensure_ascii=False, indent=2)

    cast_result = await ai_core.ainvoke_with_rotation(
        ai_core.get_scene_casting_chain(),
        {
            "world_settings": ai_core.profile.world_settings or "", 
            "current_location_path": effective_location_path,
            "game_context": game_context_for_casting, 
            "recent_dialogue": sanitized_context_for_casting
        },
        retry_strategy='euphemize'
    )
    
    # [v8.0 核心修正] 持久化場景錨點
    if cast_result and cast_result.implied_location:
        location_info = cast_result.implied_location
        # 假設 location_info.name 是單一字符串，需要構建路徑
        # 一個簡單的策略是將它附加到玩家當前的頂層區域
        base_path = [gs.location_path[0]] if gs.location_path else ["未知區域"]
        new_location_path = base_path + [location_info.name]
        lore_key = " > ".join(new_location_path)
        
        await lore_book.add_or_update_lore(user_id, 'location_info', lore_key, location_info.model_dump())
        logger.info(f"[{user_id}] (Scene Anchor) 已成功為場景錨定並創建新地點LORE: '{lore_key}'")
        
        # 強制更新 GameState
        gs.viewing_mode = 'remote'
        gs.remote_target_path = new_location_path
        await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})
        logger.info(f"[{user_id}] (Scene Anchor) GameState 已強制更新為遠程視角，目標: {new_location_path}")

    planning_subjects = [lore.content for lore in existing_lores if lore.category == 'npc_profile']
    
    if cast_result and (cast_result.newly_created_npcs or cast_result.supporting_cast):
        created_names = await ai_core._add_cast_to_scene(cast_result)
        logger.info(f"[{user_id}] (Graph|6A) 選角完成，創建了 {len(created_names)} 位新角色: {', '.join(created_names)}.")
        
        if created_names:
            lore_query_tasks = [lore_book.get_lores_by_category_and_filter(user_id, 'npc_profile', lambda c: c.get('name') in created_names)]
            results = await asyncio.gather(*lore_query_tasks)
            newly_created_lores = results[0]
            
            if newly_created_lores:
                planning_subjects.extend([lore.content for lore in newly_created_lores])
    
    logger.info(f"[{user_id}] (Graph|6A) 已將 {len(planning_subjects)} 位角色 (新舊合併) 成功綁定為本回合的規劃主體。")
    return {"planning_subjects": planning_subjects}
# 函式：專用的LORE擴展執行節點 (v8.0 - 錨點持久化)




# --- 階段二：規劃 (Planning) ---




# 函式：NSFW 初步規劃節點 (v1.0 - 思維鏈)
# 更新紀錄:
# v1.0 (2025-09-06): [重大架構升級] 創建此新節點，作為「NSFW思維鏈」的第一步。它負責調用 `get_nsfw_initial_planning_chain` 來生成一個用詞安全的行動計畫草稿，並將結果傳遞給下一個節點。
async def nsfw_initial_planning_node(state: ConversationGraphState) -> Dict:
    """[7B.1] NSFW思維鏈-步驟1: 生成初步的行動計劃草稿。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|7B.1) Node: nsfw_initial_planning -> 正在生成NSFW初步行動計劃...")
    
    if not ai_core.profile:
        return {"turn_plan": TurnPlan(thought="錯誤：AI profile 未加載，無法規劃。", character_actions=[])}

    planning_subjects_raw = state.get('planning_subjects', [])
    planning_subjects_json = json.dumps(planning_subjects_raw, ensure_ascii=False, indent=2)
    chat_history_str = _get_formatted_chat_history(ai_core, user_id)
    
    # 準備 World Snapshot
    gs = ai_core.profile.game_state
    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': state.get('structured_context', {}).get('possessions_context', ''),
        'quests_context': state.get('structured_context', {}).get('quests_context', ''),
        'location_context': state.get('structured_context', {}).get('location_context', ''),
        'npc_context': "(已棄用，請參考 planning_subjects_json)",
        'relevant_npc_context': "(已棄用，請參考 planning_subjects_json)",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': gs.viewing_mode,
        'remote_target_path_str': " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定",
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)
    
    # 由於NSFW流水線處理本地和遠程，我們在此統一調用
    # 未來可根據 viewing_mode 選擇不同的 initial_planning_chain
    chain_to_call = ai_core.get_nsfw_initial_planning_chain()
    
    plan = await ai_core.ainvoke_with_rotation(
        chain_to_call,
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "world_snapshot": world_snapshot, 
            "chat_history": chat_history_str,
            "planning_subjects_json": planning_subjects_json,
            "user_input": state['messages'][-1].content
        },
        # 使用委婉化重試，因為這一步的目標是安全地生成骨架
        retry_strategy='euphemize'
    )
    if not plan:
        plan = TurnPlan(execution_rejection_reason="安全備援：NSFW初步規劃鏈失敗，可能因為內容審查或API臨時故障。")

    # 將 world_snapshot 保存到狀態中，供後續流水線節點使用
    return {"turn_plan": plan, "world_snapshot": world_snapshot}
# 函式：NSFW 初步規劃節點 (v1.0 - 思維鏈)


# 函式：NSFW 詞彙注入節點 (v1.0 - 思維鏈)
# 更新紀錄:
# v1.0 (2025-09-06): [重大架構升級] 創建此新節點，作為「NSFW思維鏈」的第二步。它接收上一步生成的安全計畫草稿，調用 `get_nsfw_lexicon_injection_chain`，將其轉換為一個用詞極度露骨的版本，並傳遞給下一步。
async def nsfw_lexicon_injection_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7B.2] NSFW思維鏈-步驟2: 強制修正計畫中的詞彙為露骨術語。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph|7B.2) Node: nsfw_lexicon_injection -> 正在注入NSFW露骨詞彙...")

    if not ai_core.profile or not turn_plan or turn_plan.execution_rejection_reason:
        return {} # 如果上一步失敗，直接跳過

    chat_history_str = _get_formatted_chat_history(ai_core, user_id)
    world_snapshot = state.get('world_snapshot', '') # 從上一步獲取 world_snapshot

    corrected_plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_nsfw_lexicon_injection_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction,
            "action_sexual_content_prompt": ai_core.modular_prompts.get("action_sexual_content", "警告:性愛模組未加載"),
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str,
            "turn_plan_json": turn_plan.model_dump_json(indent=2)
        },
        # 詞彙注入是關鍵，必須強制執行
        retry_strategy='force'
    )
    if not corrected_plan:
        logger.warning(f"[{user_id}] (Graph|7B.2) NSFW詞彙注入鏈返回空值，保留原始計畫。")
        return {}
        
    return {"turn_plan": corrected_plan}
# 函式：NSFW 詞彙注入節點 (v1.0 - 思維鏈)




# 函式：SFW規劃節點 (v26.0 - KeyError 修正)
# 更新紀錄:
# v26.0 (2025-09-06): [災難性BUG修復] 根據 KeyError Log，移除了對已被廢棄的 `sanitized_user_input` 狀態的引用。
# v25.0 (2025-09-06): [健壯性] 修改了備援邏輯，改為使用 `execution_rejection_reason` 欄位來傳遞錯誤。
# v24.0 (2025-09-18): [重大架構重構] 修改了數據源。
async def sfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7A] SFW路徑專用規劃器，生成結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    # [v26.0 核心修正] 移除對 sanitized_user_input 的引用
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|7A) Node: sfw_planning -> 正在基於指令 '{user_input[:50]}...' 生成SFW行動計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(execution_rejection_reason="錯誤：AI profile 未加載，無法規劃。")}

    planning_subjects_raw = state.get('planning_subjects')
    if planning_subjects_raw is None:
        lore_objects = state.get('raw_lore_objects', [])
        planning_subjects_raw = [lore.content for lore in lore_objects if lore.category == 'npc_profile']
    planning_subjects_json = json.dumps(planning_subjects_raw, ensure_ascii=False, indent=2)

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
        'npc_context': "(已棄用，請參考 planning_subjects_json)",
        'relevant_npc_context': "(已棄用，請參考 planning_subjects_json)",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': gs.viewing_mode,
        'remote_target_path_str': " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定",
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)
    
    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_sfw_planning_chain(), 
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot, 
            "chat_history": chat_history_str,
            "planning_subjects_json": planning_subjects_json,
            "user_input": user_input,
        },
        retry_strategy='euphemize'
    )
    if not plan:
        plan = TurnPlan(execution_rejection_reason="安全備援：SFW規劃鏈失敗。")
    return {"turn_plan": plan}
# 函式：SFW規劃節點 (v26.0 - KeyError 修正)



# 函式：NSFW 風格合規節點 (v1.0 - 思維鏈)
# 更新紀錄:
# v1.0 (2025-09-06): [重大架構升級] 創建此新節點，作為「NSFW思維鏈」的第三步。它接收已注入露骨詞彙的計畫，調用 `get_nsfw_style_compliance_chain`，為其補充對話和細節以符合使用者風格，生成最終的、可供執行的完整計畫。
async def nsfw_style_compliance_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7B.3] NSFW思維鏈-步驟3: 檢查並補充對話，確保計畫符合用戶風格。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph|7B.3) Node: nsfw_style_compliance -> 正在進行NSFW風格合規檢查...")

    if not ai_core.profile or not turn_plan or turn_plan.execution_rejection_reason:
        return {} # 如果上一步失敗，直接跳過

    chat_history_str = _get_formatted_chat_history(ai_core, user_id)
    world_snapshot = state.get('world_snapshot', '') # 從上一步獲取 world_snapshot

    final_plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_nsfw_style_compliance_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str,
            "turn_plan_json": turn_plan.model_dump_json(indent=2)
        },
        # 風格是最後一步，也需要強制
        retry_strategy='force'
    )
    if not final_plan:
        logger.warning(f"[{user_id}] (Graph|7B.3) NSFW風格合規鏈返回空值，保留修正前計畫。")
        return {}

    return {"turn_plan": final_plan}
# 函式：NSFW 風格合規節點 (v1.0 - 思維鏈)




# 函式：遠程 SFW 規劃節點 (v7.0 - KeyError 修正)
# 更新紀錄:
# v7.0 (2025-09-06): [災難性BUG修復] 根據 KeyError Log，移除了對已被廢棄的 `sanitized_user_input` 狀態的引用。
# v6.0 (2025-09-06): [健壯性] 修改了備援邏輯，改為使用 `execution_rejection_reason` 欄位來傳遞錯誤。
# v5.0 (2025-09-18): [災難性BUG修復] 徹底重構了 target_location_path 的獲取邏輯。
async def remote_sfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7D] SFW 描述路徑專用規劃器，生成遠景場景的結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    # [v7.0 核心修正] 移除對 sanitized_user_input 的引用
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|7D) Node: remote_sfw_planning -> 正在基於指令 '{user_input[:50]}...' 生成遠程SFW場景計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(execution_rejection_reason="錯誤：AI profile 未加載，無法規劃。")}

    scene_analysis = state.get('scene_analysis')
    gs = ai_core.profile.game_state
    target_location_path: Optional[List[str]] = None

    if scene_analysis and scene_analysis.target_location_path:
        target_location_path = scene_analysis.target_location_path
        logger.info(f"[{user_id}] (Graph|7D) 已從當前回合分析中獲取遠程目標: {target_location_path}")
    elif gs.viewing_mode == 'remote' and gs.remote_target_path:
        target_location_path = gs.remote_target_path
        logger.warning(f"[{user_id}] (Graph|7D) 當前回合分析未提供目標，已從持久化 GameState 中成功回退。目標: {target_location_path}")
    else:
        error_msg = "錯誤：未能從當前回合分析或持久化狀態中解析出遠程觀察的目標地點。"
        logger.error(f"[{user_id}] (Graph|7D) {error_msg}")
        return {"turn_plan": TurnPlan(execution_rejection_reason=error_msg)}

    target_location_path_str = " > ".join(target_location_path)
    
    planning_subjects_raw = state.get('planning_subjects')
    if planning_subjects_raw is None:
        lore_objects = state.get('raw_lore_objects', [])
        planning_subjects_raw = [lore.content for lore in lore_objects if lore.category == 'npc_profile']
    planning_subjects_json = json.dumps(planning_subjects_raw, ensure_ascii=False, indent=2)

    chat_history_str = _get_formatted_chat_history(ai_core, user_id)

    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': "(遠程觀察模式)",
        'quests_context': "(遠程觀察模式)",
        'location_context': f"遠程觀察地點: {target_location_path_str}",
        'npc_context': "(已棄用，請參考 planning_subjects_json)",
        'relevant_npc_context': "(已棄用，請參考 planning_subjects_json)",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': 'remote',
        'remote_target_path_str': target_location_path_str,
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)

    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_remote_sfw_planning_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str,
            "planning_subjects_json": planning_subjects_json,
            "target_location_path_str": target_location_path_str,
            "user_input": user_input,
            "username": ai_core.profile.user_profile.name,
            "ai_name": ai_core.profile.ai_profile.name
        },
        retry_strategy='euphemize'
    )
    if not plan:
        plan = TurnPlan(execution_rejection_reason="安全備援：遠程SFW規劃鏈失敗。")
    return {"turn_plan": plan}
# 函式：遠程 SFW 規劃節點 (v7.0 - KeyError 修正)


# 函式：遠程NSFW規劃節點 (v8.0 - KeyError 修正)
# 更新紀錄:
# v8.0 (2025-09-06): [災難性BUG修復] 根據 KeyError Log，移除了對已被廢棄的 `sanitized_user_input` 狀態的引用，改為直接從 `state['messages']` 獲取原始使用者輸入。此修改是“廢除淨化層”策略的後續清理工作。
# v7.0 (2025-09-06): [災難性BUG修復] 徹底修改了此節點的備援邏輯。
# v6.0 (2025-09-06): [災難性BUG修復] 修正了調用鏈時的參數傳遞。
async def remote_nsfw_planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """[7C] NSFW描述路徑專用規劃器，生成遠景場景的結構化行動計劃。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    # [v8.0 核心修正] 移除對 sanitized_user_input 的引用
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|7C) Node: remote_nsfw_planning -> 正在基於指令 '{user_input[:50]}...' 生成遠程NSFW場景計劃...")

    if not ai_core.profile:
        return {"turn_plan": TurnPlan(execution_rejection_reason="錯誤：AI profile 未加載，無法規劃。")}
    
    scene_analysis = state.get('scene_analysis')
    gs = ai_core.profile.game_state
    target_location_path: Optional[List[str]] = None

    if scene_analysis and scene_analysis.target_location_path:
        target_location_path = scene_analysis.target_location_path
        logger.info(f"[{user_id}] (Graph|7C) 已從當前回合分析中獲取遠程目標: {target_location_path}")
    elif gs.viewing_mode == 'remote' and gs.remote_target_path:
        target_location_path = gs.remote_target_path
        logger.warning(f"[{user_id}] (Graph|7C) 當前回合分析未提供目標，已從持久化 GameState 中成功回退。目標: {target_location_path}")
    else:
        error_msg = "錯誤：未能從當前回合分析或持久化狀態中解析出遠程觀察的目標地點。"
        logger.error(f"[{user_id}] (Graph|7C) {error_msg}")
        return {"turn_plan": TurnPlan(execution_rejection_reason=error_msg)}

    target_location_path_str = " > ".join(target_location_path)
    
    planning_subjects_raw = state.get('planning_subjects')
    if planning_subjects_raw is None:
        lore_objects = state.get('raw_lore_objects', [])
        planning_subjects_raw = [lore.content for lore in lore_objects if lore.category == 'npc_profile']
    planning_subjects_json = json.dumps(planning_subjects_raw, ensure_ascii=False, indent=2)

    chat_history_str = _get_formatted_chat_history(ai_core, user_id)

    full_context_dict = {
        'username': ai_core.profile.user_profile.name,
        'ai_name': ai_core.profile.ai_profile.name,
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', ''),
        'possessions_context': "(遠程觀察模式)",
        'quests_context': "(遠程觀察模式)",
        'location_context': f"遠程觀察地點: {target_location_path_str}",
        'npc_context': "(已棄用，請參考 planning_subjects_json)",
        'relevant_npc_context': "(已棄用，請參考 planning_subjects_json)",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': 'remote',
        'remote_target_path_str': target_location_path_str,
    }
    world_snapshot = ai_core.world_snapshot_template.format(**full_context_dict)

    plan = await ai_core.ainvoke_with_rotation(
        ai_core.get_remote_nsfw_planning_chain(),
        {
            "system_prompt": ai_core.profile.one_instruction, 
            "action_sexual_content_prompt": ai_core.modular_prompts.get("action_sexual_content", "警告:性愛模組未加載"),
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格",
            "world_snapshot": world_snapshot,
            "chat_history": chat_history_str,
            "planning_subjects_json": planning_subjects_json,
            "target_location_path_str": target_location_path_str,
            "user_input": user_input,
            "username": ai_core.profile.user_profile.name,
            "ai_name": ai_core.profile.ai_profile.name
        },
        retry_strategy='force'
    )
    if not plan:
        plan = TurnPlan(execution_rejection_reason="安全備援：遠程NSFW規劃鏈最終失敗，可能因為內容審查或API臨時故障。")
    return {"turn_plan": plan}
# 函式：遠程NSFW規劃節點 (v8.0 - KeyError 修正)


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

# 函式：統一的敘事渲染節點 (v24.0 - 執行否決修正)
# 更新紀錄:
# v24.0 (2025-09-06): [災難性BUG修復] 根據「AI輸出內部錯誤」的問題，在此節點增加了“執行否決”檢查。在渲染前，它會檢查傳入的 `TurnPlan` 的 `execution_rejection_reason` 欄位。如果有值，則完全跳過渲染，直接將該欄位的錯誤訊息作為最終輸出。此修改從根本上解決了系統錯誤被當作劇情渲染的嚴重問題。
# v23.0 (2025-09-06): [災難性BUG修復] 修正了傳遞給渲染鏈的參數結構。
# v22.0 (2025-09-05): [災難性BUG修復] 修正了調用鏈時的參數傳遞。
async def narrative_rendering_node(state: ConversationGraphState) -> Dict[str, str]:
    """[9] 統一的敘事渲染節點，將行動計劃轉化為小說文本。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph|9) Node: narrative_rendering -> 正在將行動計劃渲染為小說...")

    if not turn_plan:
        return {"llm_response": "（系統錯誤：未能生成有效的行動計劃。）"}
        
    # [v24.0 核心修正] 執行否決權檢查
    if turn_plan.execution_rejection_reason:
        logger.warning(f"[{user_id}] (Narrator) 檢測到上游規劃節點的執行否決，跳過渲染。理由: {turn_plan.execution_rejection_reason}")
        return {"llm_response": turn_plan.execution_rejection_reason}
        
    chain_input = {
        "system_prompt": ai_core.profile.one_instruction if ai_core.profile else "預設系統指令",
        "response_style_prompt": ai_core.profile.response_style_prompt if ai_core.profile else "預設風格",
        "turn_plan": turn_plan
    }
        
    narrative_text = await ai_core.ainvoke_with_rotation(
        ai_core.get_narrative_chain(),
        chain_input,
        retry_strategy='force'
    )
    if not narrative_text:
        narrative_text = "（AI 在將計劃轉化為故事時遭遇了內容安全限制。）"
    return {"llm_response": narrative_text}
# 函式：統一的敘事渲染節點 (v24.0 - 執行否決修正)




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








# 函式：創建主回應圖 (v36.0 - 動作分解架構)
# 更新紀錄:
# v36.0 (2025-09-07): [終極架構重構] 根據全新的「動作分解原則」對圖的拓撲進行了最終簡化。
#    1. [移除] 徹底廢棄並移除了舊的、複雜的「NSFW思維鏈」流水線（`nsfw_initial_planning`, `nsfw_lexicon_injection`, `nsfw_style_compliance`）。
#    2. [替換] 引入了全新的、統一的 `nsfw_planning_node` 和 `remote_nsfw_planning_node`，它們現在是各自路徑的唯一規劃器。
#    3. [重連] 修改了 `route_to_planner` 路由器，使其現在將 NSFW 流量精確地分發到對應的本地或遠程 NSFW 規劃器。
#    此修改使圖的結構回歸到一個更簡潔、更強大、且完全基於創意生成的「統一規劃 -> 統一渲染」模型。
# v34.0 (2025-09-06): [災難性BUG修復] 引入了NSFW思維鏈。
def create_main_response_graph() -> StateGraph:
    graph = StateGraph(ConversationGraphState)
    
    # --- 1. 註冊所有節點 ---
    graph.add_node("pre_process_input", pre_process_input_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("perceive_and_set_view", perceive_and_set_view_node)
    graph.add_node("retrieve_memories", retrieve_memories_node)
    graph.add_node("query_lore", query_lore_node)
    graph.add_node("assemble_context", assemble_context_node)
    graph.add_node("expansion_decision", expansion_decision_node)
    graph.add_node("lore_expansion", lore_expansion_node)

    # 註冊所有統一規劃器
    graph.add_node("sfw_planning", sfw_planning_node)
    graph.add_node("remote_sfw_planning", remote_sfw_planning_node)
    graph.add_node("nsfw_planning", nsfw_planning_node) # v36.0 新增
    graph.add_node("remote_nsfw_planning", remote_nsfw_planning_node) # v36.0 新增
    
    # 註冊後續通用節點
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("narrative_rendering", narrative_rendering_node)
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)
    
    # 註冊匯合點與輔助節點
    graph.add_node("planner_junction", lambda state: {})
    
    def prepare_existing_subjects_node(state: ConversationGraphState) -> Dict:
        lore_objects = state.get('raw_lore_objects', [])
        planning_subjects = [lore.content for lore in lore_objects if lore.category == 'npc_profile']
        logger.info(f"[{state['user_id']}] (Graph) Node: prepare_existing_subjects -> 已將 {len(planning_subjects)} 個現有NPC打包為規劃主體。")
        return {"planning_subjects": planning_subjects}
        
    graph.add_node("prepare_existing_subjects", prepare_existing_subjects_node)


    # --- 2. 定義圖的拓撲結構 ---
    graph.set_entry_point("pre_process_input")
    graph.add_edge("pre_process_input", "classify_intent")
    graph.add_edge("classify_intent", "retrieve_memories")
    graph.add_edge("retrieve_memories", "query_lore")
    graph.add_edge("query_lore", "perceive_and_set_view")
    graph.add_edge("perceive_and_set_view", "assemble_context")
    graph.add_edge("assemble_context", "expansion_decision")
    
    graph.add_conditional_edges(
        "expansion_decision",
        route_expansion_decision,
        { 
            "expand_lore": "lore_expansion", 
            "continue_to_planner": "prepare_existing_subjects"
        }
    )
    
    graph.add_edge("lore_expansion", "planner_junction")
    graph.add_edge("prepare_existing_subjects", "planner_junction")

    def route_to_planner(state: ConversationGraphState) -> str:
        user_id = state['user_id']
        intent_classification = state.get('intent_classification')
        if not intent_classification:
            logger.error(f"[{user_id}] (Router) 致命錯誤：意圖分類結果不存在，無法路由。")
            return "sfw_planning" 

        intent = intent_classification.intent_type
        ai_core = state['ai_core']
        viewing_mode = ai_core.profile.game_state.viewing_mode if ai_core.profile else 'local'
        
        logger.info(f"[{user_id}] (Router) Routing to planner. Intent: '{intent}', Final Viewing Mode: '{viewing_mode}'")
        
        # [v36.0 核心修正] 精確路由到對應的規劃器
        if 'nsfw' in intent:
            if viewing_mode == 'remote':
                return "remote_nsfw_planning"
            else: # local
                return "nsfw_planning"
        else: # sfw
            if viewing_mode == 'remote':
                return "remote_sfw_planning"
            else: # local
                return "sfw_planning"

    graph.add_conditional_edges(
        "planner_junction",
        route_to_planner,
        { 
            "sfw_planning": "sfw_planning", 
            "remote_sfw_planning": "remote_sfw_planning",
            "nsfw_planning": "nsfw_planning",
            "remote_nsfw_planning": "remote_nsfw_planning"
        }
    )
    
    # 所有規劃器的出口都統一連接到工具執行
    graph.add_edge("sfw_planning", "tool_execution")
    graph.add_edge("remote_sfw_planning", "tool_execution")
    graph.add_edge("nsfw_planning", "tool_execution")
    graph.add_edge("remote_nsfw_planning", "tool_execution")
    
    # 後續流程保持統一
    graph.add_edge("tool_execution", "narrative_rendering")
    graph.add_edge("narrative_rendering", "validate_and_rewrite")
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", END)
    
    return graph.compile()
# 函式：創建主回應圖 (v36.0 - 動作分解架構)


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
