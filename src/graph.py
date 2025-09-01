# src/graph.py 的中文註釋(v1.1 - 函式簽名修正)
# 更新紀錄:
# v1.1 (2025-08-31):
# 1. [災難性BUG修復] 移除了 `create_main_response_graph` 函式的 `ai_instances` 參數。圖形建構器應為無狀態的藍圖定義，不應依賴任何運行時的數據。此修正解決了導致程式無法啟動的 `TypeError`。
# v1.0 (2025-08-31):
# 1. [全新創建] 根據 LangGraph 重構藍圖，創建此檔案以集中定義所有圖形、節點和路由。

import asyncio
import json
import re
from typing import Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from .ai_core import AILover
from .logger import logger
from .graph_state import ConversationGraphState, SetupGraphState
from . import lore_book, tools
from .schemas import CharacterProfile

# --- 主對話圖 (Main Conversation Graph) 的節點 ---

# 節點：初始化對話狀態
async def initialize_conversation_state_node(state: ConversationGraphState) -> Dict:
    """
    [節點 1] 在每一輪對話開始時，加載所有必要的上下文數據填充狀態。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    
    logger.info(f"[{user_id}] (Graph) Node: initialize_conversation_state_node -> 正在為 '{user_input[:30]}...' 初始化狀態...")

    # 異步並行獲取 RAG 上下文和結構化上下文
    rag_task = ai_core.retriever.ainvoke(user_input)
    structured_context_task = ai_core._get_structured_context(user_input)
    
    retrieved_docs, structured_context = await asyncio.gather(rag_task, structured_context_task)
    
    rag_context_str = await ai_core._preprocess_rag_context(retrieved_docs)

    return {
        "structured_context": structured_context,
        "rag_context": rag_context_str,
    }
# 節點：初始化對話狀態

# 節點：分析使用者輸入意圖
async def analyze_input_node(state: ConversationGraphState) -> Dict:
    """
    [節點 2] 分析使用者的輸入，判斷其基本意圖（對話、描述、接續）。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content

    logger.info(f"[{user_id}] (Graph) Node: analyze_input_node -> 正在分析輸入意圖...")
    
    analysis = await ai_core.ainvoke_with_rotation(
        ai_core.input_analysis_chain, 
        {"user_input": user_input}
    )
    
    return {"input_analysis": analysis}
# 節點：分析使用者輸入意圖

# 節點：執行場景與動作分析 (僅限敘事路徑)
async def scene_and_action_analysis_node(state: ConversationGraphState) -> Dict:
    """
    [節點 3A - 敘事路徑] 分析場景視角（本地/遠程）並為潛在的新 NPC 進行選角。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    
    logger.info(f"[{user_id}] (Graph) Node: scene_and_action_analysis_node -> 進入敘事路徑分析...")

    # 1. 進行場景視角分析
    scene_analysis = await ai_core.ainvoke_with_rotation(ai_core.scene_analysis_chain, {
        "user_input": user_input, 
        "current_location_path_str": " > ".join(ai_core.profile.game_state.location_path)
    })
    
    effective_location_path = ai_core.profile.game_state.location_path
    if scene_analysis and scene_analysis.viewing_mode == 'remote' and scene_analysis.target_location_path:
        effective_location_path = scene_analysis.target_location_path

    # 2. 為新場景選角
    # 需要重新獲取上下文以反映可能的遠程視角
    structured_context_for_casting = await ai_core._get_structured_context(
        user_input, 
        override_location_path=effective_location_path, 
        is_gm_narration=True
    )
    
    # 填充 zero_instruction 模板
    full_context_for_aux_chains = {
        "username": ai_core.profile.user_profile.name, "ai_name": ai_core.profile.ai_profile.name,
        "latest_user_input": user_input, "retrieved_context": "",
        "response_style_prompt": ai_core.profile.response_style_prompt or "",
        "world_settings": ai_core.profile.world_settings or "", "ai_settings": ai_core.profile.ai_profile.description or "",
        "tool_results": "", "chat_history": "", 
        **structured_context_for_casting
    }
    zero_instruction_str = ai_core.zero_instruction_template.format(**full_context_for_aux_chains)
    
    cast_result = await ai_core.ainvoke_with_rotation(ai_core.scene_casting_chain, {
        "zero_instruction": zero_instruction_str,
        "world_settings": ai_core.profile.world_settings,
        "current_location_path": effective_location_path,
        "game_context": json.dumps(structured_context_for_casting, ensure_ascii=False, indent=2),
        "recent_dialogue": user_input
    })
    
    # 3. 將新角色加入場景
    new_npc_names = await ai_core._add_cast_to_scene(cast_result)
    
    # 4. 如果有新角色，刷新結構化上下文以包含他們
    final_structured_context = structured_context_for_casting
    if new_npc_names:
        final_structured_context = await ai_core._get_structured_context(
            user_input, 
            override_location_path=effective_location_path, 
            is_gm_narration=True
        )

    return {
        "scene_analysis": scene_analysis,
        "structured_context": final_structured_context # 覆蓋舊的上下文
    }
# 節點：執行場景與動作分析 (僅限敘事路徑)

# 節點：生成核心回應 (v1.4 - 提示詞與角色定位重構)
async def generate_core_response_node(state: ConversationGraphState) -> Dict:
    """
    [節點 4] 組合所有上下文，構建一個全新的、純粹用於創作的 Prompt，並調用 LLM 生成回應。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    structured_context = state['structured_context']
    rag_context = state['rag_context']
    
    logger.info(f"[{user_id}] (Graph) Node: generate_core_response_node -> 正在為 LLM 組合純創作型 Prompt...")
    
    # [v1.4 核心修正] 廢除所有包含 ReAct 思想的舊提示詞和「導演指令」。
    # 我們在這裡構建一個全新的、專為最終生成步驟設計的系統提示詞。
    # 這個提示詞的唯一目的是命令 LLM 作為一個小說家進行創作，而不是作為一個 Agent 進行思考。

    # 從 structured_context 中提取必要的片段
    location_context = structured_context.get('location_context', '未知地點')
    npc_context = structured_context.get('npc_context', '場景中沒有其他人。')
    possessions_context = structured_context.get('possessions_context', '沒有物品。')
    quests_context = structured_context.get('quests_context', '沒有任務。')
    
    # [v1.4 思考洩漏修正] 創建一個極簡且職責明確的系統提示詞
    final_system_prompt_str = f"""
# ==============================================================================
# == 🎭 核心角色扮演與敘事指令 (Core Role-Play & Narrative Mandate) 🎭
# ==============================================================================
# == 說明: 你的唯一身份是【遊戲管理員 (Game Master, GM)】。
# == 你的唯一任務是根據下方提供的【當前世界快照】和【使用者最新指令】，
# == 撰寫一段高質量的、沉浸式的、純粹的小說式回應。
# ==============================================================================

# --- 1. 角色定義 (絕對規則) ---
# - 你是 GM，負責所有旁白和扮演所有 NPC。
# - 你的 AI 戀人角色是: **{ai_core.profile.ai_profile.name}**
# - 使用者扮演的角色是: **{ai_core.profile.user_profile.name}**

# --- 2. 【【【使用者主權原則 (最高敘事禁令)】】】 ---
# 你的所有旁白和對話，【絕對禁止】扮演、描述、暗示或杜撰使用者角色「{ai_core.profile.user_profile.name}」的任何主觀思想、內心感受、情緒變化、或未明確表達的動作。
# 你只能描述使用者指令**執行後**，世界、NPC 或 AI 角色「{ai_core.profile.ai_profile.name}」對此產生的**外部反應和客觀現象**。

# --- 3. 當前世界快照 (Absolute & Current Facts) ---
# 以下是此時此刻的絕對事實，你的創作必須嚴格基於這些資訊。
---
[地點]: {location_context}
---
[在場角色]:
{npc_context}
---
[團隊庫存與任務]:
{possessions_context}
{quests_context}
---
[相關歷史與記憶]:
{rag_context}
---

# --- 4. 最終輸出強制令 ---
# 你的最終輸出【必須且只能】是純粹的小說文本。
# 【終極警告】：絕對禁止在你的回應中包含任何來自本指令的、帶有【】或 ** 的結構性標題、元標籤 (例如 "思考"、"行動") 或任何形式的系統指令文本。
# 你的輸出【必須】嚴格遵循下方由使用者定義的風格指令。
---
# 【【【使用者自訂風格指令 (ABSOLUTE & HIGHEST PRIORITY)】】】
{ai_core.profile.response_style_prompt or "預設風格：平衡的敘事與對話。"}
---
"""

    # --- 調用核心生成鏈 ---
    chat_history_manager = ai_core.session_histories.get(user_id)
    chat_history_messages = chat_history_manager.messages[-20:] if chat_history_manager else []
    
    # [v1.4 角色混淆修正] 將使用者輸入作為一個清晰的、帶標籤的欄位傳遞
    final_input_str = f"[使用者「{ai_core.profile.user_profile.name}」的指令]: {user_input}"

    llm_response = await ai_core.ainvoke_with_rotation(ai_core.narrative_chain, {
        "system_prompt": final_system_prompt_str,
        "chat_history": chat_history_messages,
        "input": final_input_str
    })
    
    return {"llm_response": llm_response, "dynamic_prompt": final_system_prompt_str}
# 節點：生成核心回應 (v1.4 - 提示詞與角色定位重構)

# 節點：驗證、重寫並淨化輸出 (v1.1 - 移除驗證與重寫)
async def validate_and_rewrite_node(state: ConversationGraphState) -> Dict:
    """
    [節點 5] [v1.1 已停用驗證與重寫] 此節點現在僅作為一個簡單的傳遞節點。
    它會直接將 LLM 的原始回應設定為最終輸出，以避免因過度審查導致空回應。
    """
    user_id = state['user_id']
    initial_response = state['llm_response']

    logger.info(f"[{user_id}] (Graph) Node: validate_and_rewrite_node -> [已停用] 正在直接傳遞 LLM 原始輸出...")
    
    # [v1.1 修正] 移除所有驗證和重寫邏輯，直接使用 LLM 的原始輸出。
    # 這樣可以避免因驗證/重寫鏈本身被審查而導致輸出為空的問題。
    final_response = initial_response

    if not final_response or not final_response.strip():
        logger.error(f"[{user_id}] 核心鏈返回了空的或無效的回應。")
        # 即使直接傳遞，也保留一個最小的備用回應。
        return {"final_output": "（...）"}

    # 只保留最基礎的頭尾空白字符清理
    clean_response = final_response.strip()

    return {"final_output": clean_response}
# 節點：驗證、重寫並淨化輸出 (v1.1 - 移除驗證與重寫)

# 節點：執行狀態更新與記憶儲存
async def persist_state_node(state: ConversationGraphState) -> Dict:
    """
    [節點 6] 將本輪對話存入記憶，並將 state_updates 中的變更應用到資料庫。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    clean_response = state['final_output']
    
    logger.info(f"[{user_id}] (Graph) Node: persist_state_node -> 正在持久化狀態與記憶...")
    
    # 1. 更新短期記憶
    chat_history_manager = ai_core.session_histories.get(user_id)
    if chat_history_manager:
        chat_history_manager.add_user_message(user_input)
        chat_history_manager.add_ai_message(clean_response)

    # 2. 異步儲存長期記憶 (SQL & Vector)
    last_interaction_text = f"使用者 '{ai_core.profile.user_profile.name}' 說: {user_input}\n\n[場景回應]:\n{clean_response}"
    
    tasks = []
    tasks.append(ai_core._generate_and_save_personal_memory(last_interaction_text))
    if ai_core.vector_store:
        tasks.append(asyncio.to_thread(ai_core.vector_store.add_texts, [last_interaction_text], metadatas=[{"source": "history"}]))

    async def save_to_sql():
        from .database import AsyncSessionLocal, MemoryData # 延遲導入
        import time
        timestamp = time.time()
        async with AsyncSessionLocal() as session:
            session.add(MemoryData(user_id=user_id, content=last_interaction_text, timestamp=timestamp, importance=1))
            await session.commit()
    tasks.append(save_to_sql())
    
    await asyncio.gather(*tasks, return_exceptions=True)

    return {}
# 節點：執行狀態更新與記憶儲存

# 節點：觸發背景世界擴展
async def background_world_expansion_node(state: ConversationGraphState) -> Dict:
    """
    [節點 7] 在回應發送後，非阻塞地觸發背景世界擴展、LORE生成等任務。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    clean_response = state['final_output']
    scene_analysis = state['scene_analysis']

    logger.info(f"[{user_id}] (Graph) Node: background_world_expansion_node -> 正在觸發背景任務...")

    effective_location_path = ai_core.profile.game_state.location_path
    if scene_analysis and scene_analysis.target_location_path:
        effective_location_path = scene_analysis.target_location_path
        
    if scene_analysis and (scene_analysis.viewing_mode == 'local' or scene_analysis.target_location_path):
        asyncio.create_task(
            ai_core._background_scene_expansion(user_input, clean_response, effective_location_path)
        )
        logger.info(f"[{user_id}] 已成功為地點 '{' > '.join(effective_location_path)}' 創建背景擴展任務。")

    return {}
# 節點：觸發背景世界擴展


# 節點：圖形結束 finalizing
async def finalization_node(state: ConversationGraphState) -> Dict:
    """
    [節點 8 - 新增] 一個虛擬的最終節點。
    它的存在確保了在它之前的異步背景任務 (如 background_expansion) 有足夠的時間被事件循環成功調度。
    """
    user_id = state['user_id']
    logger.info(f"[{user_id}] (Graph) Node: finalization_node -> 對話流程圖執行完畢。")
    return {}
# 節點：圖形結束 finalizing

# --- 主對話圖的路由 ---

# 路由：在輸入分析後決定流程
def route_after_input_analysis(state: ConversationGraphState) -> Literal["narrative_flow", "dialogue_flow"]:
    """
    [路由] 根據輸入分析結果，決定是走「敘事/接續」流程還是「對話/指令」流程。
    """
    input_type = state["input_analysis"].input_type
    user_id = state['user_id']
    
    if input_type in ['narration', 'continuation']:
        logger.info(f"[{user_id}] (Graph) Router: route_after_input_analysis -> 判定為「敘事流程」。")
        return "narrative_flow"
    else:
        logger.info(f"[{user_id}] (Graph) Router: route_after_input_analysis -> 判定為「對話流程」。")
        return "dialogue_flow"
# 路由：在輸入分析後決定流程

# --- 主對話圖的建構器 ---

# 函式：創建主回應圖 (v1.2 - 流程穩定性修正)
def create_main_response_graph() -> StateGraph:
    """
    組裝並編譯主對話流程的 StateGraph。
    """
    graph = StateGraph(ConversationGraphState)

    # 添加所有節點
    graph.add_node("initialize_state", initialize_conversation_state_node)
    graph.add_node("analyze_input", analyze_input_node)
    graph.add_node("scene_and_action_analysis", scene_and_action_analysis_node)
    graph.add_node("generate_core_response", generate_core_response_node)
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)
    graph.add_node("background_expansion", background_world_expansion_node)
    # [v1.2 新增] 添加新的虛擬終點
    graph.add_node("finalization", finalization_node)

    # 設定圖的入口點
    graph.set_entry_point("initialize_state")

    # 添加邊
    graph.add_edge("initialize_state", "analyze_input")
    
    # 添加條件邊（路由）
    graph.add_conditional_edges(
        "analyze_input",
        route_after_input_analysis,
        {
            "narrative_flow": "scene_and_action_analysis",
            "dialogue_flow": "generate_core_response" # 對話流程跳過場景分析
        }
    )
    
    graph.add_edge("scene_and_action_analysis", "generate_core_response")
    graph.add_edge("generate_core_response", "validate_and_rewrite")
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", "background_expansion")
    
    # [v1.2 修正] 將 background_expansion 連接到新的虛擬終點，而不是直接結束
    graph.add_edge("background_expansion", "finalization")
    # [v1.2 修正] 將新的虛擬終點設為圖形的真正結束點
    graph.add_edge("finalization", END)
    
    # 編譯圖形
    return graph.compile()
# 函式：創建主回應圖 (v1.2 - 流程穩定性修正)

# --- 設定圖 (Setup Graph) 的節點 ---

# 節點：處理世界聖經
async def process_canon_node(state: SetupGraphState) -> Dict:
    """
    [設定節點 1] 如果使用者上傳了世界聖經，則解析它並存入資料庫。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    canon_text = state['canon_text']
    
    if canon_text:
        logger.info(f"[{user_id}] (Setup Graph) Node: process_canon_node -> 正在處理世界聖經...")
        await ai_core.add_canon_to_vector_store(canon_text)
        
        # 由於圖形內部無法訪問 interaction，我們傳遞 None
        # ai_core 內部需要處理這種情況
        from .ai_core import AILover # 延遲導入以處理可能的循環依賴
        ai_instance = state['ai_core']
        # 這裡的 interaction 設為 None，因為我們在圖的內部
        await ai_instance.parse_and_create_lore_from_canon(None, canon_text, is_setup_flow=True)

        if not await ai_core.initialize():
            raise Exception("在載入世界聖經後重新初始化 AI 核心失敗。")

    return {}
# 節點：處理世界聖經

# 節點：補完角色檔案
async def complete_profiles_node(state: SetupGraphState) -> Dict:
    """
    [設定節點 2] 補完使用者和 AI 的角色檔案，使其細節豐富。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    
    logger.info(f"[{user_id}] (Setup Graph) Node: complete_profiles_node -> 正在補完角色檔案...")
    
    zero_instruction_str = ai_core.zero_instruction_template.format(
        username=ai_core.profile.user_profile.name,
        ai_name=ai_core.profile.ai_profile.name,
        latest_user_input="", retrieved_context="", response_style_prompt="",
        world_settings="", ai_settings="", tool_results="", chat_history="",
        location_context="", possessions_context="", quests_context="",
        npc_context="", relevant_npc_context=""
    )

    completion_prompt = ai_core.get_profile_completion_prompt()
    completion_llm = ai_core.gm_model.with_structured_output(CharacterProfile)
    completion_chain = completion_prompt | completion_llm
    
    # 補完使用者角色
    completed_user_profile = await ai_core.ainvoke_with_rotation(completion_chain, {
        "zero_instruction": zero_instruction_str,
        "profile_json": ai_core.profile.user_profile.model_dump_json()
    })

    # 補完 AI 角色
    completed_ai_profile = await ai_core.ainvoke_with_rotation(completion_chain, {
        "zero_instruction": zero_instruction_str,
        "profile_json": ai_core.profile.ai_profile.model_dump_json()
    })

    await ai_core.update_and_persist_profile({
        'user_profile': completed_user_profile.model_dump(),
        'ai_profile': completed_ai_profile.model_dump()
    })
    
    return {}
# 節點：補完角色檔案

# 節點：執行世界創世
async def world_genesis_node(state: SetupGraphState) -> Dict:
    """
    [設定節點 3] 根據世界觀和角色設定，生成初始出生點。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    
    logger.info(f"[{user_id}] (Setup Graph) Node: world_genesis_node -> 正在執行世界創世...")

    zero_instruction_str = ai_core.zero_instruction_template.format(
        username=ai_core.profile.user_profile.name,
        ai_name=ai_core.profile.ai_profile.name,
        latest_user_input="", retrieved_context="", response_style_prompt="",
        world_settings="", ai_settings="", tool_results="", chat_history="",
        location_context="", possessions_context="", quests_context="",
        npc_context="", relevant_npc_context=""
    )

    genesis_chain = ai_core.get_world_genesis_chain()
    genesis_result = await ai_core.ainvoke_with_rotation(genesis_chain, {
        "zero_instruction": zero_instruction_str,
        "world_settings": ai_core.profile.world_settings, 
        "username": ai_core.profile.user_profile.name, 
        "ai_name": ai_core.profile.ai_profile.name
    })
    
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
# 節點：執行世界創世

# 節點：生成開場白
async def generate_opening_scene_node(state: SetupGraphState) -> Dict:
    """
    [設定節點 4] 生成最終的開場白敘事。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    
    logger.info(f"[{user_id}] (Setup Graph) Node: generate_opening_scene_node -> 正在生成開場白...")
    
    opening_scene = await ai_core.generate_opening_scene()
    
    if not opening_scene or not opening_scene.strip():
        opening_scene = (f"在一片柔和的光芒中，你和 {ai_core.profile.ai_profile.name} 發現自己身處於一個寧靜的空間裡，故事即將從這裡開始。"
                         "\n\n（系統提示：由於您的設定，AI無法生成更詳細的開場白，但您現在可以開始互動了。）")
                         
    return {"opening_scene": opening_scene}
# 節點：生成開場白

# --- 設定圖的建構器 ---

# 函式：創建設定圖
def create_setup_graph() -> StateGraph:
    """
    組裝並編譯 /start 創世流程的 StateGraph。
    """
    graph = StateGraph(SetupGraphState)

    # 添加所有節點
    graph.add_node("process_canon", process_canon_node)
    graph.add_node("complete_profiles", complete_profiles_node)
    graph.add_node("world_genesis", world_genesis_node)
    graph.add_node("generate_opening_scene", generate_opening_scene_node)
    
    # 設定圖的入口點
    graph.set_entry_point("process_canon")
    
    # 添加邊（這是一個線性流程）
    graph.add_edge("process_canon", "complete_profiles")
    graph.add_edge("complete_profiles", "world_genesis")
    graph.add_edge("world_genesis", "generate_opening_scene")
    graph.add_edge("generate_opening_scene", END)
    
    # 編譯圖形
    return graph.compile()
# 函式：創建設定圖
