# src/graph.py 的中文註釋(v7.2 - NameError 修正)
# 更新紀錄:
# v7.2 (2025-09-05): [災難性BUG修復] 根據 NameError Log，補全了在 v7.1 版本中被意外遺漏的 `generate_nsfw_response_node` 函式的完整定義。此錯誤導致圖在編譯時因找不到節點定義而崩潰。
# v7.1 (2025-09-05): [災難性BUG修復] 修復了圖結構分叉問題，確保 SFW 路徑不會被重複執行。
# v7.0 (2025-09-05): [重大架構重構] 實現了混合模式圖架構，引入了 NSFW/SFW 雙路徑處理流程。

import sys
print(f"[DEBUG] graph.py loaded from: {__file__}", file=sys.stderr)
import asyncio
import json
import re
from typing import Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, END

from .ai_core import AILover
from .logger import logger
from .graph_state import ConversationGraphState, SetupGraphState
from . import lore_book, tools
from .schemas import CharacterProfile, TurnPlan, ExpansionDecision
from .tool_context import tool_context

# --- 主對話圖 (Main Conversation Graph) 的節點 ---

# 函式：初始化對話狀態
async def initialize_conversation_state_node(state: ConversationGraphState) -> Dict:
    """
    [節點 1] 在每一輪對話開始時，加載所有必要的上下文數據填充狀態。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: initialize_conversation_state_node -> 正在為 '{user_input[:30]}...' 初始化狀態...")
    rag_task = ai_core.retriever.ainvoke(user_input)
    structured_context_task = ai_core._get_structured_context(user_input)
    retrieved_docs, structured_context = await asyncio.gather(rag_task, structured_context_task)
    rag_context_str = await ai_core._preprocess_rag_context(retrieved_docs)
    return {"structured_context": structured_context, "rag_context": rag_context_str}
# 函式：初始化對話狀態

# 函式：分析使用者輸入意圖
async def analyze_input_node(state: ConversationGraphState) -> Dict:
    """
    [節點 2] 分析使用者的輸入，判斷其基本意圖（對話、描述、接續）。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: analyze_input_node -> 正在分析輸入意圖...")
    analysis = await ai_core.ainvoke_with_rotation(ai_core.input_analysis_chain, {"user_input": user_input})
    return {"input_analysis": analysis}
# 函式：分析使用者輸入意圖

# 函式：判斷是否需要進行LORE擴展
async def expansion_decision_node(state: ConversationGraphState) -> Dict:
    """
    [SFW 路徑節點] 一個“守門人”節點，在LORE創造流程前判斷使用者的“探索意圖”。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: expansion_decision_node -> 正在判斷是否需要擴展LORE...")
    
    chat_history_manager = ai_core.session_histories.get(user_id, ChatMessageHistory())
    recent_dialogue = "\n".join([f"{'使用者' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in chat_history_manager.messages[-6:]])

    decision = await ai_core.ainvoke_with_rotation(ai_core.expansion_decision_chain, {
        "user_input": user_input,
        "recent_dialogue": recent_dialogue
    })
    
    logger.info(f"[{user_id}] (Graph) LORE擴展決策: {decision.should_expand}。理由: {decision.reasoning}")
    return {"expansion_decision": decision}
# 函式：判斷是否需要進行LORE擴展

# 函式：執行場景與動作分析 (v3.1 - 路由前置分析)
# 更新紀錄:
# v3.1 (2025-09-05): [重大架構修正] 提升了此節點在 SFW 路徑中的執行順序。它現在會在 LORE 擴展決策之後、但在路由之前立即執行。這樣，它生成的 `scene_analysis` 結果就可以被 `route_expansion` 路由用來判斷是進入遠程觀察還是本地擴展，使其成為 SFW 探索路徑的核心分析中樞。
# v3.0 (2025-09-03): [功能擴展] 注入了選角上下文以生成更相關的 NPC。
async def scene_and_action_analysis_node(state: ConversationGraphState) -> Dict:
    """
    [SFW-探索路徑分析中樞] 分析場景視角，並為潛在的本地LORE擴展進行選角。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: scene_and_action_analysis_node -> 正在進行場景視角分析與潛在選角...")
    
    current_location_path = ai_core.profile.game_state.location_path if ai_core.profile else []
    
    # 步驟 1: 進行場景視角分析，這是後續路由的關鍵依據
    scene_analysis = await ai_core.ainvoke_with_rotation(ai_core.scene_analysis_chain, {
        "user_input": user_input, 
        "current_location_path_str": " > ".join(current_location_path)
    })
    
    # 步驟 2: 如果是本地擴展模式，則繼續執行選角
    if scene_analysis.viewing_mode == 'local':
        logger.info(f"[{user_id}] (Graph) ...視角為本地，繼續執行選角流程。")
        effective_location_path = current_location_path
            
        structured_context_for_casting = await ai_core._get_structured_context(
            user_input, 
            override_location_path=effective_location_path
        )
        game_context_for_casting = json.dumps(structured_context_for_casting, ensure_ascii=False, indent=2)

        cast_result = await ai_core.ainvoke_with_rotation(ai_core.scene_casting_chain, {
            "world_settings": ai_core.profile.world_settings or "",
            "current_location_path": effective_location_path, 
            "game_context": game_context_for_casting,
            "recent_dialogue": user_input
        })
        
        new_npc_names = await ai_core._add_cast_to_scene(cast_result)
        
        # 如果創建了新 NPC，則刷新上下文
        if new_npc_names:
            final_structured_context = await ai_core._get_structured_context(
                user_input, 
                override_location_path=effective_location_path
            )
            return {"scene_analysis": scene_analysis, "structured_context": final_structured_context}

    return {"scene_analysis": scene_analysis}
# 函式：執行場景與動作分析 (v3.1 - 路由前置分析)



    









# 函式：執行 SFW 回合規劃 (v3.3 - 指令統一化)
# 更新紀錄:
# v3.3 (2025-09-05): [重大架構修正] 根據敏感內容在 SFW 路徑中被攔截的報告，回滾了 v3.2 的簡化邏輯。此節點現在會再次負責組裝一個完整的 `system_prompt`，該提示詞整合了 `one_instruction`、`response_style` 和可選的動作模組。這確保了 SFW 路徑在需要時也能獲得最高級別的對抗性指令，以應對平台級內容審查。
# v3.2 (2025-09-05): [重大架構修正] 簡化了此節點的邏輯。
async def planning_node(state: ConversationGraphState) -> Dict[str, TurnPlan]:
    """
    [SFW 路徑核心] SFW 架構的核心“思考”節點。
    準備完整的上下文和系統指令，並調用 planning_chain 生成結構化的行動計劃。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    
    logger.info(f"[{user_id}] (Graph) Node: planning_node -> 正在為 SFW 規劃鏈準備材料...")

    # 步驟 1: 強制刷新結構化上下文
    try:
        structured_context = await ai_core._get_structured_context(user_input)
        state['structured_context'] = structured_context
    except Exception as e:
        logger.error(f"[{user_id}] 在 planning_node 中刷新上下文失敗: {e}", exc_info=True)
        structured_context = state.get('structured_context', {})

    # 步驟 2: 準備一個包含所有模板所需變數的完整字典
    full_context_dict = {
        "username": ai_core.profile.user_profile.name,
        "ai_name": ai_core.profile.ai_profile.name,
        "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格：平衡的敘事與對話。",
        "world_settings": ai_core.profile.world_settings or "未設定",
        "ai_settings": ai_core.profile.ai_profile.description or "未設定",
        "retrieved_context": state['rag_context'],
        "user_input": user_input,
        "latest_user_input": user_input,
        **(structured_context or {})
    }

    # 步驟 3: [v3.3 核心修正] 動態構建完整的系統指令
    base_system_prompt = ai_core.profile.one_instruction or "錯誤：未加載基礎系統指令。"
    action_module_name = ai_core._determine_action_module(user_input)
    
    system_prompt_parts = [base_system_prompt]
    if action_module_name and action_module_name != "action_sexual_content" and action_module_name in ai_core.modular_prompts:
        module_prompt = ai_core.modular_prompts[action_module_name]
        system_prompt_parts.append("\n\n# --- 動作模組已激活 --- #\n")
        system_prompt_parts.append(module_prompt)
        logger.info(f"[{user_id}] (Graph) SFW 規劃：已準備加載戰術模組 '{action_module_name}'。")

    def safe_format(template: str, data: dict) -> str:
        for key, value in data.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template

    final_system_prompt = safe_format("".join(system_prompt_parts), full_context_dict)
    world_snapshot = safe_format(ai_core.world_snapshot_template, full_context_dict)
    
    # 步驟 4: 調用規劃鏈，傳入所有需要的原始材料
    if not ai_core.planning_chain:
        raise ValueError("Planning chain is not initialized.")
    
    plan = await ai_core.ainvoke_with_rotation(ai_core.planning_chain, {
        "system_prompt": final_system_prompt,
        "world_snapshot": world_snapshot,
        "user_input": user_input,
    })

    return {"turn_plan": plan, "world_snapshot": world_snapshot}
# 函式：執行 SFW 回合規劃 (v3.3 - 指令統一化)




# 函式：生成遠程場景 (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-09-05): [重大功能擴展] 創建此節點作為 SFW 探索路徑的新分支。它的職責是：
#    1. 獲取由 `scene_and_action_analysis_node` 確定的遠程地點。
#    2. 為該遠程地點專門生成一個詳細的上下文情報簡報。
#    3. 調用 `remote_scene_generator_chain` 來生成純粹的小說式場景描述。
async def remote_scene_generation_node(state: ConversationGraphState) -> Dict[str, str]:
    """
    [SFW-遠程觀察分支] 專門用於生成遠程地點的電影式場景描述。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    scene_analysis = state['scene_analysis']
    
    if not (scene_analysis and scene_analysis.target_location_path):
        logger.error(f"[{user_id}] 進入了 remote_scene_generation_node，但 scene_analysis 中沒有 target_location_path。")
        return {"llm_response": "（系統錯誤：無法確定要觀察的遠程目標。）"}

    target_path = scene_analysis.target_location_path
    logger.info(f"[{user_id}] (Graph) Node: remote_scene_generation_node -> 正在為遠程地點 '{' > '.join(target_path)}' 生成場景...")

    # 為遠程地點生成專門的上下文
    remote_context = await ai_core._get_structured_context(
        user_input="", # 輸入為空，因為我們只關心地點本身
        override_location_path=target_path
    )
    
    # 將結構化上下文格式化為單一字串
    remote_scene_context_str = "\n".join([f"【{k.replace('_context', '').title()}】\n{v}" for k, v in remote_context.items()])

    if not ai_core.remote_scene_generator_chain:
        raise ValueError("Remote Scene Generator chain is not initialized.")
        
    # 調用場景生成鏈
    scene_text = await ai_core.ainvoke_with_rotation(
        ai_core.remote_scene_generator_chain,
        {
            "username": ai_core.profile.user_profile.name,
            "ai_name": ai_core.profile.ai_profile.name,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格：平衡的敘事與對話。",
            "world_settings": ai_core.profile.world_settings or "未設定",
            "remote_scene_context": remote_scene_context_str,
        }
    )
    
    return {"llm_response": scene_text}
# 函式：生成遠程場景 (v1.0 - 全新創建)




# 函式：執行 NSFW 直通生成 (v1.1 - 災難性 KeyError 修正)
# 更新紀錄:
# v1.1 (2025-09-05): [災難性BUG修復] 徹底重構了此節點的參數準備邏輯。現在它不再試圖預先格式化任何提示詞，而是將所有從上下文和 profile 中獲取的原始數據（如 npc_context, ai_name 等）連同 user_input 一起，打包成一個【完整的、未經處理的】字典。這個完整的字典將被直接傳遞給 v1.1 版本的 direct_nsfw_chain，由鏈內部的 RunnablePassthrough.assign() 負責處理所有變數的填充，從而根除了因參數傳遞不完整導致的 KeyError。
# v1.0 (2025-09-05): [全新創建] 創建此節點作為 NSFW 直通路徑的核心。
async def generate_nsfw_response_node(state: ConversationGraphState) -> Dict[str, str]:
    """
    [NSFW 路徑核心] 為 NSFW 指令準備上下文，並直接調用高對抗性的 NSFW 直通鏈生成最終文本。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph) Node: generate_nsfw_response_node -> 正在準備上下文並直接生成 NSFW 回應...")

    # 步驟 1: 準備一個包含所有鏈所需變數的完整上下文大字典
    try:
        structured_context = state.get('structured_context') or await ai_core._get_structured_context(user_input)
        rag_context = state.get('rag_context') or await ai_core._preprocess_rag_context(await ai_core.retriever.ainvoke(user_input))
        
        # [v1.1 核心修正] 創建一個包含所有可能變數的完整字典
        full_context_for_chain = {
            "user_input": user_input,
            "latest_user_input": user_input, # 兼容舊模板
            "username": ai_core.profile.user_profile.name,
            "ai_name": ai_core.profile.ai_profile.name,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格：平衡的敘事與對話。",
            "world_settings": ai_core.profile.world_settings or "未設定",
            "ai_settings": ai_core.profile.ai_profile.description or "未設定",
            "retrieved_context": rag_context,
            **structured_context
        }

    except Exception as e:
        logger.error(f"[{user_id}] 在 generate_nsfw_response_node 中準備上下文失敗: {e}", exc_info=True)
        # 在失敗時提供備援值，以避免後續鏈出錯
        full_context_for_chain = {
            "user_input": user_input, "latest_user_input": user_input,
            "username": "使用者", "ai_name": "AI",
            "response_style_prompt": "", "world_settings": "", "ai_settings": "",
            "retrieved_context": "上下文加載失敗。", "location_context": "", 
            "possessions_context": "", "quests_context": "", "npc_context": "",
            "relevant_npc_context": ""
        }

    if not ai_core.direct_nsfw_chain:
        raise ValueError("Direct NSFW chain is not initialized.")
        
    # 步驟 2: [v1.1 核心修正] 將完整的、未經處理的字典直接傳遞給鏈
    response_text = await ai_core.ainvoke_with_rotation(
        ai_core.direct_nsfw_chain,
        full_context_for_chain
    )

    # 將結果以與 narrative_node 相同的格式返回，以便後續節點處理
    return {"llm_response": response_text}
# 函式：執行 NSFW 直通生成 (v1.1 - 災難性 KeyError 修正)

# 函式：執行工具調用
async def tool_execution_node(state: ConversationGraphState) -> Dict[str, str]:
    """
    [SFW 路徑核心] SFW 架構的核心“執行”節點。在安全的上下文中執行計劃中的所有工具調用。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    plan = state['turn_plan']
    logger.info(f"[{user_id}] (Graph) Node: tool_execution_node -> 正在執行行動計劃中的工具...")
    
    if not plan or not plan.character_actions:
        return {"tool_results": "系統事件：無任何工具被調用。"}
    
    try:
        results_summary = await ai_core._execute_planned_actions(plan)
    except Exception as e:
        logger.error(f"[{user_id}] (Graph) Node: tool_execution_node -> 在執行工具時發生未捕獲的異常: {e}", exc_info=True)
        results_summary = f"系統事件：工具執行時發生嚴重錯誤: {e}"
    finally:
        tool_context.set_context(None, None)
        logger.info(f"[{user_id}] (Graph) Node: tool_execution_node -> 工具上下文已清理。")

    return {"tool_results": results_summary}
# 函式：執行工具調用






# 函式：生成敘事文本 (v2.1 - 職責簡化)
# 更新紀錄:
# v2.1 (2025-09-05): [重大架構修正] 根據 SFW 風格不生效的報告，移除了此節點中所有關於準備和傳遞 `final_output_mandate` 的邏輯。此節點的職責回歸為純粹地將 `turn_plan` 和 `tool_results` 傳遞給 `narrative_chain`。
# v2.0 (2025-09-04): [架構清理] 移除了手動組裝 final_output_mandate 的邏輯。
async def narrative_node(state: ConversationGraphState) -> Dict[str, str]:
    """
    [SFW 路徑核心] SFW 架構的核心“寫作”節點。接收結構化的行動計劃和工具執行結果，並將其渲染成纯粹的小說文本。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    turn_plan = state['turn_plan']
    tool_results = state['tool_results']
    logger.info(f"[{user_id}] (Graph) Node: narrative_node -> 正在处理行动计划...")

    if not turn_plan:
        logger.error(f"[{user_id}] 叙事节点接收到空的行动计划，无法生成回应。")
        return {"llm_response": "（系统错误：未能生成有效的行动计划。）"}

    if turn_plan.execution_rejection_reason:
        logger.info(f"[{user_id}] (Graph) Node: narrative_node -> 检测到拒绝执行的理由，将直接输出。理由: {turn_plan.execution_rejection_reason}")
        return {"llm_response": turn_plan.execution_rejection_reason}

    logger.info(f"[{user_id}] (Graph) Node: narrative_node -> 正在将行动计划和工具结果渲染为小说文本...")
    if not ai_core.narrative_chain:
        raise ValueError("Narrative chain is not initialized.")

    turn_plan.thought += f"\n\n[系统后台执行结果]:\n{tool_results}"

    # [v2.1 核心修正] 不再需要手動組裝和傳遞 final_output_mandate
    narrative_text = await ai_core.ainvoke_with_rotation(
        ai_core.narrative_chain,
        {
            "turn_plan": turn_plan
        }
    )
    
    return {"llm_response": narrative_text}
# 函式：生成敘事文本 (v2.1 - 職責簡化)






# 函式：驗證與淨化輸出
async def validate_and_rewrite_node(state: ConversationGraphState) -> Dict:
    """
    [路徑匯合點] 使用保守且安全的規則，強制淨化 LLM 的原始輸出，同時最大限度地保全有效內容。
    """
    user_id = state['user_id']
    initial_response = state['llm_response']
    logger.info(f"[{user_id}] (Graph) Node: validate_and_rewrite_node -> [已啟用] 正在對 LLM 原始輸出進行內容保全式淨化...")
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
# 函式：驗證與淨化輸出

# 函式：執行狀態更新與記憶儲存
async def persist_state_node(state: ConversationGraphState) -> Dict:
    """
    [收尾] 將本輪對話存入記憶，並將 state_updates 中的變更應用到資料庫。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    clean_response = state['final_output']
    logger.info(f"[{user_id}] (Graph) Node: persist_state_node -> 正在持久化狀態與記憶...")
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
# 函式：執行狀態更新與記憶儲存

# 函式：觸發背景世界擴展
async def background_world_expansion_node(state: ConversationGraphState) -> Dict:
    """
    [收尾] 在回應發送後，根據擴展決策，非阻塞地觸發背景世界擴展、LORE生成等任務。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    clean_response = state['final_output']
    scene_analysis = state.get('scene_analysis') # 使用 .get() 安全访问
    expansion_decision = state.get('expansion_decision')

    logger.info(f"[{user_id}] (Graph) Node: background_world_expansion_node -> 正在檢查是否觸發背景任務...")

    # 只有在守门人允许时才执行
    if expansion_decision and expansion_decision.should_expand:
        effective_location_path = ai_core.profile.game_state.location_path
        if scene_analysis and scene_analysis.target_location_path:
            effective_location_path = scene_analysis.target_location_path
        
        if clean_response and clean_response != "（...）":
            asyncio.create_task(ai_core._background_scene_expansion(user_input, clean_response, effective_location_path))
            logger.info(f"[{user_id}] 已成功為地點 '{' > '.join(effective_location_path)}' 創建背景擴展任務。")
    else:
        logger.info(f"[{user_id}] (Graph) Node: background_world_expansion_node -> 根據決策，本輪跳過背景擴展。")

    return {}
# 函式：觸發背景世界擴展

# 函式：圖形結束 finalizing
async def finalization_node(state: ConversationGraphState) -> Dict:
    """
    [收尾] 一個虛擬的最終節點，確保所有異步背景任務都被成功調度。
    """
    user_id = state['user_id']
    logger.info(f"[{user_id}] (Graph) Node: finalization_node -> 對話流程圖執行完畢。")
    return {}
# 函式：圖形結束 finalizing

# --- 主對話圖的路由 ---





# 函式：[SFW路由] 在擴展決策後決定流程 (v2.0 - 遠程觀察分支)
# 更新紀錄:
# v2.0 (2025-09-05): [重大功能擴展] 重構了此路由的邏輯。現在，它會優先檢查 `scene_analysis` 來判斷是否為“遠程觀察”指令。如果是，則將流程引導至新增的 `remote_scene` 分支；否則，才繼續執行原有的 LORE 擴展或跳過的判斷。
# v1.0 (2025-09-03): [全新創建] 創建此路由以分離 LORE 擴展和常規互動。
def route_expansion(state: ConversationGraphState) -> Literal["remote_scene", "expand_lore", "skip_expansion"]:
    """
    [SFW路由] 根據意圖，決定是進行遠程場景生成、本地LORE擴展，還是直接跳到核心規劃。
    """
    user_id = state['user_id']
    scene_analysis = state.get("scene_analysis")
    
    # 優先判斷是否為遠程觀察模式
    if scene_analysis and scene_analysis.viewing_mode == 'remote':
        logger.info(f"[{user_id}] (Graph) Router: route_expansion -> 判定為【遠程觀察】，進入場景生成路徑。")
        return "remote_scene"
    
    # 如果不是遠程觀察，則執行舊的 LORE 擴展判斷
    should_expand = state.get("expansion_decision")
    if should_expand and should_expand.should_expand:
        logger.info(f"[{user_id}] (Graph) Router: route_expansion -> 判定為【本地LORE擴展】。")
        return "expand_lore"
    else:
        logger.info(f"[{user_id}] (Graph) Router: route_expansion -> 判定為【跳過LORE擴展】，進入規劃。")
        return "skip_expansion"
# 函式：[SFW路由] 在擴展決策後決定流程 (v2.0 - 遠程觀察分支)

# 函式：[核心路由] 根據意圖決定路徑 (v3.0 - 意圖感知)
# 更新紀錄:
# v3.0 (2025-09-05): [災難性BUG修復] 徹底重構了路由邏輯，引入了對 `input_analysis` 的依賴。現在，路由會優先根據指令的【類型】（描述 vs. 互動）來決策。只有【互動式】指令才會根據敏感詞進入 NSFW 路徑，而【描述性】指令則會被強制導向 SFW 的遠程觀察路徑。此修改從根本上解決了描述性指令被錯誤路由到 NSFW 路徑而導致思考洩漏和場景錯亂的問題。
# v2.0 (2025-09-05): [重大邏輯修正] 路由的判斷依據修改為更寬泛的敏感內容檢測器。
def route_based_on_intent(state: ConversationGraphState) -> Literal["nsfw_path", "remote_scene_path", "sfw_path"]:
    """
    [核心路由] 根據使用者的輸入意圖類型和內容敏感性，決定最終的處理路徑。
    """
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    input_analysis = state.get('input_analysis')

    if not input_analysis:
        logger.error(f"[{user_id}] 核心路由：input_analysis 為空，無法決策。默認進入 SFW 路徑。")
        return "sfw_path"

    # 規則 1: 如果指令類型是描述性 (narration)，無論內容如何，都必須走遠程場景生成路徑
    if input_analysis.input_type == 'narration':
        logger.info(f"[{user_id}] (Graph) Router: 判定為【描述性指令】，強制進入【遠程場景路徑】。")
        return "remote_scene_path"
        
    # 規則 2: 如果指令類型是互動式 (dialogue_or_command)，則檢查其內容敏感性
    is_sensitive = ai_core._is_potentially_sensitive_request(user_input)
    if is_sensitive:
        logger.info(f"[{user_id}] (Graph) Router: 判定為【互動式敏感指令】，進入【NSFW 直通路徑】。")
        return "nsfw_path"
    else:
        logger.info(f"[{user_id}] (Graph) Router: 判定為【常規 SFW 指令】，進入【SFW 工具路徑】。")
        return "sfw_path"
# 函式：[核心路由] 根據意圖決定路徑 (v3.0 - 意圖感知)

# --- 主對話圖的建構器 ---

# 函式：創建主回應圖 (v7.3 - 遠程觀察路徑)
# 更新紀錄:
# v7.3 (2025-09-05): [重大功能擴展] 徹底重構了 SFW 探索路徑。
#    1. [新增節點] 註冊了新的 `remote_scene_generation_node`。
#    2. [路由升級] `route_expansion` 現在是一個三向路由，可以將流程引導至 `remote_scene` (遠程觀察)、`expand_lore` (本地擴展) 或 `skip_expansion` (規劃)。
#    3. [拓撲重構] SFW 探索路徑現在是 `expansion_decision` -> `scene_and_action_analysis` -> `route_expansion`，確保路由時擁有足夠的決策資訊。
#    4. [路徑匯合] 新的遠程場景路徑 (`remote_scene_generation_node`) 會直接連接到 `validate_and_rewrite`，與其他路徑匯合。
def create_main_response_graph() -> StateGraph:
    """
    組裝並編譯主對話流程的 StateGraph，現在採用包含遠程觀察路徑的混合模式架構。
    """
    graph = StateGraph(ConversationGraphState)

    # 註冊所有節點
    graph.add_node("initialize_state", initialize_conversation_state_node)
    graph.add_node("analyze_input", analyze_input_node)
    
    # NSFW 路徑節點
    graph.add_node("generate_nsfw_response", generate_nsfw_response_node)
    
    # SFW 路徑節點
    graph.add_node("expansion_decision", expansion_decision_node)
    graph.add_node("scene_and_action_analysis", scene_and_action_analysis_node)
    graph.add_node("remote_scene_generation", remote_scene_generation_node) # [v7.3 新增]
    graph.add_node("planning", planning_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("narrative", narrative_node)

    # 共享的後續節點
    graph.add_node("validate_and_rewrite", validate_and_rewrite_node)
    graph.add_node("persist_state", persist_state_node)
    graph.add_node("background_expansion", background_world_expansion_node)
    graph.add_node("finalization", finalization_node)

    # 設定圖的入口和初始流程
    graph.set_entry_point("initialize_state")
    graph.add_edge("initialize_state", "analyze_input")

    # 核心的 NSFW/SFW 條件路由
    graph.add_conditional_edges(
        "analyze_input",
        route_based_on_intent,
        {
            "nsfw_path": "generate_nsfw_response",
            "sfw_path": "expansion_decision" 
        }
    )

    # NSFW 路徑流程
    graph.add_edge("generate_nsfw_response", "validate_and_rewrite")

    # SFW 路徑的完整流程
    # 步驟 1: 判斷是否需要擴展
    graph.add_edge("expansion_decision", "scene_and_action_analysis")
    
    # 步驟 2: 分析場景後，進行三向路由
    graph.add_conditional_edges(
        "scene_and_action_analysis",
        route_expansion,
        {
            "remote_scene": "remote_scene_generation", # 新的遠程觀察路徑
            "expand_lore": "planning", # 本地擴展後直接去規劃
            "skip_expansion": "planning"  # 跳過擴展也直接去規劃
        }
    )
    
    # 新的遠程觀察路徑的終點
    graph.add_edge("remote_scene_generation", "validate_and_rewrite")
    
    # 原有的規劃 -> 執行 -> 寫作路徑
    graph.add_edge("planning", "tool_execution")
    graph.add_edge("tool_execution", "narrative")
    graph.add_edge("narrative", "validate_and_rewrite")

    # 所有路徑在此匯合後的共享流程
    graph.add_edge("validate_and_rewrite", "persist_state")
    graph.add_edge("persist_state", "background_expansion")
    graph.add_edge("background_expansion", "finalization")
    graph.add_edge("finalization", END)
    
    return graph.compile()
# 函式：創建主回應圖 (v7.3 - 遠程觀察路徑)







# --- 設定圖 (Setup Graph) 的節點 (保持不變) ---
async def process_canon_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    canon_text = state['canon_text']
    if canon_text:
        logger.info(f"[{user_id}] (Setup Graph) Node: process_canon_node -> 正在處理世界聖經...")
        await ai_core.add_canon_to_vector_store(canon_text)
        await ai_core.parse_and_create_lore_from_canon(None, canon_text, is_setup_flow=True)
        if not await ai_core.initialize():
            raise Exception("在載入世界聖經後重新初始化 AI 核心失敗。")
    return {}

async def complete_profiles_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Setup Graph) Node: complete_profiles_node -> 正在補完角色檔案...")
    completion_prompt = ai_core.get_profile_completion_prompt()
    completion_llm = ai_core.gm_model.with_structured_output(CharacterProfile)
    completion_chain = completion_prompt | completion_llm
    completed_user_profile = await ai_core.ainvoke_with_rotation(completion_chain, {"profile_json": ai_core.profile.user_profile.model_dump_json()})
    completed_ai_profile = await ai_core.ainvoke_with_rotation(completion_chain, {"profile_json": ai_core.profile.ai_profile.model_dump_json()})
    await ai_core.update_and_persist_profile({'user_profile': completed_user_profile.model_dump(), 'ai_profile': completed_ai_profile.model_dump()})
    return {}

async def world_genesis_node(state: SetupGraphState) -> Dict:
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Setup Graph) Node: world_genesis_node -> 正在執行世界創世...")
    genesis_chain = ai_core.get_world_genesis_chain()
    genesis_result = await ai_core.ainvoke_with_rotation(genesis_chain, {"world_settings": ai_core.profile.world_settings, "username": ai_core.profile.user_profile.name, "ai_name": ai_core.profile.ai_profile.name})
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
    logger.info(f"[{user_id}] (Setup Graph) Node: generate_opening_scene_node -> 正在生成開場白...")
    opening_scene = await ai_core.generate_opening_scene()
    if not opening_scene or not opening_scene.strip():
        opening_scene = (f"在一片柔和的光芒中，你和 {ai_core.profile.ai_profile.name} 發現自己身處於一個寧靜的空間裡，故事即將從這裡開始。"
                         "\n\n（系統提示：由於您的設定，AI無法生成更詳細的開場白，但您現在可以開始互動了。）")
    return {"opening_scene": opening_scene}

def create_setup_graph() -> StateGraph:
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
