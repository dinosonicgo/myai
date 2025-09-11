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
async def perceive_scene_node(state: ConversationGraphState) -> Dict:
    """[1] 分析用户输入，推断目标地点和视角模式（近景/远景）。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|1) Node: perceive_scene -> 正在感知场景...")

    if not ai_core.profile:
        logger.error(f"[{user_id}] (Graph|1) ai_core.profile 未加载，无法感知场景。")
        return {"scene_analysis": SceneAnalysisResult(viewing_mode='local', reasoning='错误：AI profile 未加载。', action_summary=user_input)}

    # Plan A: 尝试使用 LLM 进行智能推断
    location_chain = ai_core.get_contextual_location_chain()
    # 为了地点推断，我们需要一个临时的、轻量级的上下文
    scene_context_lores = [lore.content for lore in state.get('raw_lore_objects_for_view_decision', []) if lore.category == 'npc_profile']
    scene_context_json_str = json.dumps(scene_context_lores, ensure_ascii=False, indent=2)

    location_result = await ai_core.ainvoke_with_rotation(
        location_chain, 
        {
            "user_input": user_input,
            "world_settings": ai_core.profile.world_settings or "未设定",
            "scene_context_json": scene_context_json_str
        },
        retry_strategy='euphemize'
    )

    if location_result and location_result.location_path:
        logger.info(f"[{user_id}] (Graph|1) LLM 感知成功。推断出的目标地点: {location_result.location_path}")
        target_location = location_result.location_path
        # 如果 LLM 推断出了一个不同于玩家当前位置的地点，则自动设为远景
        viewing_mode = 'remote' if target_location != ai_core.profile.game_state.location_path else 'local'
    else:
        # Plan B (备援): LLM 失败，启动基于规则的备援推断
        logger.warning(f"[{user_id}] (Graph|1) 场景感知链失败，启动【基于规则的备援推断】。")
        viewing_mode = 'local' # 默认为本地
        target_location = ai_core.profile.game_state.location_path
        
        # 规则 1: 检查远景关键词
        remote_keywords = ["观察", "看看", "描述", "什么样"]
        if any(keyword in user_input for keyword in remote_keywords):
            # 规则 2: 尝试用非 LLM 方式提取地点
            # (这是一个简化的实现，未来可以集成 jieba 等本地库)
            all_locations = await lore_book.get_lores_by_category_and_filter(user_id, 'location_info')
            for loc_lore in all_locations:
                loc_name = loc_lore.content.get('name')
                if loc_name and loc_name in user_input:
                    logger.info(f"[{user_id}] (Graph|1) 备援规则匹配成功: 找到远景关键词和地点 '{loc_name}'。")
                    viewing_mode = 'remote'
                    target_location = loc_lore.key.split(' > ')
                    break
    
    # 更新并持久化游戏状态
    gs = ai_core.profile.game_state
    gs.viewing_mode = viewing_mode
    # 在远景模式下，我们将 remote_target_path 设为我们推断出的目标
    gs.remote_target_path = target_location if viewing_mode == 'remote' else None
    await ai_core.update_and_persist_profile({'game_state': gs.model_dump()})
    
    logger.info(f"[{user_id}] (Graph|1) 场景感知完成。最终视角: '{viewing_mode}', 目标地点: {target_location}")
    
    # SceneAnalysisResult 仍然有用，因为它为下游节点提供了一个统一的场景信息接口
    scene_analysis = SceneAnalysisResult(
        viewing_mode=gs.viewing_mode,
        reasoning=f"场景感知完成。",
        target_location_path=gs.remote_target_path,
        action_summary=user_input
    )
    return {"scene_analysis": scene_analysis}
# 函式：[新] 场景感知节点

# 函式：[新] 记忆与 LORE 查询节点
async def retrieve_and_query_node(state: ConversationGraphState) -> Dict:
    """[2] 清洗用户输入，检索 RAG 记忆，并查询所有相关的 LORE。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    scene_analysis = state['scene_analysis']
    logger.info(f"[{user_id}] (Graph|2) Node: retrieve_and_query -> 正在检索记忆与查询LORE...")

    # 源头清洗
    sanitized_query = user_input
    try:
        literary_chain = ai_core.get_literary_euphemization_chain()
        result = await ai_core.ainvoke_with_rotation(literary_chain, {"dialogue_history": user_input}, retry_strategy='euphemize')
        if result:
            sanitized_query = result
    except Exception:
        logger.warning(f"[{user_id}] (Graph|2) 源头清洗失败，将使用原始输入进行查询。")

    # RAG 检索
    rag_context_str = await ai_core.retrieve_and_summarize_memories(sanitized_query)

    # LORE 查询
    is_remote = scene_analysis.viewing_mode == 'remote'
    lores_from_input = await ai_core._query_lore_from_entities(sanitized_query, is_remote_scene=is_remote)
    
    # 查询当前场景的所有 LORE
    effective_location_path = scene_analysis.target_location_path if is_remote else ai_core.profile.game_state.location_path
    lores_in_scene = await lore_book.get_lores_by_category_and_filter(
        user_id, 'npc_profile', lambda c: c.get('location_path') == effective_location_path
    )
    
    # 合并并去重
    final_lores_map = {lore.key: lore for lore in lores_in_scene}
    for lore in lores_from_input:
        final_lores_map.setdefault(lore.key, lore)
        
    logger.info(f"[{user_id}] (Graph|2) 查询完成。检索到 {len(final_lores_map)} 条相关LORE。")
    
    return {
        "rag_context": rag_context_str,
        "raw_lore_objects": list(final_lores_map.values()),
        "sanitized_query_for_tools": sanitized_query
    }
# 函式：[新] 记忆与 LORE 查询节点

# 函式：[新] LORE 扩展决策与执行节点
async def expansion_decision_and_execution_node(state: ConversationGraphState) -> Dict:
    """[3] 决策是否需要扩展 LORE，如果需要，则立即执行扩展。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    safe_query_text = state['sanitized_query_for_tools']
    raw_lore_objects = state.get('raw_lore_objects', [])
    logger.info(f"[{user_id}] (Graph|3) Node: expansion_decision_and_execution -> 正在决策是否扩展LORE...")

    # Plan A: 尝试使用 LLM 进行决策
    lightweight_lore_json = json.dumps(
        [{"name": lore.content.get("name"), "description": lore.content.get("description")} for lore in raw_lore_objects if lore.category == 'npc_profile'],
        ensure_ascii=False
    )
    decision_chain = ai_core.get_expansion_decision_chain()
    decision = await ai_core.ainvoke_with_rotation(
        decision_chain, 
        {"user_input": safe_query_text, "existing_characters_json": lightweight_lore_json, "examples": ""},
        retry_strategy='euphemize'
    )

    if not decision:
        # Plan B (备援): LLM 失败，启动基于 LORE 覆盖率的备援决策
        logger.warning(f"[{user_id}] (Graph|3) LORE扩展决策链失败，启动【基于LORE覆盖率的备援决策】。")
        # (这是一个简化的备援实现)
        if len(raw_lore_objects) < 3 and len(safe_query_text) > 15:
            decision = ExpansionDecision(should_expand=True, reasoning="备援：场景中角色较少且用户输入较长，可能需要新角色。")
        else:
            decision = ExpansionDecision(should_expand=False, reasoning="备援：决策链失败，默认不扩展。")

    if not decision.should_expand:
        logger.info(f"[{user_id}] (Graph|3) 决策结果：无需扩展。理由: {decision.reasoning}")
        return {"planning_subjects": [lore.content for lore in raw_lore_objects]}

    # --- 如果需要扩展，则执行扩展 ---
    logger.info(f"[{user_id}] (Graph|3) 决策结果：需要扩展。理由: {decision.reasoning}。正在执行LORE扩展...")
    
    # Plan A: 尝试使用主 casting_chain
    try:
        logger.info(f"[{user_id}] (Graph|3) 扩展 Plan A: 尝试使用主选角链...")
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
            
            if not cast_result: raise Exception("主选角链返回空值")

            created_names = await ai_core._add_cast_to_scene(cast_result)
            logger.info(f"[{user_id}] (Graph|3) 扩展 Plan A 成功，创建了 {len(created_names)} 位新角色。")
            
            # 获取更新后的所有 LORE
            all_lores_after_expansion = await retrieve_and_query_node(state)
            return {"planning_subjects": [lore.content for lore in all_lores_after_expansion.get("raw_lore_objects", [])]}

    except Exception as e:
        # Plan B (备援): 主链失败，启动 Gemini 子任务链备援
        logger.warning(f"[{user_id}] (Graph|3) 扩展 Plan A 失败: {e}。启动【Gemini子任务链备援】...")
        # (这是一个简化的接口，实际逻辑在 ai_core 中)
        newly_created_lores = await ai_core.gemini_subtask_expansion_fallback(safe_query_text)
        if newly_created_lores:
             logger.info(f"[{user_id}] (Graph|3) 子任务链备援成功，创建了 {len(newly_created_lores)} 位新角色。")
             all_current_lores = state.get('raw_lore_objects', [])
             all_current_lores.extend(newly_created_lores)
             return {"planning_subjects": [lore.content for lore in all_current_lores]}
        else:
             logger.error(f"[{user_id}] (Graph|3) 子任务链备援最终失败。")
             return {"planning_subjects": [lore.content for lore in raw_lore_objects]}

# 函式：[新] 前置工具调用节点
async def preemptive_tool_call_node(state: ConversationGraphState) -> Dict:
    """[4] (全新) 判断并执行用户指令中明确的、需要改变世界状态的动作。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    logger.info(f"[{user_id}] (Graph|4) Node: preemptive_tool_call -> 正在解析前置工具调用...")

    # Plan A: 尝试使用 LLM 解析工具调用
    tool_parsing_chain = ai_core.get_preemptive_tool_parsing_chain()
    tool_call_plan = await ai_core.ainvoke_with_rotation(
        tool_parsing_chain,
        {"user_input": user_input, "character_list_str": ", ".join([ps.get("name", "") for ps in state.get("planning_subjects", [])])},
        retry_strategy='euphemize'
    )
    
    # 简单的备援：如果 LLM 无法解析，就认为没有工具调用
    if not tool_call_plan or not tool_call_plan.plan:
        logger.info(f"[{user_id}] (Graph|4) 未解析到明确的工具调用。")
        return {"tool_results": "系統事件：無前置工具被調用。"}

    logger.info(f"[{user_id}] (Graph|4) 解析到 {len(tool_call_plan.plan)} 个工具调用，准备执行...")
    
    # 执行工具
    tool_context.set_context(user_id, ai_core)
    try:
        # 这是一个简化的 TurnPlan，只用于工具执行
        from .schemas import TurnPlan, CharacterAction
        simple_turn_plan = TurnPlan(
            character_actions=[CharacterAction(character_name="system", reasoning="preemptive", tool_call=call) for call in tool_call_plan.plan]
        )
        results_summary = await ai_core._execute_planned_actions(simple_turn_plan)
    except Exception as e:
        logger.error(f"[{user_id}] (Graph|4) 前置工具执行时发生错误: {e}", exc_info=True)
        results_summary = f"系統事件：工具執行時發生嚴重錯誤: {e}"
    finally:
        tool_context.set_context(None, None)
    
    logger.info(f"[{user_id}] (Graph|4) 前置工具执行完毕。")
    return {"tool_results": results_summary}
# 函式：[新] 前置工具调用节点

# 函式：[新] 世界快照组装节点
async def assemble_world_snapshot_node(state: ConversationGraphState) -> Dict:
    """[5] (核心) 汇集所有信息，使用模板格式化成最终的 world_snapshot 字符串。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    logger.info(f"[{user_id}] (Graph|5) Node: assemble_world_snapshot -> 正在组装最终上下文...")
    
    # 确保我们有最新的 LORE 列表（可能已在扩展节点中更新）
    planning_subjects = state.get("planning_subjects", [])
    
    # 格式化 LORE 供模板使用
    npc_context_str = "\n".join([f"- **{npc.get('name', '未知NPC')}**: {npc.get('description', '无描述')}" for npc in planning_subjects])
    if not npc_context_str: npc_context_str = "当前场景没有已知的特定角色。"

    gs = ai_core.profile.game_state
    
    # 构建填充模板所需的所有变量
    context_vars = {
        'world_settings': ai_core.profile.world_settings or "未設定",
        'ai_settings': ai_core.profile.ai_profile.description or "未設定",
        'retrieved_context': state.get('rag_context', '无'),
        'possessions_context': f"团队库存: {', '.join(gs.inventory) or '空的'}",
        'quests_context': "当前无任务。", # 简化
        'location_context': f"当前地点: {' > '.join(gs.location_path)}",
        'npc_context': npc_context_str,
        'relevant_npc_context': "请参考上方在场角色列表。",
        'player_location': " > ".join(gs.location_path),
        'viewing_mode': gs.viewing_mode,
        'remote_target_path_str': " > ".join(gs.remote_target_path) if gs.remote_target_path else "未指定",
    }
    
    final_world_snapshot = ai_core.world_snapshot_template.format(**context_vars)
    
    logger.info(f"[{user_id}] (Graph|5) 最终上下文组装完毕。")
    return {"world_snapshot": final_world_snapshot}
# 函式：[新] 世界快照组装节点

# 函式：[新] 最终生成节点
async def final_generation_node(state: ConversationGraphState) -> Dict:
    """[6] (全新) 单一的最终生成节点，一步到位创作小说。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    world_snapshot = state['world_snapshot']
    user_input = state['messages'][-1].content
    tool_results = state['tool_results']
    logger.info(f"[{user_id}] (Graph|6) Node: final_generation -> 启动最终生成流程...")

    # 将工具执行结果附加到 world_snapshot 中，作为最新的“事实”
    full_context_for_generation = f"{world_snapshot}\n\n--- 【本回合即時事件】 ---\n{tool_results}\n---"

    generation_chain = ai_core.get_unified_generation_chain()
    
    final_response = await ai_core.ainvoke_with_rotation(
        generation_chain,
        {
            "core_protocol_prompt": ai_core.core_protocol_prompt,
            "final_world_snapshot": full_context_for_generation,
            "latest_user_input": user_input,
            "response_style_prompt": ai_core.profile.response_style_prompt or "預設風格"
        },
        retry_strategy='force' # 这是唯一使用 force 的地方
    )

    if not final_response:
        # 备援：返回固定的中文错误消息
        logger.critical(f"[{user_id}] (Graph|6) 核心生成链在指令轰炸后最终失败！")
        final_response = "（抱歉，我好像突然断线了，脑海中一片空白... 这很可能是因为您的指令触发了无法绕过的核心内容安全限制，或者是一个暂时的、严重的 API 服务问题。请尝试用完全不同的方式表达您的意图，或稍后再试。）"
        
    logger.info(f"[{user_id}] (Graph|6) 最终生成流程完成。")
    return {"llm_response": final_response}
# 函式：[新] 最终生成节点

# 函式：验证、学习与持久化节点
async def validate_and_persist_node(state: ConversationGraphState) -> Dict:
    """[7] 清理文本、事后 LORE 提取、保存对话历史。"""
    user_id = state['user_id']
    ai_core = state['ai_core']
    user_input = state['messages'][-1].content
    llm_response = state['llm_response']
    logger.info(f"[{user_id}] (Graph|7) Node: validate_and_persist -> 正在验证、学习与持久化...")

    # 1. 验证与清理
    # (简化的清理逻辑，可以根据需要扩展)
    clean_response = llm_response.strip()
    
    # 2. 学习 (事后 LORE 提取)
    try:
        logger.info(f"[{user_id}] (Graph|7) 正在启动事后 LORE 学习...")
        lore_extraction_chain = ai_core.get_lore_extraction_chain()
        # 这是一个简化的备援
        if lore_extraction_chain:
             # 使用 ainvoke 而不是 ainvoke_with_rotation 来避免循环依赖
            extraction_plan = await lore_extraction_chain.ainvoke({
                "existing_lore_summary": "", # 简化
                "user_input": user_input,
                "final_response_text": clean_response
            })
            if extraction_plan and extraction_plan.plan:
                logger.info(f"[{user_id}] (Graph|7) 事后学习到 {len(extraction_plan.plan)} 条新 LORE，正在后台保存...")
                # 在后台执行，不阻塞主流程
                asyncio.create_task(ai_core._execute_tool_call_plan(extraction_plan, ai_core.profile.game_state.location_path))
    except Exception as e:
        logger.warning(f"[{user_id}] (Graph|7) 事后 LORE 学习失败，已跳过。核心对话保存不受影响。错误: {e}")

    # 3. 持久化
    if clean_response and "抱歉" not in clean_response:
        chat_history_manager = ai_core.session_histories.setdefault(user_id, ChatMessageHistory())
        chat_history_manager.add_user_message(user_input)
        chat_history_manager.add_ai_message(clean_response)
        
        last_interaction_text = f"使用者: {user_input}\n\nAI:\n{clean_response}"
        # 异步保存到数据库和向量库
        asyncio.create_task(ai_core._save_interaction_to_dbs(last_interaction_text))

    logger.info(f"[{user_id}] (Graph|7) 状态持久化完成。")
    return {"final_output": clean_response}
# 函式：验证、学习与持久化节点

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
