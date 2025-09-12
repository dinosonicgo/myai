# src/graph_state.py 的中文註釋(v12.0 - 信息注入式架构)
# 更新紀錄:
# v12.0 (2025-10-07): [重大架構重構] 根據全新的「資訊注入式架構」，徹底重構了此狀態。移除了所有與 `TurnPlan` 相關的欄位，並簡化了中間分析結果，使其完美適配新的線性資訊流。
# v11.0 (2025-09-22): [重大架構重構] 根据“数据伪装”策略，移除了 `turn_plan` 字段，并新增了 `narrative_outline: str` 字段。
# v10.0 (2025-09-08): [重大架構重構] 移除了已废弃的 `sanitized_user_input` 欄位。
from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage

from .schemas import (SceneAnalysisResult, WorldGenesisResult)
from .ai_core import AILover

# 類別：對話圖狀態 (v12.0 - 信息注入式架构)
# 更新紀錄:
# v12.0 (2025-10-07): [重大架構重構] 根據全新的「資訊注入式架構」，徹底重構了此狀態。
# v11.0 (2025-09-22): [重大架構重構] 移除了 `turn_plan` 字段。
class ConversationGraphState(TypedDict):
    """
    主對話流程的狀態容器。
    它像一個數據托盤，在圖形中的每個處理節點之間流動，攜帶著所有必要的資訊。
    """
    # --- 核心標識符 ---
    user_id: str
    ai_core: "AILover"

    # --- 核心對話數據 ---
    messages: List[BaseMessage]
    
    # --- 資訊收集階段的數據載體 ---
    scene_analysis: Optional["SceneAnalysisResult"]
    sanitized_query_for_tools: str
    rag_context: str
    raw_lore_objects: List[Any]
    planning_subjects: Optional[List[Dict[str, Any]]]
    tool_results: str

    # --- 資訊彙總階段的數據載體 ---
    world_snapshot: str

    # --- 最終生成與輸出的數據載體 ---
    llm_response: str
    final_output: str
# 類別：對話圖狀態 (v12.0 - 信息注入式架构)

# 類別：設定圖狀態
class SetupGraphState(TypedDict):
    """
    /start 創世流程的狀態容器。
    負責在創世的各個步驟（處理聖經、補完角色、生成世界）之間傳遞數據。
    """
    user_id: str
    ai_core: "AILover"
    canon_text: Optional[str]
    genesis_result: Optional["WorldGenesisResult"]
    opening_scene: str
# 類別：設定圖狀態
