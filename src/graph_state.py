# src/graph_state.py 的中文註釋(v14.0 - 永久性轟炸架構)
# 更新紀錄:
# v14.0 (2025-10-15): [架構簡化] 移除了 `current_intent` 欄位，因為已採用永久性轟炸策略，不再需要意圖分類。
# v13.0 (2025-10-15): [健壯性] 新增了 `last_response_text` 欄位，用於在連續性指令下實現無損的上下文傳遞（劇情錨點）。
# v12.0 (2025-10-07): [重大架構重構] 根據全新的「資訊注入式架構」，徹底重構了此狀態。
from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage

from .schemas import (SceneAnalysisResult, WorldGenesisResult)
from .ai_core import AILover

# 類別：對話圖狀態 (v16.0 - 新增重生成计数器)
# 更新纪录:
# v16.0 (2025-10-13): [健壮性] 新增了 `regeneration_count` 栏位，用于在“违规审查”循环中计数，防止因AI持续生成违规内容而导致的无限循环。
# v15.0 (2025-10-03): [重大架構重構] 根據全新的「永久性轟炸」與「直接RAG」架構，徹底重構了此狀態。
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

    # --- 上下文快照與恢復 ---
    last_response_text: Optional[str]

    # --- 資訊彙總階段的數據載體 ---
    world_snapshot: str
    
    # --- [v16.0 新增] 违规审查与重生成循环 ---
    regeneration_count: int

    # --- 最終生成與輸出的數據載體 ---
    llm_response: str
    final_output: str
# 類別：對話圖狀態 結束
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



