# src/graph_state.py 的中文註釋(v5.0 - 適配精細化節點)
# 更新紀錄:
# v5.0 (2025-09-10): [重大架構升級] 新增了 `raw_lore_objects` 欄位。此欄位是實現“LORE查詢”與“上下文組裝”節點分離的關鍵，它作為一個數據載體，負責在兩個新節點之間傳遞原始的、未經處理的 LORE 資料庫對象列表。
# v4.0 (2025-09-06): [重大架構升級] 新增了 `style_analysis` 欄位。
# v3.0 (2025-09-06): [重大架構升級] 新增了 `intent_classification` 欄位。

from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage

from .schemas import (UserInputAnalysis, SceneAnalysisResult, WorldGenesisResult, 
                      TurnPlan, ExpansionDecision, IntentClassificationResult, StyleAnalysisResult)
from .ai_core import AILover

# 類別：對話圖狀態
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
    
    # --- [v21.0 新增] 精細化節點的數據載體 ---
    raw_lore_objects: List[Any] # 用於在 query_lore 和 assemble_context 之間傳遞原始LORE對象

    # --- 中間處理結果 ---
    intent_classification: Optional["IntentClassificationResult"]
    style_analysis: Optional["StyleAnalysisResult"]
    input_analysis: Optional["UserInputAnalysis"]
    expansion_decision: Optional["ExpansionDecision"]
    scene_analysis: Optional["SceneAnalysisResult"]
    rag_context: str
    structured_context: Dict[str, str]
    world_snapshot: str
    
    # "規劃-渲染"模式的核心數據載體
    turn_plan: Optional["TurnPlan"]
    tool_results: str

    # 最終輸出與狀態變更
    llm_response: str
    final_output: str
    state_updates: Dict[str, Any]
# 類別：對話圖狀態

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
