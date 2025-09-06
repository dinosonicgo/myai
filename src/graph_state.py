# src/graph_state.py 的中文註釋(v7.1 - 完整性還原)
# 更新紀錄:
# v7.1 (2025-09-18): [災難性BUG修復] 再次恢復了被先前版本錯誤省略的 SetupGraphState 類別的完整程式碼，以解決 ImportError。
# v7.0 (2025-09-18): [重大架構升級] 新增了 `planning_subjects` 欄位。
# v6.1 (2025-09-18): [災難性BUG修復] 恢復了被先前版本錯誤省略的 SetupGraphState 類別的完整程式碼。

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
    
    # --- 精細化節點的數據載體 ---
    raw_lore_objects: List[Any]
    
    # [v7.0 新增] 強制綁定給規劃器的核心角色LORE
    planning_subjects: Optional[List[Dict[str, Any]]]

    # --- 中間處理結果 ---
    intent_classification: Optional["IntentClassificationResult"]
    style_analysis: Optional["StyleAnalysisResult"]
    input_analysis: Optional["UserInputAnalysis"]
    expansion_decision: Optional["ExpansionDecision"]
    scene_analysis: Optional["SceneAnalysisResult"]
    rag_context: str
    structured_context: Dict[str, str]
    world_snapshot: str
    
    sanitized_user_input: Optional[str]

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
