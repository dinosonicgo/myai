# src/graph_state.py

# src/graph_state.py 的中文註釋(v9.0 - 新增量化角色列表)
# 更新紀錄:
# v9.0 (2025-09-08): [災難性BUG修復] 新增了 `quantified_character_list` 欄位。此修改是為了解決“分而治之”LORE擴展策略中的狀態傳遞問題，確保由 `character_quantification_node` 生成的量化角色列表能夠被正確地傳遞到 `lore_expansion_node`。
# v8.0 (2025-09-06): [重大架構升級] 新增了 `sanitized_user_input` 欄位。
# v7.1 (2025-09-18): [災難性BUG修復] 再次恢復了被先前版本錯誤省略的 SetupGraphState 類別的完整程式碼。
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
    
    sanitized_user_input: Optional[str]
    
    # --- 精細化節點的數據載體 ---
    raw_lore_objects: List[Any]
    
    planning_subjects: Optional[List[Dict[str, Any]]]

    # [v9.0 新增] 用於在 LORE 擴展流程中傳遞量化後的角色列表
    quantified_character_list: Optional[List[str]]

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
