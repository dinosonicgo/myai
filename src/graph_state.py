# src/graph_state.py 的中文註釋(v4.0 - 新增風格分析狀態)
# 更新紀錄:
# v4.0 (2025-09-06): [重大架構升級] 新增了 `style_analysis` 欄位。此欄位用於儲存新增的 `style_analysis_node` 的輸出結果，是將風格指令從“軟建議”變為“硬約束”的關鍵數據載體。
# v3.0 (2025-09-06): [重大架構升級] 新增了 `intent_classification` 欄位。
# v2.2 (2025-09-04): [灾难性BUG修复] 在文件顶部增加了 `from typing import TypedDict`。

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
    
    # --- 中間處理結果 ---
    intent_classification: Optional["IntentClassificationResult"]
    style_analysis: Optional["StyleAnalysisResult"] # [v4.0 新增]
    input_analysis: Optional["UserInputAnalysis"]
    expansion_decision: Optional["ExpansionDecision"]
    scene_analysis: Optional["SceneAnalysisResult"]
    rag_context: str
    structured_context: Dict[str, str]
    world_snapshot: str
    
    # 新架構的核心數據載體
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
