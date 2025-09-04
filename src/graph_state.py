# src/graph_state.py 的中文註釋(v3.0 - 新增意圖分類)
# 更新紀錄:
# v3.0 (2025-09-06): [重大架構升級] 新增了 `intent_classification` 欄位。這是為了支持“先分類，後處理”的新圖架構，該欄位將儲存圖入口點的意圖分類結果，並指導主路由器的決策。
# v2.2 (2025-09-04): [灾难性BUG修复] 在文件顶部增加了 `from typing import TypedDict`。
# v2.1 (2025-09-04): [代碼嚴謹性] 移除了已廢棄的 `dynamic_prompt` 欄位。

from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage

from .schemas import UserInputAnalysis, SceneAnalysisResult, WorldGenesisResult, TurnPlan, ExpansionDecision, IntentClassificationResult
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
    # [v3.0 新增] 儲存初始的意圖分類結果
    intent_classification: Optional["IntentClassificationResult"]
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
