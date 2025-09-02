# src/graph_state.py 的中文註釋(v2.2 - 依赖修复)
# 更新紀錄:
# v2.2 (2025-09-04): [灾难性BUG修复] 在文件顶部增加了 `from typing import TypedDict`，以解决因缺少依赖而导致的 `NameError: name 'TypedDict' is not defined` 致命启动错误。
# v2.1 (2025-09-04): [代碼嚴謹性] 移除了已廢棄的 `dynamic_prompt` 欄位，確保狀態定義與 LangGraph 的實際數據流完全一致。
# v2.0 (2025-09-03): [災難性BUG修復] 新增了 `expansion_decision` 欄位。

from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage

from .schemas import UserInputAnalysis, SceneAnalysisResult, WorldGenesisResult, TurnPlan, ExpansionDecision
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
