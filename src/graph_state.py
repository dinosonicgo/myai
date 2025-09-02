# src/graph_state.py 的中文註釋(v1.4 - 移除冗餘欄位)
# 更新紀錄:
# v1.4 (2025-09-02): [架構清理] 移除了已廢棄的 `dynamic_prompt` 欄位。在新的“思考->執行->寫作”架構中，提示詞組合已內化至各個節點，不再需要一個統一的預組合提示詞欄位。此修改使狀態定義與最終的圖形架構完全一致。
# v1.3 (2025-09-02): [架構重構] 新增了 `tool_results` 欄位，用於儲存“執行”節點的結果。
# v1.2 (2025-09-02): [架構重構] 新增了 `turn_plan` 欄位，作為“思考”節點的核心數據載體。

from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage

from .schemas import UserInputAnalysis, SceneAnalysisResult, WorldGenesisResult, TurnPlan
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
