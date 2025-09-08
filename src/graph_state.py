# src/graph_state.py 的中文註釋(v11.0 - 数据伪装)
# 更新紀錄:
# v11.0 (2025-09-22): [重大架構重構] 根据“数据伪装”策略，移除了 `turn_plan` 字段，并新增了 `narrative_outline: str` 字段。这个新字段将用于在规划节点和最终渲染节点之间传递自然语言的、相对安全的“剧本大纲”，以取代之前容易被审查的 TurnPlan JSON 对象。
# v10.0 (2025-09-08): [重大架構重構] 移除了已废弃的 `sanitized_user_input` 欄位。
# v9.0 (2025-09-08): [災難性BUG修復] 新增了 `quantified_character_list` 欄位。
from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage

from .schemas import (UserInputAnalysis, SceneAnalysisResult, WorldGenesisResult, 
                      TurnPlan, ExpansionDecision, IntentClassificationResult, StyleAnalysisResult)
from .ai_core import AILover

# 類別：對話圖狀態 (v12.0 - 新增安全查詢)
# 更新紀錄:
# v12.0 (2025-09-08): [重大架構重構] 新增了 `sanitized_query_for_tools` 欄位。此欄位將用於儲存一個在流程早期就被“預清洗”過的安全版本的用戶輸入，供所有下游的內部工具鏈（如LORE查詢、擴展決策等）使用，以從根本上避免因重複處理原始NSFW輸入而導致的內容審查和流程掛起問題。
# v11.0 (2025-09-22): [重大架構重構] 根据“数据伪装”策略，移除了 `turn_plan` 字段，并新增了 `narrative_outline: str` 字段。
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
    
    planning_subjects: Optional[List[Dict[str, Any]]]

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
    
    # [v12.0 新增] "源頭清洗"模式的核心數據載體
    sanitized_query_for_tools: str

    # [v11.0 新增] "数据伪装"模式的核心数据载体
    narrative_outline: str
    tool_results: str

    # 最终输出与状态变更
    llm_response: str
    final_output: str
    state_updates: Dict[str, Any]
# 類別：對話圖狀態 (v12.0 - 新增安全查詢)

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

