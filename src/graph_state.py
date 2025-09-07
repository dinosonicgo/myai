# src/graph_state.py 的中文註釋(v10.0 - 修正數據流)
# 更新紀錄:
# v10.0 (2025-09-22): [災難性BUG修復] 在 ConversationGraphState 中正式聲明了 `scene_casting_requirements` 欄位。這個被遺漏的步驟是導致“選角需求單”無法在節點間正確傳遞的根本原因。此修改打通了智能LORE擴展的數據流，從而解決了下游節點因缺少數據而擴展失敗的問題。
# v9.0 (2025-09-22): [重大架構升級] 新增了 `reroll_instruction` 欄位。
# v8.0 (2025-09-06): [重大架構升級] 新增了 `sanitized_user_input` 欄位。

from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage

# [v10.0 核心修正] 導入 SceneCastingRequirements
from .schemas import (UserInputAnalysis, SceneAnalysisResult, WorldGenesisResult, 
                      TurnPlan, ExpansionDecision, IntentClassificationResult, StyleAnalysisResult,
                      SceneCastingRequirements)
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
    reroll_instruction: Optional[str]
    
    # --- 精細化節點的數據載體 ---
    raw_lore_objects: List[Any]
    planning_subjects: Optional[List[Dict[str, Any]]]

    # --- 中間處理結果 ---
    intent_classification: Optional["IntentClassificationResult"]
    
    # [v10.0 核心修正] 新增選角需求單欄位
    scene_casting_requirements: Optional["SceneCastingRequirements"]

    style_analysis: Optional["StyleAnalysisResult"]
    input_analysis: Optional["UserInputAnalysis"]
    expansion_decision: Optional["ExpansionDecision"]
    scene_analysis: Optional["SceneAnalysisResult"]
    rag_context: str
    structured_context: Dict[str, str]
    world_snapshot: str
    
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
