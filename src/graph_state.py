# src/graph_state.py 的中文註釋(v1.1 - 循環依賴修復)
# 更新紀錄:
# v1.1 (2025-08-31):
# 1. [災難性BUG修復] 將 `ai_core` 欄位的類型提示從 `"AILover"` 修改為 `Any`。
# 2. [健壯性] 此修改徹底解決了 `ai_core.py` 和 `graph_state.py` 之間的運行時循環導入問題，從而修復了導致程式無法啟動的 `NameError`。
# v1.0 (2025-08-31):
# 1. [全新創建] 根據 LangGraph 重構藍圖，創建此檔案以集中定義所有 StateGraph 的核心狀態物件。

from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage

# [v1.1 修正] 為了避免循環導入，我們不再從 .ai_core 導入類型
# 而是使用 Any 來表示 ai_core 實例。
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
from .schemas import UserInputAnalysis, SceneAnalysisResult, WorldGenesisResult
from .ai_core import AILover

# 類別：對話圖狀態
# 說明：定義了主對話流程 (MainResponseGraph) 中，在各個節點之間傳遞的核心數據容器。
class ConversationGraphState(TypedDict):
    """
    主對話流程的狀態容器。
    它像一個數據托盤，在圖形中的每個處理節點之間流動，攜帶著所有必要的資訊。
    """
    # --- 核心標識符 ---
    user_id: str
    """當前使用者的唯一ID。"""
    
    ai_core: "AILover"
    """AI 核心實例的引用，提供對模型、資料庫連接、工具等的訪問。"""

    # --- 核心對話數據 ---
    messages: List[BaseMessage]
    """LangChain 格式的聊天歷史記錄，是圖形的主要輸入和輸出。"""
    
    # --- 中間處理結果 ---
    input_analysis: Optional["UserInputAnalysis"]
    """儲存對使用者最新輸入的意圖分析結果。"""
    
    scene_analysis: Optional["SceneAnalysisResult"]
    """儲存對場景視角（本地/遠程）的分析結果。"""
    
    rag_context: str
    """經過 RAG 檢索和預處理後的上下文文本。"""
    
    structured_context: Dict[str, str]
    """遊戲狀態、地點、NPC 等結構化上下文的字典。"""

    dynamic_prompt: str
    """根據當前情境（如移動、性愛）為 LLM 動態組合的最終系統提示詞。"""
    
    llm_response: str
    """從核心 LLM 生成的、未經處理的原始回應。"""

    # --- 最終輸出與狀態變更 ---
    final_output: str
    """經過驗證、重寫和淨化後，準備發送給使用者的最終文本。"""
    
    state_updates: Dict[str, Any]
    """一個字典，暫存所有需要被持久化到資料庫的狀態變更（例如，好感度、物品欄、金錢等）。"""
# 類別：對話圖狀態

# 類別：設定圖狀態
# 說明：定義了 /start 創世流程 (SetupGraph) 中，在各個節點之間傳遞的核心數據容器。
class SetupGraphState(TypedDict):
    """
    /start 創世流程的狀態容器。
    負責在創世的各個步驟（處理聖經、補完角色、生成世界）之間傳遞數據。
    """
    # --- 核心標識符 ---
    user_id: str
    """當前使用者的唯一ID。"""

    ai_core: "AILover"
    """AI 核心實例的引用。"""

    # --- 流程數據 ---
    canon_text: Optional[str]
    """使用者上傳的世界聖經（World Canon）的完整文本內容。"""
    
    genesis_result: Optional["WorldGenesisResult"]
    """由 world_genesis_chain 生成的、包含初始地點和 NPC 的結構化結果。"""
    
    opening_scene: str
    """由 generate_opening_scene 函式生成的、最終要發送給使用者的開場白文本。"""
# 類別：設定圖狀態