# src/tool_context.py 的中文註釋(v1.1 - 移除循環導入)
# 更新紀錄:
# v1.1 (2025-10-04): [災難性BUG修復] 移除了在 TYPE_CHECKING 區塊中對 ai_core.AILover 的導入，並將類型提示改為 `Any`。此修改旨在打破 'ai_core -> tools -> tool_context -> ai_core' 的導入循環，從根本上解決因模組部分初始化導致的 NameError。
# v1.0 (2025-09-02): [重大架構重構] 創建此檔案以提供一個單一的、全域共享的工具上下文實例。

from typing import Any

# 類別：工具上下文
class ToolContext:
    """
    一個全域上下文管理器，用於在工具執行期間儲存和傳遞 user_id 和 ai_core 實例，
    確保所有工具都能訪問到正確的運行時環境。
    """
    def __init__(self):
        self.user_id: str | None = None
        self.ai_core_instance: "Any" | None = None

    def set_context(self, user_id: str, ai_core_instance: "Any"):
        """設置當前執行的上下文。"""
        self.user_id = user_id
        self.ai_core_instance = ai_core_instance

    def get_user_id(self) -> str:
        """獲取當前的使用者 ID。"""
        if not self.user_id:
            raise ValueError("Tool context user_id is not set.")
        return self.user_id

    def get_ai_core(self) -> "Any":
        """獲取當前的 AI 核心實例。"""
        if not self.ai_core_instance:
            raise ValueError("Tool context ai_core_instance is not set.")
        return self.ai_core_instance
# 類別：工具上下文

# 創建一個全域共享的單一實例
tool_context = ToolContext()
