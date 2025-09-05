# ai_core.py 的中文註釋(v210.0 - “規劃-渲染”統一架構重構)
# 更新紀錄:
# v210.0 (2025-09-09): [重大架構重構] 根據“極致準確”藍圖，對核心生成流程進行了根本性重構。
#    1. [“規劃-渲染”統一化] 廢棄了所有“一步到位”的生成鏈 (如 get_direct_nsfw_chain)，將所有 SFW、NSFW、遠景路徑的生成邏輯統一為“規劃-渲染”兩階段模式。
#    2. [新增專用規劃器] 創建了新的 get_nsfw_planning_chain 和 get_remote_planning_chain，它們只負責輸出結構化的 TurnPlan JSON，將決策與寫作徹底分離。
#    3. [強化渲染器] 強化了 get_narrative_chain，使其成為能夠處理所有類型 TurnPlan 的統一“小說家”節點。
#    4. [預備節點拆分] 新增了 retrieve_and_summarize_memories, _query_lore_from_entities, _assemble_context_from_lore 等輔助函式，為下一步將 graph.py 中的初始化節點拆分為更細粒度的獨立節點做好了準備。
# v209.0 (2025-09-06): [架構適配] 適配了 v209.0 版本的、更簡化的最終安全網委婉化策略。
# v208.0 (2025-09-06): [災難性BUG修復] 徹底重寫此函式，實現最終的“程序化解構-重構”策略。

import re
import json
import time
import shutil
import warnings
import datetime
from typing import List, Dict, Optional, Any, Literal, Callable
import asyncio
import gc
from pathlib import Path
from sqlalchemy import select, or_
from collections import defaultdict
import functools

from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable, DeadlineExceeded

from langchain_google_genai import (
    ChatGoogleGenerativeAI, 
    GoogleGenerativeAIEmbeddings,
    HarmCategory,
    HarmBlockThreshold
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import BooleanOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel, RunnableBinding, RunnableLambda
from langchain_core.documents import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core._api.deprecation import LangChainDeprecationWarning
from pydantic import BaseModel, Field, ValidationError, field_validator
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from chromadb.errors import InternalError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from Levenshtein import ratio as levenshtein_ratio

from pydantic import BaseModel, Field

from . import tools, lore_tools, lore_book
from .lore_book import add_or_update_lore as db_add_or_update_lore, get_lores_by_category_and_filter, Lore
from .models import UserProfile, PersonalMemoryEntry, GameState, CharacterProfile
from .schemas import (WorldGenesisResult, ToolCallPlan, CanonParsingResult, 
                      BatchResolutionPlan, TurnPlan, ToolCall, SceneCastingResult, 
                      UserInputAnalysis, SceneAnalysisResult, ValidationResult, ExtractedEntities, ExpansionDecision,
                      IntentClassificationResult, StyleAnalysisResult, CharacterAction)
from .database import AsyncSessionLocal, UserData, MemoryData
from src.config import settings
from .logger import logger
from .tool_context import tool_context

# 全局常量：Gemini 安全閥值設定
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

PROJ_DIR = Path(__file__).resolve().parent.parent

# 類別：AI核心類
class AILover:
    MODEL_NAME = "models/gemini-2.5-flash-lite"

    # 函式：初始化AI核心
    def __init__(self, user_id: str):
        self.user_id: str = user_id
        self.profile: Optional[UserProfile] = None
        self.gm_model: Optional[Runnable] = None
        
        # 鏈初始化
        self.intent_classification_chain: Optional[Runnable] = None
        self.style_analysis_chain: Optional[Runnable] = None
        self.input_analysis_chain: Optional[Runnable] = None
        self.expansion_decision_chain: Optional[Runnable] = None
        self.scene_casting_chain: Optional[Runnable] = None
        self.sfw_planning_chain: Optional[Runnable] = None
        self.nsfw_planning_chain: Optional[Runnable] = None
        self.remote_planning_chain: Optional[Runnable] = None
        self.narrative_chain: Optional[Runnable] = None
        self.output_validation_chain: Optional[Runnable] = None
        self.rewrite_chain: Optional[Runnable] = None
        self.personal_memory_chain: Optional[Runnable] = None
        self.entity_extraction_chain: Optional[Runnable] = None
        self.euphemization_chain: Optional[Runnable] = None
        
        # LORE/Setup 相關鏈
        self.world_genesis_chain: Optional[Runnable] = None
        self.canon_parser_chain: Optional[Runnable] = None
        self.batch_entity_resolution_chain: Optional[Runnable] = None
        self.single_entity_resolution_chain: Optional[Runnable] = None
        self.param_reconstruction_chain: Optional[Runnable] = None
        self.profile_completion_chain: Optional[Runnable] = None
        self.profile_parser_chain: Optional[Runnable] = None
        self.profile_rewriting_chain: Optional[Runnable] = None

        # Prompt 模板
        self.profile_parser_prompt: Optional[ChatPromptTemplate] = None
        self.profile_completion_prompt: Optional[ChatPromptTemplate] = None
        self.profile_rewriting_prompt: Optional[ChatPromptTemplate] = None
        
        self.modular_prompts: Dict[str, str] = {}
        self.world_snapshot_template: str = ""
        
        # 記憶與RAG
        self.session_histories: Dict[str, ChatMessageHistory] = {}
        self.vector_store: Optional[Chroma] = None
        self.retriever: Optional[EnsembleRetriever] = None
        self.embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
        
        # 其他
        self.available_tools: Dict[str, Runnable] = {}
        
        # API 金鑰管理
        self.api_keys: List[str] = settings.GOOGLE_API_KEYS_LIST
        self.current_key_index: int = 0
        if not self.api_keys:
            raise ValueError("未找到任何 Google API 金鑰。")
        
        self.vector_store_path = str(PROJ_DIR / "data" / "vector_stores" / self.user_id)
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
    # 函式：初始化AI核心

    # 函式：創建一個原始的 LLM 實例
    def _create_llm_instance(self, temperature: float = 0.7) -> ChatGoogleGenerativeAI:
        """創建並返回一個原始的 ChatGoogleGenerativeAI 實例，並自動輪換到下一個 API 金鑰以實現負載均衡。"""
        key_to_use = self.api_keys[self.current_key_index]
        llm = ChatGoogleGenerativeAI(
            model=self.MODEL_NAME,
            google_api_key=key_to_use,
            temperature=temperature,
            safety_settings=SAFETY_SETTINGS,
        )
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.info(f"[{self.user_id}] LLM 實例已使用 API Key #{self.current_key_index} 創建。下一次將使用 Key #{ (self.current_key_index % len(self.api_keys)) + 1 }。")
        return llm
    # 函式：創建一個原始的 LLM 實例
    
    # 函式：初始化AI實例
    async def initialize(self) -> bool:
        async with AsyncSessionLocal() as session:
            result = await session.get(UserData, self.user_id)
            if not result:
                return False
            if not result.one_instruction:
                try:
                    one_instruction_path = PROJ_DIR / "prompts" / "one_instruction_template.txt"
                    with open(one_instruction_path, "r", encoding="utf-8") as f:
                        result.one_instruction = f.read()
                    await session.commit()
                    await session.refresh(result)
                except FileNotFoundError: pass
            if result.username and result.ai_name and (not result.user_profile or not result.ai_profile):
                result.user_profile = CharacterProfile(name=result.username).model_dump()
                result.ai_profile = CharacterProfile(name=result.ai_name, description=result.ai_settings).model_dump()
                await session.commit()
                await session.refresh(result)
            self.profile = UserProfile(
                user_id=result.user_id,
                user_profile=CharacterProfile.model_validate(result.user_profile or {}),
                ai_profile=CharacterProfile.model_validate(result.ai_profile or {}),
                affinity=result.affinity,
                world_settings=result.world_settings,
                one_instruction=result.one_instruction,
                response_style_prompt=result.response_style_prompt,
                game_state=GameState.model_validate(result.game_state or {})
            )
        
        try:
            await self._configure_pre_requisites()
            await self._rehydrate_short_term_memory()
        except Exception as e:
            logger.error(f"[{self.user_id}] 配置前置資源或恢復記憶時發生致命錯誤: {e}", exc_info=True)
            return False
        return True
    # 函式：初始化AI實例

    # 函式：更新並持久化使用者設定檔
    async def update_and_persist_profile(self, updates: Dict[str, Any]) -> bool:
        if not self.profile:
            logger.error(f"[{self.user_id}] 嘗試在未初始化的 profile 上進行更新。")
            return False
        
        try:
            logger.info(f"[{self.user_id}] 接收到 profile 更新請求: {list(updates.keys())}")
            
            profile_dict = self.profile.model_dump()
            
            for key, value in updates.items():
                if key in profile_dict:
                    profile_dict[key] = value

            self.profile = UserProfile.model_validate(profile_dict)

            async with AsyncSessionLocal() as session:
                user_data = await session.get(UserData, self.user_id)
                
                if not user_data:
                    logger.warning(f"[{self.user_id}] 在持久化更新時找不到使用者資料，將創建新記錄。")
                    user_data = UserData(user_id=self.user_id)
                    session.add(user_data)
                    try:
                        with open(PROJ_DIR / "prompts" / "zero_instruction.txt", "r", encoding="utf-8") as f:
                            user_data.one_instruction = f.read()
                    except FileNotFoundError:
                        user_data.one_instruction = "# 預設指令"

                user_data.user_profile = self.profile.user_profile.model_dump()
                user_data.ai_profile = self.profile.ai_profile.model_dump()
                user_data.world_settings = self.profile.world_settings
                user_data.response_style_prompt = self.profile.response_style_prompt
                user_data.game_state = self.profile.game_state.model_dump()
                user_data.affinity = self.profile.affinity
                
                user_data.username = self.profile.user_profile.name
                user_data.ai_name = self.profile.ai_profile.name
                user_data.ai_settings = self.profile.ai_profile.description
                
                await session.commit()
            
            logger.info(f"[{self.user_id}] Profile 更新並持久化成功。")
            return True
        except ValidationError as e:
            logger.error(f"[{self.user_id}] 更新 profile 時發生 Pydantic 驗證錯誤: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"[{self.user_id}] 更新並持久化 profile 時發生未知錯誤: {e}", exc_info=True)
            return False
    # 函式：更新並持久化使用者設定檔

    # 函式：輕量級重建核心模型
    async def _rebuild_agent_with_new_key(self):
        """輕量級地重新初始化所有核心模型，以應用新的 API 金鑰策略（如負載均衡）。"""
        if not self.profile:
            logger.error(f"[{self.user_id}] 嘗試在無 profile 的情況下重建 Agent。")
            return
        logger.info(f"[{self.user_id}] 正在輕量級重建核心模型以應用金鑰策略...")
        self._initialize_models()
        logger.info(f"[{self.user_id}] 核心模型已成功重建。")
    # 函式：輕量級重建核心模型

    # 函式：從資料庫恢復短期記憶
    async def _rehydrate_short_term_memory(self):
        logger.info(f"[{self.user_id}] 正在從資料庫恢復短期記憶...")
        
        if self.user_id not in self.session_histories:
            self.session_histories[self.user_id] = ChatMessageHistory()
        
        chat_history_manager = self.session_histories[self.user_id]
        
        if chat_history_manager.messages:
            logger.info(f"[{self.user_id}] 短期記憶已存在，跳過恢復。")
            return

        async with AsyncSessionLocal() as session:
            stmt = select(MemoryData).where(MemoryData.user_id == self.user_id).order_by(MemoryData.timestamp.desc()).limit(20)
            result = await session.execute(stmt)
            recent_memories = result.scalars().all()
        
        recent_memories.reverse()

        if not recent_memories:
            logger.info(f"[{self.user_id}] 未找到歷史對話記錄，無需恢復記憶。")
            return

        for record in recent_memories:
            try:
                parts = record.content.split("\n\n[場景回應]:\n", 1)
                if len(parts) == 2:
                    user_part, ai_part = parts
                    user_input_match = re.search(r"說: (.*)", user_part, re.DOTALL)
                    if user_input_match:
                        user_input = user_input_match.group(1).strip()
                        ai_response = ai_part.strip()
                        chat_history_manager.add_user_message(user_input)
                        chat_history_manager.add_ai_message(ai_response)
            except Exception as e:
                logger.warning(f"[{self.user_id}] 解析記憶記錄 ID {record.id} 時出錯: {e}")
        
        logger.info(f"[{self.user_id}] 成功恢復了 {len(recent_memories)} 條對話記錄到短期記憶中。")
    # 函式：從資料庫恢復短期記憶

    # 函式：關閉 AI 實例並釋放資源
    async def shutdown(self):
        logger.info(f"[{self.user_id}] 正在關閉 AI 實例並釋放資源...")
        
        if self.vector_store:
            try:
                client = self.vector_store._client
                if client and hasattr(client, '_system') and hasattr(client._system, 'stop'):
                    client._system.stop()
                    logger.info(f"[{self.user_id}] ChromaDB 後台服務已請求停止。")
            except Exception as e:
                logger.warning(f"[{self.user_id}] 關閉 ChromaDB 客戶端時發生非致命錯誤: {e}", exc_info=True)
        
        self.vector_store = None
        self.retriever = None
        gc.collect()
        await asyncio.sleep(1.0)
        
        self.gm_model = None
        # ... 清理所有鏈 ...
        self.session_histories.clear()
        
        logger.info(f"[{self.user_id}] AI 實例資源已釋放。")
    # 函式：關閉 AI 實例並釋放資源

    # ==============================================================================
    # == 📝 Prompt 模板獲取器 v210.0 📝
    # ==============================================================================

    # 函式：獲取角色檔案解析器 Prompt
    def get_profile_parser_prompt(self) -> ChatPromptTemplate:
        if self.profile_parser_prompt is None:
            prompt_str = """你是一位精確的數據分析師。你的任務是將【原始角色JSON】與【使用者的自然語言輸入】相結合，生成一個更新后的、結構化的角色JSON。

【核心規則】
1.  **以原始JSON為基礎**: 你必須以【原始角色JSON】為基礎進行更新。
2.  **智能提取與合併**: 從【使用者的自然語言輸入】中，智能地提取所有關於角色的具體資訊（如名字、性別、年齡、種族、髮型、瞳色、三圍、身高體重、性格、背景故事等），並將這些新資訊填寫或覆蓋到對應的欄位中。
3.  **保留未提及的資訊**: 對於使用者沒有提及的欄位，你必須保留【原始角色JSON】中的原有數值。
4.  **輸出純淨JSON**: 你的最終輸出【必須且只能】是一個更新後的、符合 CharacterProfile Pydantic 格式的 JSON 物件。

---
【原始角色JSON】:
{original_profile_json}
---
【使用者的自然語言輸入】:
{user_text_input}
---"""
            self.profile_parser_prompt = ChatPromptTemplate.from_template(prompt_str)
        return self.profile_parser_prompt
    # 函式：獲取角色檔案解析器 Prompt

    # 函式：將世界聖經添加到向量儲存 (v1.0 - 恢復)
    # 更新紀錄:
    # v1.0 (2025-09-10): [災難性BUG修復] 恢復這個在 v210.0 重構中被意外刪除的核心函式。此函式負責處理世界聖經文本，將其分塊並存入向量數據庫，是 RAG 功能的基礎。
    async def add_canon_to_vector_store(self, text_content: str) -> int:
        """
        將世界聖經（canon）文本內容分割成塊，並將其添加到向量儲存中，用於後續的檢索。
        在添加新內容前，會先清除所有舊的 'canon' 來源數據。
        """
        if not self.vector_store:
            logger.error(f"[{self.user_id}] 嘗試將聖經添加到未初始化的向量儲存中。")
            raise ValueError("Vector store is not initialized.")
            
        try:
            # 步驟 1: 刪除舊的聖經條目以確保數據最新
            collection = await asyncio.to_thread(self.vector_store.get)
            ids_to_delete = [
                doc_id for i, doc_id in enumerate(collection['ids']) 
                if collection['metadatas'][i].get('source') == 'canon'
            ]
            if ids_to_delete:
                await asyncio.to_thread(self.vector_store.delete, ids=ids_to_delete)
                logger.info(f"[{self.user_id}] 已從向量儲存中刪除 {len(ids_to_delete)} 條舊的聖經條目。")

            # 步驟 2: 初始化文本分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200, 
                length_function=len
            )
            
            # 步驟 3: 創建文檔塊
            docs = text_splitter.create_documents([text_content])
            
            # 步驟 4: 將新文檔添加到向量儲存
            if docs:
                await asyncio.to_thread(
                    self.vector_store.add_texts, 
                    texts=[doc.page_content for doc in docs], 
                    metadatas=[{"source": "canon"} for _ in docs]
                )
                logger.info(f"[{self.user_id}] 已成功將聖經文本分割並添加為 {len(docs)} 個知識片段。")
                return len(docs)
                
            return 0
        except Exception as e:
            logger.error(f"[{self.user_id}] 在處理世界聖經並添加到向量儲存時發生錯誤: {e}", exc_info=True)
            raise
    # 函式：將世界聖經添加到向量儲存 (v1.0 - 恢復)

    

    # 函式：獲取角色檔案補完 Prompt
    def get_profile_completion_prompt(self) -> ChatPromptTemplate:
        if self.profile_completion_prompt is None:
            prompt_str = """你是一位资深的角色扮演游戏设定师。你的任务是接收一个不完整的角色 JSON，并将其补完为一个细节豐富、符合逻辑的完整角色。
【核心規則】
1.  **絕對保留原則**: 对于輸入JSON中【任何已經存在值】的欄位（特别是 `appearance_details` 字典內的鍵值對），你【絕對必須】原封不動地保留它們，【絕對禁止】修改或覆蓋。
2.  **增量補完原則**: 你的任務是【只】填寫那些值為`null`、空字符串`""`、空列表`[]`或空字典`{{}}`的欄位。你【必須】基於已有的資訊（如名字、描述、已有的外觀細節），富有創造力地補完【其他缺失的部分】。
3.  **細節豐富化**: 对于 `appearance_details`，如果缺少身高、体重、三围等細節，請基於角色描述進行合理的創造。
4.  **初始裝備**: 对于 `equipment`，如果該欄位為空，請生成一套符合角色背景和描述的初始服裝或裝備。
5.  **輸出格式**: 你的最終輸出【必須且只能】是一個符合 CharacterProfile Pydantic 格式的、補完後的完整 JSON 物件。

【不完整的角色 JSON】:
{profile_json}"""
            self.profile_completion_prompt = ChatPromptTemplate.from_template(prompt_str)
        return self.profile_completion_prompt
    # 函式：獲取角色檔案補完 Prompt

    # 函式：獲取角色檔案重寫 Prompt
    def get_profile_rewriting_prompt(self) -> ChatPromptTemplate:
        if self.profile_rewriting_prompt is None:
            prompt_str = """你是一位技藝精湛的作家和角色編輯。
你的任務是根據使用者提出的【修改指令】，重寫一份【原始的角色描述】。
【核心規則】
1.  **理解並融合**: 你必須深刻理解【修改指令】的核心意圖，並將其無縫地、創造性地融合進【原始的角色描述】中。
2.  **保留精髓**: 在修改的同時，盡力保留角色原有的核心身份和關鍵背景，除非指令明確要求改變它們。你的目標是「演進」角色，而不是「替換」角色。
3.  **輸出純淨**: 你的最終輸出【必須且只能】是重寫後得到的、全新的角色描述文字。禁止包含任何額外的解釋、標題或評論。
---
【原始的角色描述】:
{original_description}
---
【使用者的修改指令】:
{edit_instruction}
---
【重寫後的角色描述】:"""
            self.profile_rewriting_prompt = ChatPromptTemplate.from_template(prompt_str)
        return self.profile_rewriting_prompt
    # 函式：獲取角色檔案重寫 Prompt

    # ==============================================================================
    # == ⚙️ 核心輔助函式 v210.0 ⚙️
    # ==============================================================================
    
    # 函式：加載所有模板檔案
    def _load_templates(self):
        """從 prompts/ 目錄加載所有需要的核心及模組化模板檔案。"""
        try:
            template_path = PROJ_DIR / "prompts" / "world_snapshot_template.txt"
            with open(template_path, "r", encoding="utf-8") as f:
                self.world_snapshot_template = f.read()
            logger.info(f"[{self.user_id}] 核心數據模板 'world_snapshot_template.txt' 已成功加載。")
        except FileNotFoundError:
            logger.error(f"[{self.user_id}] 致命錯誤: 未找到核心數據模板 'world_snapshot_template.txt'！")
            self.world_snapshot_template = "錯誤：世界快照數據模板未找到。"

        self.modular_prompts = {}
        try:
            modular_prompts_dir = PROJ_DIR / "prompts" / "modular"
            if not modular_prompts_dir.is_dir():
                logger.warning(f"[{self.user_id}] 未找到模組化提示詞目錄: {modular_prompts_dir}，將跳過加載。")
                return

            loaded_modules = []
            for prompt_file in modular_prompts_dir.glob("*.txt"):
                module_name = prompt_file.stem
                with open(prompt_file, "r", encoding="utf-8") as f:
                    self.modular_prompts[module_name] = f.read()
                loaded_modules.append(module_name)

            if loaded_modules:
                logger.info(f"[{self.user_id}] 已成功加載 {len(loaded_modules)} 個戰術指令模組: {', '.join(loaded_modules)}")
            else:
                logger.info(f"[{self.user_id}] 在模組化目錄中未找到可加載的戰術指令。")
        except Exception as e:
            logger.error(f"[{self.user_id}] 加載模組化戰術指令時發生未預期錯誤: {e}", exc_info=True)
    # 函式：加載所有模板檔案

    # 函式：[新] 檢索並總結記憶 (用於 retrieve_memories_node)
    async def retrieve_and_summarize_memories(self, user_input: str) -> str:
        """[新] 執行RAG檢索並將結果總結為摘要。這是專門為新的 retrieve_memories_node 設計的。"""
        if not self.retriever:
            logger.warning(f"[{self.user_id}] 檢索器未初始化，無法檢索記憶。")
            return "沒有檢索到相關的長期記憶。"
        
        retrieved_docs = await self.ainvoke_with_rotation(self.retriever, user_input, retry_strategy='euphemize')
        if retrieved_docs is None:
            logger.warning(f"[{self.user_id}] RAG 檢索返回 None (可能因委婉化失敗)，使用空列表作為備援。")
            retrieved_docs = []
            
        if not retrieved_docs:
            return "沒有檢索到相關的長期記憶。"

        summarized_context = await self.ainvoke_with_rotation(
            self.get_rag_summarizer_chain(), 
            retrieved_docs, 
            retry_strategy='euphemize'
        )

        if not summarized_context or not summarized_context.strip():
             logger.warning(f"[{self.user_id}] RAG 總結鏈返回了空的內容（可能因委婉化重試失敗）。")
             summarized_context = "从記憶中檢索到一些相關片段，但無法生成清晰的摘要。"
        
        logger.info(f"[{self.user_id}] 已成功將 RAG 上下文提煉為事實要點。")
        return f"【背景歷史參考（事實要點）】:\n{summarized_context}"
    # 函式：[新] 檢索並總結記憶 (用於 retrieve_memories_node)

    # 函式：[新] 從實體查詢LORE (用於 query_lore_node)
    async def _query_lore_from_entities(self, user_input: str, is_remote_scene: bool = False) -> List[Lore]:
        """[新] 提取實體並查詢其原始LORE對象。這是專門為新的 query_lore_node 設計的。"""
        if not self.profile: return []

        if is_remote_scene:
            text_for_extraction = user_input
        else:
            chat_history_manager = self.session_histories.get(self.user_id, ChatMessageHistory())
            recent_dialogue = "\n".join([f"{'使用者' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in chat_history_manager.messages[-2:]])
            text_for_extraction = f"{user_input}\n{recent_dialogue}"

        entity_extraction_chain = self.get_entity_extraction_chain()
        entity_result = await self.ainvoke_with_rotation(entity_extraction_chain, {"text_input": text_for_extraction})
        extracted_names = set(entity_result.names if entity_result else [])
        
        location_path = self.profile.game_state.location_path
        if not is_remote_scene:
            extracted_names.add(self.profile.user_profile.name)
            extracted_names.add(self.profile.ai_profile.name)
        extracted_names.update(location_path)
        
        logger.info(f"[{self.user_id}] (LORE Querier) 提取到以下關鍵實體: {list(extracted_names)}")

        all_lore_categories = ["npc_profile", "location_info", "item_info", "creature_info", "quest", "world_lore"]
        
        async def find_lore(name: str):
            tasks = [get_lores_by_category_and_filter(self.user_id, category, lambda c: name.lower() in c.get('name', '').lower() or name.lower() in c.get('title', '').lower()) for category in all_lore_categories]
            results_per_name = await asyncio.gather(*tasks, return_exceptions=True)
            return [lore for res in results_per_name if isinstance(res, list) for lore in res]

        query_tasks = [find_lore(name) for name in extracted_names if name]
        all_query_results = await asyncio.gather(*query_tasks, return_exceptions=True)
        
        final_lores = []
        unique_keys = set()
        for result_list in all_query_results:
            if isinstance(result_list, list):
                for lore in result_list:
                    if lore.key not in unique_keys:
                        unique_keys.add(lore.key)
                        final_lores.append(lore)
        
        logger.info(f"[{self.user_id}] (LORE Querier) 查詢到 {len(final_lores)} 條唯一的LORE記錄。")
        return final_lores
    # 函式：[新] 從實體查詢LORE (用於 query_lore_node)

    # 函式：[新] 從LORE組裝上下文 (用於 assemble_context_node)
    def _assemble_context_from_lore(self, raw_lore_objects: List[Lore], is_remote_scene: bool = False) -> Dict[str, str]:
        """[新] 將原始LORE對象和遊戲狀態格式化為最終的上下文簡報。"""
        if not self.profile: return {}
        
        gs = self.profile.game_state
        location_path = gs.location_path
        current_path_str = " > ".join(location_path)
        dossiers = []
        
        if not is_remote_scene:
            dossiers.append(f"--- 檔案: {self.profile.user_profile.name} (使用者角色) ---\n"
                            f"- 描述: {self.profile.user_profile.description}\n...")
            dossiers.append(f"--- 檔案: {self.profile.ai_profile.name} (AI 角色) ---\n"
                            f"- 描述: {self.profile.ai_profile.description}\n...")
        
        for lore in raw_lore_objects:
            content = lore.content
            name = content.get('name') or content.get('title', '未知名稱')
            dossier_content = [f"--- 檔案: {name} ({lore.category}) ---"]
            if 'description' in content: dossier_content.append(f"- 描述: {content['description']}")
            dossiers.append("\n".join(dossier_content))
            
        location_context = f"當前地點: {current_path_str}"
        inventory_context = f"團隊庫存: {', '.join(gs.inventory) or '空的'}" if not is_remote_scene else "（遠程觀察模式）"
        dossier_context = "\n".join(dossiers) if dossiers else "場景中無已知的特定情報。"

        final_context = {
            "location_context": location_context,
            "possessions_context": inventory_context,
            "quests_context": "當前任務: (已整合進情報檔案)",
            "npc_context": dossier_context,
            "relevant_npc_context": ""
        }
        logger.info(f"[{self.user_id}] (Context Assembler) 上下文簡報組裝完畢。")
        return final_context
    # 函式：[新] 從LORE組裝上下文 (用於 assemble_context_node)
    
    # ... (此處保留舊的 _get_structured_context 和 _preprocess_rag_context 作為過渡)

    # ==============================================================================
    # == ⛓️ 鏈的延遲加載 (Lazy Loading) 構建器 v210.0 ⛓️
    # ==============================================================================
    
    # --- 預處理與分析鏈 ---
    
    # 函式：獲取意圖分類鏈
    def get_intent_classification_chain(self) -> Runnable:
        """獲取或創建一個專門用於語意意圖分類的鏈。"""
        if not hasattr(self, 'intent_classification_chain') or self.intent_classification_chain is None:
            classifier_llm = self._create_llm_instance(temperature=0.0).with_structured_output(IntentClassificationResult)
            prompt_template = """你是一位專業的內容策略與意圖分析專家...（省略以保持簡潔）...
---
【使用者輸入】:
{user_input}
---
請開始分析並生成結構化的 JSON 輸出。"""
            prompt = ChatPromptTemplate.from_template(prompt_template)
            self.intent_classification_chain = prompt | classifier_llm
        return self.intent_classification_chain
    # 函式：獲取意圖分類鏈

    # 函式：獲取風格分析鏈
    def get_style_analysis_chain(self) -> Runnable:
        """獲取或創建一個專門用於分析風格指令並將其結構化的鏈。"""
        if not hasattr(self, 'style_analysis_chain') or self.style_analysis_chain is None:
            analysis_llm = self._create_llm_instance(temperature=0.0).with_structured_output(StyleAnalysisResult)
            prompt_template = """你是一位專業的 AI 行為分析師...（省略以保持簡潔）...
---
【AI 回覆風格指令】:
{response_style_prompt}
---
請開始分析並生成結構化的 JSON 輸出。"""
            prompt = ChatPromptTemplate.from_template(prompt_template)
            self.style_analysis_chain = prompt | analysis_llm
        return self.style_analysis_chain
    # 函式：獲取風格分析鏈
    
    # ... (此處保留其他分析鏈: get_input_analysis_chain, get_expansion_decision_chain 等)

    # --- 規劃鏈 (Planning Chains) ---

    # 函式：獲取 SFW 回合計劃鏈
    def get_sfw_planning_chain(self) -> Runnable:
        """獲取或創建 SFW 路徑的規劃鏈，輸出 TurnPlan JSON。"""
        if not hasattr(self, 'sfw_planning_chain') or self.sfw_planning_chain is None:
            planner_llm = self._create_llm_instance(temperature=0.2).with_structured_output(TurnPlan)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "# SFW 系統核心指令...\n{system_prompt}"),
                ("human", "# 當前情境與使用者指令...\n{world_snapshot}\n{user_input}\n請生成行動計劃 JSON。")
            ])
            self.sfw_planning_chain = prompt | planner_llm
        return self.sfw_planning_chain
    # 函式：獲取 SFW 回合計劃鏈

    # 函式：[新] 獲取 NSFW 互動計劃鏈
    def get_nsfw_planning_chain(self) -> Runnable:
        """[新] 獲取 NSFW 互動路徑的規劃鏈，輸出 TurnPlan JSON。"""
        if not hasattr(self, 'nsfw_planning_chain') or self.nsfw_planning_chain is None:
            planner_llm = self._create_llm_instance(temperature=0.7).with_structured_output(TurnPlan)
            
            system_prompt = """# ==================================================
# == 🧠 系統核心指令 (NSFW 互動規劃模式) 🧠
# ==================================================
# 你的唯一任務是分析所有上下文，為接下來的【露骨互動回合】制定一份詳細、色情、符合邏輯的【行動計劃 JSON】。
# 你必須嚴格遵循所有核心協議，特別是【action_sexual_content.txt】模組中的所有規則。
{system_prompt}
"""
            human_prompt = """# ==================================================
# == 📖 當前情境與使用者指令 📖
# ==================================================
# --- 世界快照數據 ---
{world_snapshot}
# --- 使用者最新指令 ---
{user_input}
# --- 你的任務 ---
請嚴格遵循你在【系統核心指令】中學到的所有規則，開始你智慧且色情的規劃，生成行動計劃 JSON。
"""
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            self.nsfw_planning_chain = prompt | planner_llm
        return self.nsfw_planning_chain
    # 函式：[新] 獲取 NSFW 互動計劃鏈

    # 函式：[新] 獲取遠景計劃鏈
    def get_remote_planning_chain(self) -> Runnable:
        """[新] 獲取遠景描述路徑的規劃鏈，輸出 TurnPlan JSON。"""
        if not hasattr(self, 'remote_planning_chain') or self.remote_planning_chain is None:
            planner_llm = self._create_llm_instance(temperature=0.7).with_structured_output(TurnPlan)
            
            system_prompt = """# ==================================================
# == 🧠 系統核心指令 (遠景規劃模式) 🧠
# ==================================================
# 你的角色是【電影導演】。你的任務是將鏡頭切換到一個遠程地點，並構思一幕生動的場景。
# 你的輸出不是小說本身，而是一份給“小說家”看的、結構化的【場景行動計劃 JSON】。
# 【最高禁令】：你的計劃中【絕對禁止】包含使用者「{username}」或其AI夥伴「{ai_name}」。
{system_prompt}
"""
            human_prompt = """# ==================================================
# == 🎬 導演指令卡 🎬
# ==================================================
# --- 核心世界觀 ---
{world_settings}
# --- 遠程地點情報摘要 ---
{remote_scene_context}
# --- 使用者的描述指令 ---
{user_input}
# --- 你的任務 ---
請嚴格遵循所有規則，構思一幕發生在遠程地點的場景，並將其轉化為一份詳細的 TurnPlan JSON。
計畫中的 character_actions 必須包含你為此場景創造的【有名有姓】的NPC。
"""
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            self.remote_planning_chain = prompt | planner_llm
        return self.remote_planning_chain
    # 函式：[新] 獲取遠景計劃鏈
    
    # --- 渲染鏈 (Rendering Chain) ---

    # 函式：獲取統一敘事渲染鏈
    def get_narrative_chain(self) -> Runnable:
        """[強化] 創建一個統一的“小說家”鏈，負責將任何結構化的回合計劃渲染成符合使用者風格的小說文本。"""
        if not hasattr(self, 'narrative_chain') or self.narrative_chain is None:
            system_prompt_template = """你是一位技藝精湛的小說家和敘事者。
你的唯一任務是將下方提供的【回合行動計畫】...（省略以保持簡潔）...
---
【【【最終輸出強制令 (ABSOLUTE & HIGHEST PRIORITY)】】】
{response_style_prompt}
---
"""
            human_prompt_template = """
---
【回合行動計畫 (JSON)】:
{turn_plan_json}
---
【生成的小說場景】:
"""
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt_template),
                ("human", human_prompt_template)
            ])
            self.narrative_chain = (
                {
                    "turn_plan_json": lambda x: x.get("turn_plan").model_dump_json(indent=2) if x.get("turn_plan") else "{}",
                    "response_style_prompt": lambda x: self.profile.response_style_prompt if self.profile else "預設風格"
                }
                | prompt
                | self.gm_model
                | StrOutputParser()
            )
        return self.narrative_chain
    # 函式：獲取統一敘事渲染鏈

    # ... (此處保留所有其他工具、輔助函式和鏈的定義，它們不受此次重構影響)

    # 函式：帶金鑰輪換與委婉化重試的非同步呼叫
    async def ainvoke_with_rotation(self, chain: Runnable, params: Any, retry_strategy: Literal['euphemize', 'force', 'none'] = 'euphemize') -> Any:
        if not self.api_keys:
            raise ValueError("No API keys available.")
        max_retries = len(self.api_keys)
        base_delay = 5
        
        for attempt in range(max_retries):
            try:
                result = await chain.ainvoke(params)
                is_empty_or_invalid = not result or (hasattr(result, 'content') and not getattr(result, 'content', True))
                if is_empty_or_invalid:
                    raise Exception("SafetyError: The model returned an empty or invalid response.")
                return result
            except (ResourceExhausted, InternalServerError, ServiceUnavailable, DeadlineExceeded) as e:
                delay = base_delay * (attempt + 1)
                logger.warning(f"[{self.user_id}] API 遭遇資源或伺服器錯誤: {type(e).__name__}. 將在 {delay:.1f} 秒後使用下一個金鑰重試...")
                await asyncio.sleep(delay)
                self._initialize_models()
            except Exception as e:
                error_str = str(e).lower()
                is_safety_error = "safety" in error_str or "blocked" in error_str or "empty or invalid response" in error_str
                if is_safety_error:
                    if retry_strategy == 'euphemize':
                        return await self._euphemize_and_retry(chain, params)
                    elif retry_strategy == 'force':
                        return await self._force_and_retry(chain, params)
                    else:
                        logger.warning(f"[{self.user_id}] 鏈遭遇內容審查，且重試策略為 'none'。返回 None。")
                        return None
                logger.error(f"[{self.user_id}] 在 ainvoke 期間發生未知錯誤: {e}", exc_info=True)
                raise e

        logger.error(f"[{self.user_id}] 所有 API 金鑰均嘗試失敗。")
        if retry_strategy == 'euphemize':
            return await self._euphemize_and_retry(chain, params)
        elif retry_strategy == 'force':
            return await self._force_and_retry(chain, params)
        return None
    # 函式：帶金鑰輪換與委婉化重試的非同步呼叫

# 類別結束
