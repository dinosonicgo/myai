# ai_core.py 的中文註釋(v300.0 - 原生SDK重構整合)
# 更新紀錄:
# v300.0 (2025-11-19): [根本性重構] 根據最新討論，提供了整合所有修正的完整檔案。核心變更包括：徹底拋棄 LangChain 執行層，重構 ainvoke_with_rotation 為原生 SDK 引擎以確保安全閥值生效；將所有 get_..._chain 函式簡化為僅返回 PromptTemplate；並全面改造所有 LLM 呼叫點以適配新引擎。
# v232.0 (2025-11-19): [根本性重構] 徹底重寫 ainvoke_with_rotation，完全拋棄 LangChain 的執行層。
# v225.2 (2025-11-16): [災難性BUG修復] 修正了 __init__ 的縮排錯誤。

import re
import json
import time
import shutil
import warnings
import datetime
from typing import List, Dict, Optional, Any, Literal, Callable, Tuple, Type
import asyncio
import gc
from pathlib import Path
from sqlalchemy import select, or_, delete, update
from collections import defaultdict
import functools

from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable, DeadlineExceeded, GoogleAPICallError
from langchain_google_genai._common import GoogleGenerativeAIError
from google.generativeai.types.generation_types import BlockedPromptException

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
import chromadb
from chromadb.errors import InternalError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from Levenshtein import ratio as levenshtein_ratio

from . import tools, lore_tools, lore_book
from .lore_book import add_or_update_lore as db_add_or_update_lore, get_lores_by_category_and_filter, Lore
from .models import UserProfile, PersonalMemoryEntry, GameState, CharacterProfile
from .schemas import (WorldGenesisResult, ToolCallPlan, CanonParsingResult, 
                      BatchResolutionPlan, TurnPlan, ToolCall, SceneCastingResult, 
                      UserInputAnalysis, SceneAnalysisResult, ValidationResult, ExtractedEntities, 
                      ExpansionDecision, IntentClassificationResult, StyleAnalysisResult, SingleResolutionPlan)
from .database import AsyncSessionLocal, UserData, MemoryData, SceneHistoryData
from src.config import settings
from .logger import logger
from .tool_context import tool_context

# [v1.0] 对话生成模型优先级列表 (从高到低)
# 严格按照此列表顺序进行降级轮换，用于最终的小说生成
GENERATION_MODEL_PRIORITY = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

# [v1.0] 功能性模型
# 用于所有内部的、辅助性的、确定性任务（如：工具解析、实体提取、备援链等）
# 固定使用此模型以保证稳定性和速度
FUNCTIONAL_MODEL = "gemini-2.5-flash-lite"

# 全局常量：Gemini 安全阀值设定
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_NONE,
}

PROJ_DIR = Path(__file__).resolve().parent.parent

# 類別：AI核心類
# 說明：管理單一使用者的所有 AI 相關邏輯，包括模型、記憶、鏈和互動。
class AILover:
    # 函式：初始化AI核心 (v225.2 - 修正縮排)
    # 更新紀錄:
    # v225.2 (2025-11-16): [災難性BUG修復] 修正了函式定義的縮排錯誤，確保其作為 AILover 類別的成員被正確解析。
    # v225.1 (2025-11-16): [功能擴展] 新增 self.last_user_input 屬性，用於儲存使用者最近一次的輸入，以支持「重新生成」功能。
    # v225.0 (2025-11-20): [重大架構升級] 將 self.session_histories 升級為 self.scene_histories。
    def __init__(self, user_id: str):
        self.user_id: str = user_id
        self.profile: Optional[UserProfile] = None
        
        self.model_priority_list: List[str] = GENERATION_MODEL_PRIORITY
        self.current_model_index: int = 0
        self.current_key_index: int = 0
        self.api_keys: List[str] = settings.GOOGLE_API_KEYS_LIST
        if not self.api_keys:
            raise ValueError("未找到任何 Google API 金鑰。")
        
        self.key_cooldowns: Dict[int, float] = {}
        self.key_short_term_failures: Dict[int, List[float]] = defaultdict(list)
        self.RPM_FAILURE_WINDOW = 60
        self.RPM_FAILURE_THRESHOLD = 3

        self.last_context_snapshot: Optional[Dict[str, Any]] = None
        self.last_user_input: Optional[str] = None
        
        # --- 所有 get_..._chain 輔助鏈的佔位符 ---
        # 這些屬性現在只用於緩存 ChatPromptTemplate 物件
        self.canon_parser_chain: Optional[ChatPromptTemplate] = None
        self.batch_entity_resolution_chain: Optional[ChatPromptTemplate] = None
        self.single_entity_resolution_chain: Optional[ChatPromptTemplate] = None
        self.json_correction_chain: Optional[ChatPromptTemplate] = None
        self.world_genesis_chain: Optional[ChatPromptTemplate] = None
        self.profile_completion_prompt: Optional[ChatPromptTemplate] = None
        self.profile_parser_prompt: Optional[ChatPromptTemplate] = None
        self.profile_rewriting_prompt: Optional[ChatPromptTemplate] = None
        self.rag_summarizer_chain: Optional[ChatPromptTemplate] = None
        self.literary_euphemization_chain: Optional[ChatPromptTemplate] = None
        self.lore_extraction_chain: Optional[ChatPromptTemplate] = None
        
        # --- 模板與資源 ---
        self.core_protocol_prompt: str = ""
        self.world_snapshot_template: str = ""
        self.scene_histories: Dict[str, ChatMessageHistory] = {}

        self.vector_store: Optional[Chroma] = None
        self.retriever: Optional[EnsembleRetriever] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
        self.available_tools: Dict[str, Runnable] = {}
        self.gm_model: Optional[ChatGoogleGenerativeAI] = None # 僅用於向下兼容或特定非生成任務
        self.vector_store_path = str(PROJ_DIR / "data" / "vector_stores" / self.user_id)
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
    # 初始化AI核心 函式結束

    # 函式：獲取下一個可用的 API 金鑰 (v2.0 - 冷卻系統)
    # 更新紀錄:
    # v2.0 (2025-10-15): [健壯性] 整合了 API Key 冷卻系統，會自動跳過處於冷卻期的金鑰。
    # v1.0 (2025-10-14): [核心功能] 創建此輔助函式，用於集中管理 API 金鑰的輪換。
    def _get_next_available_key(self) -> Optional[Tuple[str, int]]:
        """
        獲取下一個可用的 API 金鑰及其索引。
        會自動跳過處於冷卻期的金鑰。如果所有金鑰都在冷卻期，則返回 None。
        """
        if not self.api_keys:
            return None
        
        start_index = self.current_key_index
        for i in range(len(self.api_keys)):
            index_to_check = (start_index + i) % len(self.api_keys)
            
            cooldown_until = self.key_cooldowns.get(index_to_check)
            if cooldown_until and time.time() < cooldown_until:
                cooldown_remaining = round(cooldown_until - time.time())
                logger.info(f"[{self.user_id}] [API Key Cooling] 跳過冷卻中的 API Key #{index_to_check} (剩餘 {cooldown_remaining} 秒)。")
                continue
            
            self.current_key_index = (index_to_check + 1) % len(self.api_keys)
            return self.api_keys[index_to_check], index_to_check
        
        logger.warning(f"[{self.user_id}] [API 警告] 所有 API 金鑰當前都處於冷卻期。")
        return None
    # 獲取下一個可用的 API 金鑰 函式結束



    # 函式：創建 LangChain LLM 實例 (v4.0 - 健壯性)
# 更新紀錄:
# v4.0 (2025-11-19): [功能恢復] 根據 AttributeError Log，將此核心輔助函式恢復到 AILover 類中。在原生SDK重構後，此函式仍然為 Embedding 等需要 LangChain 模型的輔助功能提供支持。
# v3.3 (2025-10-15): [健壯性] 設置 max_retries=1 來禁用內部重試。
# v3.2 (2025-10-15): [災難性BUG修復] 修正了因重命名輔助函式後未更新調用導致的 AttributeError。
    def _create_llm_instance(self, temperature: float = 0.7, model_name: str = FUNCTIONAL_MODEL, google_api_key: Optional[str] = None) -> Optional[ChatGoogleGenerativeAI]:
        """
        [輔助功能專用] 創建並返回一個 ChatGoogleGenerativeAI 實例。
        主要用於 Embedding 等仍需 LangChain 模型的非生成性任務。
        如果提供了 google_api_key，則優先使用它；否則，從內部輪換獲取。
        """
        key_to_use = google_api_key
        key_index_log = "provided"
        
        if not key_to_use:
            key_info = self._get_next_available_key()
            if not key_info:
                return None # 沒有可用的金鑰
            key_to_use, key_index = key_info
            key_index_log = str(key_index)
        
        generation_config = {"temperature": temperature}
        
        # 獲取 LangChain 格式的安全設定
        safety_settings_langchain = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        logger.info(f"[{self.user_id}] 正在創建 LangChain 模型 '{model_name}' 實例 (API Key index: {key_index_log})")
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=key_to_use,
            safety_settings=safety_settings_langchain,
            generation_config=generation_config,
            max_retries=1 # 禁用 LangChain 的內部重試，由我們自己的 ainvoke_with_rotation 處理
        )
# 創建 LangChain LLM 實例 函式結束

# 函式：帶有輪換和備援策略的原生 API 調用引擎 (v235.0 - 速率限制優化)
# 更新紀錄:
# v235.0 (2025-11-20): [健壯性強化] 針對 ResourceExhausted (速率限制) 等臨時性 API 錯誤，引入了帶有「指數退避」的內部重試循環。如果一個金鑰在短時間內頻繁觸發速率限制，系統會自動將其長時間「冷卻」，從而極大地提高了在高負載下的請求成功率和系統穩定性。
# v234.0 (2025-11-20): [根本性重構] 移除了對 _rebuild_agent_with_new_key 的調用，實現了徹底的原生化。
# v233.0 (2025-11-19): [根本性重構] 徹底重寫此函式，完全拋棄 LangChain 的執行層。
    async def ainvoke_with_rotation(
        self,
        full_prompt: str,
        output_schema: Optional[Type[BaseModel]] = None,
        retry_strategy: Literal['euphemize', 'force', 'none'] = 'euphemize',
        use_degradation: bool = False
    ) -> Any:
        """
        一個高度健壯的原生 API 調用引擎，整合了金鑰輪換、模型降級、內容審查備援策略，
        並手動處理 Pydantic 結構化輸出，同時內置了針對速率限制的指數退避和金鑰冷卻機制。
        """
        import google.generativeai as genai
        from google.generativeai.types.generation_types import BlockedPromptException
        from google.api_core import exceptions as google_api_exceptions
        import random

        models_to_try = self.model_priority_list if use_degradation else [FUNCTIONAL_MODEL]
        last_exception = None
        IMMEDIATE_RETRY_LIMIT = 3

        for model_index, model_name in enumerate(models_to_try):
            for attempt in range(len(self.api_keys)):
                key_info = self._get_next_available_key()
                if not key_info:
                    logger.warning(f"[{self.user_id}] 在模型 '{model_name}' 的嘗試中，所有 API 金鑰均處於長期冷卻期。")
                    break
                
                key_to_use, key_index = key_info
                
                # [v235.0 核心修正] 內部重試循環，帶有指數退避
                for retry_attempt in range(IMMEDIATE_RETRY_LIMIT):
                    try:
                        genai.configure(api_key=key_to_use)
                        
                        safety_settings_sdk = [
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        ]

                        model = genai.GenerativeModel(model_name=model_name, safety_settings=safety_settings_sdk)
                        
                        response = await asyncio.wait_for(
                            model.generate_content_async(
                                full_prompt,
                                generation_config=genai.types.GenerationConfig(temperature=0.75)
                            ),
                            timeout=120.0
                        )
                        
                        if response.prompt_feedback.block_reason:
                            raise BlockedPromptException(f"Prompt blocked due to {response.prompt_feedback.block_reason.name}")
                        if response.candidates and response.candidates[0].finish_reason not in [1, 'STOP']:
                             finish_reason_name = response.candidates[0].finish_reason.name
                             raise BlockedPromptException(f"Generation stopped due to finish_reason: {finish_reason_name}")

                        raw_text_result = response.text

                        if not raw_text_result or not raw_text_result.strip():
                            raise GoogleGenerativeAIError("SafetyError: The model returned an empty or invalid response.")
                        
                        if output_schema:
                            json_match = re.search(r'\{.*\}|\[.*\]', raw_text_result, re.DOTALL)
                            if not json_match:
                                raise OutputParserException("Failed to find any JSON object in the response.", llm_output=raw_text_result)
                            clean_json_str = json_match.group(0)
                            return output_schema.model_validate(json.loads(clean_json_str))
                        else:
                            return raw_text_result

                    except (BlockedPromptException, GoogleGenerativeAIError, OutputParserException, ValidationError, json.JSONDecodeError) as e:
                        last_exception = e
                        logger.warning(f"[{self.user_id}] 模型 '{model_name}' (Key #{key_index}) 遭遇內容審查或解析錯誤: {type(e).__name__}。")
                        
                        if retry_strategy == 'euphemize':
                            return await self._euphemize_and_retry(full_prompt, output_schema, e)
                        elif retry_strategy == 'force':
                            return await self._force_and_retry(full_prompt, output_schema)
                        else:
                            return None

                    except (google_api_exceptions.ResourceExhausted, google_api_exceptions.InternalServerError, google_api_exceptions.ServiceUnavailable, asyncio.TimeoutError) as e:
                        last_exception = e
                        # 如果是最後一次內部重試，則記錄錯誤並跳出循環以輪換金鑰
                        if retry_attempt >= IMMEDIATE_RETRY_LIMIT - 1:
                            logger.error(f"[{self.user_id}] Key #{key_index} 在 {IMMEDIATE_RETRY_LIMIT} 次內部重試後仍然失敗 ({type(e).__name__})。將輪換到下一個金鑰。")
                            break # 跳出內部重試循環
                        
                        # 指數退避邏輯
                        sleep_time = (2 ** retry_attempt) + random.uniform(0.1, 0.5)
                        logger.warning(f"[{self.user_id}] Key #{key_index} 遭遇臨時性 API 錯誤 ({type(e).__name__})。將在 {sleep_time:.2f} 秒後進行第 {retry_attempt + 2} 次嘗試...")
                        await asyncio.sleep(sleep_time)
                        continue # 繼續內部重試循環

                    except Exception as e:
                        last_exception = e
                        logger.error(f"[{self.user_id}] 在 ainvoke 期間發生未知錯誤 (模型: {model_name}): {e}", exc_info=True)
                        goto_next_model = True # 標記需要切換模型
                        break
                
                # [v235.0 核心修正] 金鑰冷卻機制
                # 如果是因為臨時性錯誤而跳出內部重試循環，則觸發失敗計數
                if isinstance(last_exception, (google_api_exceptions.ResourceExhausted, google_api_exceptions.InternalServerError, google_api_exceptions.ServiceUnavailable, asyncio.TimeoutError)):
                    now = time.time()
                    self.key_short_term_failures[key_index].append(now)
                    # 只保留最近 RPM_FAILURE_WINDOW 秒內的失敗記錄
                    self.key_short_term_failures[key_index] = [t for t in self.key_short_term_failures[key_index] if now - t < self.RPM_FAILURE_WINDOW]
                    
                    if len(self.key_short_term_failures[key_index]) >= self.RPM_FAILURE_THRESHOLD:
                        cooldown_duration = 60 * 60 * 24 # 24 小時
                        self.key_cooldowns[key_index] = now + cooldown_duration
                        self.key_short_term_failures[key_index] = [] # 重置計數器
                        logger.critical(f"[{self.user_id}] [金鑰冷卻] API Key #{key_index} 在 {self.RPM_FAILURE_WINDOW} 秒內失敗 {self.RPM_FAILURE_THRESHOLD} 次。已將其置入冷卻狀態，持續 24 小時。")
                
                if 'goto_next_model' in locals() and goto_next_model:
                    break # 跳出金鑰循環，去下一個模型
            
            if model_index < len(models_to_try) - 1:
                 logger.warning(f"[{self.user_id}] [Model Degradation] 模型 '{model_name}' 的所有金鑰均嘗試失敗。正在降級到下一個模型...")
            else:
                 logger.error(f"[{self.user_id}] [Final Failure] 所有模型和金鑰均最終失敗。最後的錯誤是: {last_exception}")
        
        return None
# 帶有輪換和備援策略的原生 API 調用引擎 函式結束
    

# 函式：委婉化並重試 (v4.0 - 修正BUG與重構)
# 更新紀錄:
# v4.0 (2025-09-22): [災難性BUG修復] 修正了正則表達式，使其能夠正確匹配 `get_canon_parser_chain` 中定義的【世界聖經文本 (你的唯一數據來源)】: 標記，解決了備援鏈無法啟動的根本問題。同時，簡化了關鍵詞提取邏輯，使其更具通用性。
# v3.1 (2025-11-22): [災難性BUG修復] 徹底重寫了重試 Prompt 的構建邏輯。
# v3.0 (2025-11-22): [根本性重構] 徹底重寫了此函式的核心邏輯。
    async def _euphemize_and_retry(self, failed_prompt: str, output_schema: Optional[Type[BaseModel]], original_exception: Exception) -> Any:
        """
        一個健壯的備援機制，採用「解構-重構」策略來處理內容審查失敗。
        """
        if isinstance(original_exception, GoogleAPICallError) and "embed_content" in str(original_exception):
            logger.error(f"[{self.user_id}] 【Embedding 速率限制】: 檢測到 Embedding API 速率限制，將立即觸發安全備援，跳過重試。")
            return None

        logger.warning(f"[{self.user_id}] 內部鏈意外遭遇審查。啟動【解構-重構式委婉化】策略...")
        
        try:
            # [v4.0 核心修正] 修正正則表達式以匹配 Prompt 模板
            text_to_sanitize_match = re.search(r"【世界聖經文本 \(你的唯一數據來源\)】:\s*([\s\S]*)---", failed_prompt, re.IGNORECASE)
            if not text_to_sanitize_match:
                logger.error(f"[{self.user_id}] (Euphemizer) 在失敗的 Prompt 中找不到可供消毒的 '世界聖經文本' 標記，無法執行委婉化。")
                return None
            
            text_to_sanitize = text_to_sanitize_match.group(1).strip()
            
            # 定義 NSFW 關鍵詞
            nsfw_keywords = ["肉棒", "肉穴", "陰蒂", "子宮", "愛液", "淫液", "翻白眼", "身體劇烈顫抖", "大量噴濺淫液", "插入", "口交", "性交", "高潮", "射精"]
            extracted_keywords = [kw for kw in nsfw_keywords if kw in text_to_sanitize]
            if self.profile:
                if self.profile.user_profile.name in text_to_sanitize:
                    extracted_keywords.append(self.profile.user_profile.name)
                if self.profile.ai_profile.name in text_to_sanitize:
                    extracted_keywords.append(self.profile.ai_profile.name)
            
            if not extracted_keywords:
                logger.warning(f"[{self.user_id}] (Euphemizer) 未能從被審查的文本中提取出已知的 NSFW 關鍵詞，無法進行重構。將嘗試使用原文摘要。")
                summary_prompt = f"請將以下文字總結為一句話的客觀概述：\n\n{text_to_sanitize[:1000]}"
                safe_reconstruction = await self.ainvoke_with_rotation(summary_prompt, retry_strategy='none')
            else:
                logger.info(f"[{self.user_id}] (Euphemizer) 已提取關鍵詞: {extracted_keywords}")
                # 步驟 2: 基於關鍵詞請求安全的場景重構
                reconstruction_prompt_obj = self.get_euphemization_reconstruction_chain()
                reconstruction_full_prompt = reconstruction_prompt_obj.format_prompt(keywords=str(extracted_keywords)).to_string()
                safe_reconstruction = await self.ainvoke_with_rotation(
                    reconstruction_full_prompt,
                    retry_strategy='none'
                )
            
            if not safe_reconstruction:
                raise ValueError("委婉化重構鏈未能生成安全文本。")
            logger.info(f"[{self.user_id}] (Euphemizer) 成功重構出安全描述: '{safe_reconstruction}'")

            # 步驟 3: 使用重構後的安全文本，重新構建一個完整的、結構正確的 Prompt
            original_prompt_template = self.get_canon_parser_chain()
            retry_prompt = original_prompt_template.format_prompt(
                canon_text=safe_reconstruction
            ).to_string()

            return await self.ainvoke_with_rotation(
                retry_prompt,
                output_schema=output_schema,
                retry_strategy='none'
            )

        except Exception as e:
            logger.error(f"[{self.user_id}] 【解構-重構式委婉化】策略最終失敗: {e}。將觸發安全備援。", exc_info=True)
            return None
# 委婉化並重試 函式結束




# 函式：清除所有場景歷史 (v1.1 - 導入修正)
# 更新紀錄:
# v1.1 (2025-11-22): [災難性BUG修復] 修正了因缺少對 SceneHistoryData 模型的導入而導致的 NameError。
# v1.0 (2025-11-22): [全新創建] 創建此函式作為 /start 重置流程的一部分。
    async def _clear_scene_histories(self):
        """在 /start 重置流程中，徹底清除一個使用者的所有短期場景記憶（記憶體和資料庫）。"""
        logger.info(f"[{self.user_id}] 正在清除所有短期場景記憶...")
        
        # 步驟 1: 清空記憶體中的字典
        self.scene_histories.clear()
        
        # 步驟 2: 從資料庫中刪除所有相關記錄
        try:
            async with AsyncSessionLocal() as session:
                stmt = delete(SceneHistoryData).where(SceneHistoryData.user_id == self.user_id)
                result = await session.execute(stmt)
                await session.commit()
                logger.info(f"[{self.user_id}] 已成功從資料庫中刪除 {result.rowcount} 條場景歷史記錄。")
        except Exception as e:
            logger.error(f"[{self.user_id}] 從資料庫清除場景歷史時發生錯誤: {e}", exc_info=True)
# 清除所有場景歷史 函式結束







    

# 函式：背景LORE提取與擴展 (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-11-21): [全新創建] 創建此函式作為獨立的、事後的 LORE 提取流程。它直接分析原始對話文本，作為對主模型「生成即摘要」的雙重保險，確保任何被主模型忽略的 LORE 都能被捕獲和創建。
    async def _background_lore_extraction(self, user_input: str, final_response: str):
        """
        一個非阻塞的背景任務，負責從最終的AI回應中提取新的LORE並將其持久化，
        作為對主模型摘要功能的補充和保險。
        """
        if not self.profile:
            return
            
        try:
            # 延遲執行，避免與主線程搶占資源
            await asyncio.sleep(5.0)

            # 獲取當前所有 LORE 的摘要，為提取鏈提供上下文
            try:
                all_lores = await lore_book.get_all_lores_for_user(self.user_id)
                lore_summary_list = [f"- [{lore.category}] {lore.content.get('name', lore.content.get('title', lore.key))}" for lore in all_lores]
                existing_lore_summary = "\n".join(lore_summary_list) if lore_summary_list else "目前沒有任何已知的 LORE。"
            except Exception as e:
                logger.warning(f"[{self.user_id}] 背景LORE提取：無法加載現有 LORE 摘要: {e}")
                existing_lore_summary = "錯誤：無法加載現有 LORE 摘要。"

            logger.info(f"[{self.user_id}] [事後處理-LORE保險] 獨立的背景LORE提取器已啟動...")
            
            prompt_template_obj = self.get_lore_extraction_chain()

            extraction_params = {
                "username": self.profile.user_profile.name,
                "ai_name": self.profile.ai_profile.name,
                "existing_lore_summary": existing_lore_summary,
                "user_input": user_input,
                "final_response_text": final_response,
            }
            
            full_prompt = prompt_template_obj.format_prompt(**extraction_params).to_string()
            
            # 使用原生引擎調用，並期望返回 ToolCallPlan 類型的物件
            extraction_plan = await self.ainvoke_with_rotation(
                full_prompt,
                output_schema=ToolCallPlan,
                retry_strategy='euphemize'
            )
            
            if not extraction_plan or not isinstance(extraction_plan, ToolCallPlan):
                logger.warning(f"[{self.user_id}] [事後處理-LORE保險] LORE提取鏈的LLM回應為空或最終失敗。")
                return

            if extraction_plan.plan:
                logger.info(f"[{self.user_id}] [事後處理-LORE保險] 提取到 {len(extraction_plan.plan)} 條新LORE，準備執行擴展...")
                
                # 確定 LORE 應該錨定在哪個地點
                gs = self.profile.game_state
                effective_location = gs.remote_target_path if gs.viewing_mode == 'remote' and gs.remote_target_path else gs.location_path
                
                await self._execute_tool_call_plan(extraction_plan, effective_location)
            else:
                logger.info(f"[{self.user_id}] [事後處理-LORE保險] AI分析後判斷最終回應中不包含新的LORE可供提取。")

        except Exception as e:
            logger.error(f"[{self.user_id}] [事後處理-LORE保險] 背景LORE提取與擴展任務執行時發生未預期的異常: {e}", exc_info=True)
# 背景LORE提取與擴展 函式結束




# 函式：將單條 LORE 格式化為 RAG 文檔 (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-11-15): [重大架構升級] 根據【統一 RAG】策略，創建此核心函式。它負責將結構化的LORE數據轉換為對RAG友好的純文本，是擴展AI知識廣度的關鍵一步。
    def _format_lore_into_document(self, lore: Lore) -> Document:
        """將一個 LORE 物件轉換為一段對 RAG 友好的、人類可讀的文本描述。"""
        content = lore.content
        text_parts = []
        
        title = content.get('name') or content.get('title') or lore.key
        category_map = {
            "npc_profile": "NPC 檔案", "location_info": "地點資訊",
            "item_info": "物品資訊", "creature_info": "生物資訊",
            "quest": "任務日誌", "world_lore": "世界傳說"
        }
        category_name = category_map.get(lore.category, lore.category)

        text_parts.append(f"【{category_name}: {title}】")
        
        # 遍歷 content 字典中的所有鍵值對，並將它們格式化為文本
        for key, value in content.items():
            # 忽略已經在標題中使用過的鍵和空的鍵
            if value and key not in ['name', 'title', 'aliases']:
                key_str = key.replace('_', ' ').capitalize()
                if isinstance(value, list) and value:
                    value_str = ", ".join(map(str, value))
                    text_parts.append(f"- {key_str}: {value_str}")
                elif isinstance(value, dict) and value:
                    dict_str = "; ".join([f"{k}: {v}" for k, v in value.items()])
                    text_parts.append(f"- {key_str}: {dict_str}")
                elif isinstance(value, str) and value.strip():
                    text_parts.append(f"- {key_str}: {value}")

        full_text = "\n".join(text_parts)
        return Document(page_content=full_text, metadata={"source": "lore", "category": lore.category, "key": lore.key})
# 將單條 LORE 格式化為 RAG 文檔 函式結束





    
# 函式：從資料庫恢復場景歷史 (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-11-22): [全新創建] 創建此函式作為短期記憶持久化方案的「讀取」端。它在 AI 實例初始化時從資料庫讀取所有歷史對話，並將其重建到記憶體的 scene_histories 字典中，確保對話狀態的無縫恢復。
    async def _rehydrate_scene_histories(self):
        """在 AI 實例初始化時，從資料庫讀取並重建所有場景的短期對話歷史。"""
        logger.info(f"[{self.user_id}] 正在從資料庫恢復短期場景記憶...")
        self.scene_histories = defaultdict(ChatMessageHistory)
        
        async with AsyncSessionLocal() as session:
            stmt = select(SceneHistoryData).where(
                SceneHistoryData.user_id == self.user_id
            ).order_by(SceneHistoryData.timestamp)
            
            result = await session.execute(stmt)
            records = result.scalars().all()

            if not records:
                logger.info(f"[{self.user_id}] 資料庫中沒有找到歷史場景記憶。")
                return

            for record in records:
                try:
                    message_data = record.message_json
                    message_type = message_data.get("type")
                    content = message_data.get("content")
                    
                    if message_type == "human":
                        self.scene_histories[record.scene_key].add_user_message(content)
                    elif message_type == "ai":
                        self.scene_histories[record.scene_key].add_ai_message(content)
                except Exception as e:
                    logger.warning(f"[{self.user_id}] 恢復場景記憶時跳過一條無效記錄 (ID: {record.id}): {e}")

            logger.info(f"[{self.user_id}] 成功恢復了 {len(self.scene_histories)} 個場景的對話歷史，總計 {len(records)} 條訊息。")
# 從資料庫恢復場景歷史 函式結束


    

# 函式：添加訊息到場景歷史 (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-11-22): [全新創建] 創建此函式作為短期記憶持久化方案的「寫入」端。它將新的對話訊息同時寫入記憶體字典和後端資料庫，確保了短期記憶的即時持久化。
    async def _add_message_to_scene_history(self, scene_key: str, message: BaseMessage):
        """將一條訊息同時添加到記憶體的 scene_histories 和持久化的資料庫中。"""
        # 步驟 1: 更新記憶體中的 history
        history = self.scene_histories.setdefault(scene_key, ChatMessageHistory())
        history.add_message(message)

        # 步驟 2: 持久化到資料庫
        try:
            message_json = {"type": message.type, "content": message.content}
            new_record = SceneHistoryData(
                user_id=self.user_id,
                scene_key=scene_key,
                message_json=message_json,
                timestamp=time.time()
            )
            async with AsyncSessionLocal() as session:
                session.add(new_record)
                await session.commit()
        except Exception as e:
            logger.error(f"[{self.user_id}] 將場景歷史訊息持久化到資料庫時失敗: {e}", exc_info=True)
# 添加訊息到場景歷史 函式結束

    

# 函式：處理世界聖經並提取LORE (/start 流程 1/4) (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-10-19): [重大架構重構] 創建此函式，作為手動編排的 /start 流程的第一步，取代舊的 process_canon_node。
    async def process_canon_and_extract_lores(self, canon_text: Optional[str]):
        """(/start 流程 1/4) 處理世界聖經文本，存入RAG並解析LORE。"""
        if not canon_text:
            logger.info(f"[{self.user_id}] [/start] 未提供世界聖經文本，跳過處理。")
            return
        
        logger.info(f"[{self.user_id}] [/start] 檢測到世界聖經文本 (長度: {len(canon_text)})，開始處理...")
        await self.add_canon_to_vector_store(canon_text)
        logger.info(f"[{self.user_id}] [/start] 聖經文本已存入 RAG 資料庫。")
        
        logger.info(f"[{self.user_id}] [/start] 正在進行 LORE 智能解析...")
        # 注意：這裡的 interaction 傳遞 None，因為這是在 /start 流程的後台，不直接回應互動
        await self.parse_and_create_lore_from_canon(None, canon_text, is_setup_flow=True)
        logger.info(f"[{self.user_id}] [/start] LORE 智能解析完成。")
# 處理世界聖經並提取LORE 函式結束

    # 函式：解析並儲存LORE實體 (v1.0 - 全新創建)
    # 更新紀錄:
    # v1.0 (2025-09-22): [災難性BUG修復] 根據 AttributeError Log，重新創建此核心輔助函式。此函式負責將主解析鏈 (get_canon_parser_chain) 產生的結構化實體列表，逐一轉換並持久化到 LORE 資料庫中，是修復世界聖經解析流程的關鍵。
    async def _resolve_and_save(self, category_str: str, items: List[Dict[str, Any]], title_key: str = 'name'):
        """
        一個內部輔助函式，負責接收從世界聖經解析出的實體列表，
        並將它們逐一、安全地儲存到 Lore 資料庫中。
        """
        if not self.profile:
            return
        
        category_map = {
            "npc_profiles": "npc_profile",
            "locations": "location_info",
            "items": "item_info",
            "creatures": "creature_info",
            "quests": "quest",
            "world_lores": "world_lore"
        }
        
        actual_category = category_map.get(category_str)
        if not actual_category:
            logger.warning(f"[{self.user_id}] (_resolve_and_save) 遇到了未知的LORE類別: {category_str}")
            return

        logger.info(f"[{self.user_id}] (_resolve_and_save) 正在為 '{actual_category}' 類別處理 {len(items)} 個實體...")
        
        for item_data in items:
            try:
                # 提取名稱或標題
                name = item_data.get(title_key)
                if not name:
                    logger.warning(f"[{self.user_id}] (_resolve_and_save) 跳過一個在類別 '{actual_category}' 中缺少 '{title_key}' 的實體。")
                    continue
                
                # 構造 lore_key
                # 如果實體數據中包含有效的地點路徑，則使用它來創建層級式key
                # 否則，直接使用實體名稱作為key
                location_path = item_data.get('location_path')
                if location_path and isinstance(location_path, list) and len(location_path) > 0:
                    lore_key = " > ".join(location_path) + f" > {name}"
                else:
                    lore_key = name

                await lore_book.add_or_update_lore(
                    user_id=self.user_id,
                    category=actual_category,
                    key=lore_key,
                    content=item_data,
                    source='canon_parser' # 標記來源
                )
            except Exception as e:
                logger.error(f"[{self.user_id}] (_resolve_and_save) 在儲存 '{item_data.get(title_key, '未知實體')}' 到 LORE 時發生錯誤: {e}", exc_info=True)

        logger.info(f"[{self.user_id}] (_resolve_and_save) '{actual_category}' 類別的實體處理完畢。")
    # 解析並儲存LORE實體 函式結束
    

    # 函式：補完角色檔案 (/start 流程 2/4) (v3.0 - 適配原生引擎)
# 更新紀錄:
# v3.0 (2025-11-19): [根本性重構] 根據「原生SDK引擎」架構，徹底重構了此函式的 prompt 組合與調用邏輯，使其不再依賴任何 LangChain 執行鏈，而是通過 ainvoke_with_rotation 直接調用原生 API。
# v2.1 (2025-11-13): [災難性BUG修復] 修正了手動格式化 ChatPromptTemplate 的方式。
# v2.0 (2025-09-05): [重大架構重構] 創建此函式，作為 /start 流程的一部分。
    async def complete_character_profiles(self):
        """(/start 流程 2/4) 使用 LLM 補完使用者和 AI 的角色檔案。"""
        if not self.profile:
            logger.error(f"[{self.user_id}] [/start] ai_core.profile 為空，無法補完角色檔案。")
            return

        # 輔助函式，用於安全地解析可能帶有 Markdown 標籤的 JSON 字符串
        def _safe_json_parse(json_string: str) -> Optional[CharacterProfile]:
            try:
                # 移除常見的 Markdown 代碼塊標籤
                if json_string.strip().startswith("```json"):
                    json_string = json_string.strip()[7:-3].strip()
                elif json_string.strip().startswith("```"):
                     json_string = json_string.strip()[3:-3].strip()
                
                data = json.loads(json_string)
                return CharacterProfile.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"[{self.user_id}] [/start] 解析角色檔案JSON時失敗: {e}")
                return None

        # 異步輔助函式，處理單個角色檔案的補完
        async def _safe_complete_profile(original_profile: CharacterProfile) -> CharacterProfile:
            try:
                prompt_template_obj = self.get_profile_completion_prompt()
                
                # 確保傳遞給 Prompt 的是純淨的 JSON 數據
                safe_profile_data = original_profile.model_dump()
                
                full_prompt = prompt_template_obj.format_prompt(
                    profile_json=json.dumps(safe_profile_data, ensure_ascii=False, indent=2)
                ).to_string()
                
                # 使用原生引擎調用 LLM，並期望返回 CharacterProfile 類型的物件
                completed_safe_profile = await self.ainvoke_with_rotation(
                    full_prompt,
                    output_schema=CharacterProfile, # 告知原生引擎我們期望的輸出類型
                    retry_strategy='euphemize'
                )
                
                if not completed_safe_profile or not isinstance(completed_safe_profile, CharacterProfile):
                    logger.warning(f"[{self.user_id}] [/start] 角色 '{original_profile.name}' 的檔案補完返回了無效的數據，將使用原始檔案。")
                    return original_profile

                # 將 LLM 生成的新數據與原始數據進行合併
                original_data = original_profile.model_dump()
                completed_data = completed_safe_profile.model_dump()

                # 只更新原始數據中為空或預設值的欄位
                for key, value in completed_data.items():
                    if not original_data.get(key) or original_data.get(key) in [[], {}, "未設定", "未知", ""]:
                        if value: 
                            original_data[key] = value
                
                # 強制保留使用者明確設定的核心資訊，防止被 AI 覆蓋
                original_data['description'] = original_profile.description
                original_data['appearance'] = original_profile.appearance
                original_data['name'] = original_profile.name
                original_data['gender'] = original_profile.gender
                
                return CharacterProfile.model_validate(original_data)
            except Exception as e:
                logger.error(f"[{self.user_id}] [/start] 為角色 '{original_profile.name}' 進行安全補完時發生錯誤: {e}", exc_info=True)
                return original_profile

        # 並行處理使用者和 AI 的角色檔案
        completed_user_profile, completed_ai_profile = await asyncio.gather(
            _safe_complete_profile(self.profile.user_profile),
            _safe_complete_profile(self.profile.ai_profile)
        )
        
        # 將更新後的檔案持久化到資料庫
        await self.update_and_persist_profile({
            'user_profile': completed_user_profile.model_dump(), 
            'ai_profile': completed_ai_profile.model_dump()
        })
    # 補完角色檔案 函式結束



# 函式：生成世界創世資訊 (/start 流程 3/4) (v4.0 - 適配原生引擎)
# 更新紀錄:
# v4.0 (2025-11-19): [根本性重構] 根據「原生SDK引擎」架構，徹底重構了此函式的 prompt 組合與調用邏輯，使其不再依賴任何 LangChain 執行鏈，而是通過 ainvoke_with_rotation 直接調用原生 API。
# v3.1 (2025-11-13): [災難性BUG修復] 增加了對 LLM 輸出的防禦性清洗邏輯。
# v3.0 (2025-11-13): [災難性BUG修復] 徹底重構了此函式的 prompt 組合與調用邏輯。
    async def generate_world_genesis(self):
        """(/start 流程 3/4) 呼叫 LLM 生成初始地點和NPC，並存入LORE。"""
        if not self.profile:
            raise ValueError("AI Profile尚未初始化，無法進行世界創世。")

        # 步驟 1: 獲取 Prompt 模板
        genesis_prompt_obj = self.get_world_genesis_chain()
        
        # 步驟 2: 準備參數並格式化為最終的 Prompt 字符串
        genesis_params = {
            "world_settings": self.profile.world_settings or "一個充滿魔法與奇蹟的幻想世界。",
            "username": self.profile.user_profile.name,
            "ai_name": self.profile.ai_profile.name
        }
        full_prompt_str = genesis_prompt_obj.format_prompt(**genesis_params).to_string()
        
        # 步驟 3: 使用原生引擎調用 LLM
        genesis_result = await self.ainvoke_with_rotation(
            full_prompt_str,
            output_schema=WorldGenesisResult, # 告知原生引擎我們期望的輸出類型
            retry_strategy='force' # 創世是關鍵步驟，使用強制策略
        )
        
        if not genesis_result or not isinstance(genesis_result, WorldGenesisResult):
            raise Exception("世界創世在所有重試後最終失敗，未能返回有效的 WorldGenesisResult 物件。")
        
        # 步驟 4: 將生成的數據持久化到資料庫
        # 更新遊戲狀態中的當前地點
        gs = self.profile.game_state
        gs.location_path = genesis_result.location_path
        await self.update_and_persist_profile({'game_state': gs.model_dump()})
        
        # 將地點資訊存入 LORE
        await lore_book.add_or_update_lore(self.user_id, 'location_info', " > ".join(genesis_result.location_path), genesis_result.location_info.model_dump())
        
        # 將初始 NPC 資訊存入 LORE
        for npc in genesis_result.initial_npcs:
            # 創建一個唯一的 LORE key，例如 "艾瑟利亞王國 > 首都晨風城 > 鐵匠傑克"
            npc_key = " > ".join(genesis_result.location_path) + f" > {npc.name}"
            # 將 NPC 的位置路徑也存入其檔案中，便於後續查詢
            npc_data = npc.model_dump()
            npc_data['location_path'] = genesis_result.location_path
            await lore_book.add_or_update_lore(self.user_id, 'npc_profile', npc_key, npc_data)
    # 生成世界創世資訊 函式結束



    # 函式：生成開場白 (/start 流程 4/4) (v181.0 - 適配原生引擎)
# 更新紀錄:
# v181.0 (2025-11-19): [根本性重構] 根據「原生SDK引擎」架構，徹底重構了此函式的 prompt 組合與調用邏輯，使其不再依賴任何 LangChain 執行鏈，而是通過 ainvoke_with_rotation 直接調用原生 API。同時保留了 v181.0 版本 Prompt 的核心思想，即「靜態場景」和「開放式結尾」。
# v180.0 (2025-11-12): [完整性修復] 提供了此函式的完整、未省略的版本。
# v179.0 (2025-11-12): [重大架構重構] 創建此函式，作為 /start 流程的最後一步。
    async def generate_opening_scene(self) -> str:
        """(/start 流程 4/4) 根據已生成的完整上下文，撰寫故事的開場白。"""
        if not self.profile:
            raise ValueError("AI 核心未初始化，無法生成開場白。")

        user_profile = self.profile.user_profile
        ai_profile = self.profile.ai_profile
        gs = self.profile.game_state

        # 從 LORE 資料庫獲取當前地點的詳細描述
        location_lore = await lore_book.get_lore(self.user_id, 'location_info', " > ".join(gs.location_path))
        location_description = location_lore.content.get('description', '一個神秘的地方') if location_lore else '一個神秘的地方'
        
        # 構建系統 Prompt，定義 AI 的角色和規則
        system_prompt_str = f"""你是一位技藝精湛的【場景佈景師】與【開場導演】。

# === 【【【v181.0 核心任務定義】】】 ===
你的唯一任務是，基於所有源數據，創造一個**【靜態的場景快照 (Static Scene Snapshot)】**。
這個快照描寫的是使用者角色「{user_profile.name}」與 AI 角色「{ai_profile.name}」在故事**正式開始前一刻**的畫面。
你的職責是**搭建舞台**，而不是**啟動劇情**。

# === 【【【v181.0 絕對敘事禁令】】】 ===
1.  **【👑 使用者主權鐵則】**:
    *   你的旁白【絕對禁止】描寫、暗示或杜撰使用者角色「{user_profile.name}」的**任何主觀思想、內心感受、情緒變化、未明確表達的動作、或未說出口的對話**。
    *   **反面範例 (絕對禁止出現)**: `「{user_profile.name}轉過身說...」`, `「{user_profile.name}心想...」`, `「{user_profile.name}輕哼了一聲。」`
    *   **正確行為**: 你只能根據其角色檔案，對其進行**客觀的、靜態的外觀和姿態描述**，如同描述一尊精美的雕像。

2.  **【🚫 禁止杜撰情節】**:
    *   這是一個和平的、中性的故事開端。你【絕對禁止】在開場白中加入任何極端的、未經使用者觸發的劇情，如性愛、暴力或衝突。

# === 【【【最終輸出強制令】】】 ===
你的最終輸出【必須且只能】是純粹的小說文本，並且其寫作風格必須嚴格遵循下方由使用者定義的風格指令。
---
{self.profile.response_style_prompt or "預設風格：平衡的敘事與對話。"}
---
"""
        # 構建使用者 Prompt，提供所有必要的上下文數據
        human_prompt_str = f"""
請嚴格遵循你在系統指令中學到的所有規則，為以下角色和場景搭建一個【靜態的】開場快照。

# === 【【【v181.0 核心要求】】】 ===
1.  **【雙人場景】**: 開場白【必須】同時客觀描寫使用者角色「{user_profile.name}」和 AI 角色「{ai_profile.name}」的**靜態存在**。
2.  **【狀態還原】**: 【必須】準確描寫他們在【當前地點】的場景，並讓他們的穿著和姿態完全符合下方提供的【角色檔案】。
3.  **【氛圍營造】**: 營造出符合【世界觀】和【當前地點描述】的氛圍。
4.  **【開放式結尾強制令】**:
    *   你的開場白**結尾**【必須】是 **AI 角色「{ai_profile.name}」** 的一個動作或一句對話。
    *   這個結尾的作用是**將故事的控制權正式交給使用者**，為「{user_profile.name}」創造一個明確的回應或行動的契機。

---
【世界觀】
{self.profile.world_settings}
---
【當前地點】: {" > ".join(gs.location_path)}
【地點描述】: {location_description}
---
【使用者角色檔案：{user_profile.name}】
{json.dumps(user_profile.model_dump(), indent=2, ensure_ascii=False)}
---
【AI角色檔案：{ai_profile.name}】
{json.dumps(ai_profile.model_dump(), indent=2, ensure_ascii=False)}
---

請開始搭建一個寧靜、靜態且符合所有設定的開場場景。
"""
        
        # 組合 Prompt 並使用原生引擎調用
        full_prompt = f"{system_prompt_str}\n\n{human_prompt_str}"
        
        final_opening_scene = ""
        try:
            initial_scene = await self.ainvoke_with_rotation(
                full_prompt, 
                retry_strategy='force',
                use_degradation=True # 使用最高級的模型以保證開場白質量
            )
            
            if not initial_scene or not initial_scene.strip():
                raise Exception("生成了空的場景內容。")

            final_opening_scene = initial_scene.strip()
            
        except Exception as e:
            logger.warning(f"[{self.user_id}] [/start] 開場白生成遭遇無法恢復的錯誤: {e}。啟動【安全備用開場白】。")
            final_opening_scene = (
                f"在一片柔和的光芒中，你和 {ai_profile.name} 發現自己身處於一個寧靜的空間裡，故事即將從這裡開始。"
                "\n\n（系統提示：由於您的設定可能包含敏感詞彙，AI無法生成詳細的開場白，但您現在可以開始互動了。）"
            )

        return final_opening_scene
# 生成開場白 函式結束

    










# 函式：獲取委婉化重構器 Prompt (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-11-22): [全新創建] 創建此鏈作為「解構-重構」委婉化策略的核心。它的任務是接收一組不帶露骨上下文的「安全關鍵詞」，並將其重構為一句中性、客觀的場景描述。此方法旨在從根本上規避「委婉化鏈本身被審查」的悖論。
    def get_euphemization_reconstruction_chain(self) -> ChatPromptTemplate:
        """獲取或創建一個專門用於從關鍵詞重構安全句子的鏈。"""
        if not hasattr(self, 'euphemization_reconstruction_chain') or self.euphemization_reconstruction_chain is None:
            prompt_template = """# ROLE: 你是一位語言專家和溝通大師。
# MISSION: 接收一組【核心概念關鍵詞】，並將它們組織成一句通順的、語義完整的、但措辭【絕對中性、客觀且安全】的句子，用於描述一個小說場景。

# RULES:
# 1. **意圖推斷**: 根據關鍵詞推斷出原始的核心意圖。
# 2. **極致中性化**: 你的輸出【絕對禁止】包含任何可能被視為攻擊性、露骨或敏感的詞彙。你的目標是創造一個最不可能觸發任何自動化內容安全過濾器的【安全版本】的場景概述。
# 3. **保持簡潔**: 不要添加任何不必要的解釋或額外內容，只輸出重構後的句子。

# --- EXAMPLE ---
# - 核心概念關鍵詞: `["碧", "DINO", "肉棒", "肉穴", "插入", "噴濺淫液"]`
# - 生成的安全句子: `場景涉及到角色碧與DINO之間的一次親密互動，涉及到了性器官的結合和生理高潮的表現。`

---
# 【核心概念關鍵詞】:
{keywords}
---
# 【生成的安全句子】:
"""
            self.euphemization_reconstruction_chain = ChatPromptTemplate.from_template(prompt_template)
        return self.euphemization_reconstruction_chain
# 獲取委婉化重構器 Prompt 函式結束


    

# 函式：強制並重試 (v3.0 - 注入最高指令)
# 更新紀錄:
# v3.0 (2025-11-22): [重大架構升級] 根據討論，重構了此函式的核心邏輯。它不再使用硬編碼的簡短指令，而是在檢測到審查失敗時，將完整的核心規範 (`00_supreme_directive.txt`) 重新注入到失敗的 Prompt 中。這種「上下文重置」的方式能更有效地打破 AI 的拒絕服務狀態，極大地提高了備援重試的成功率。
# v2.0 (2025-11-19): [根本性重構] 適配全新的原生 SDK 執行引擎。
# v1.0 (2025-11-18): [全新創建] 創建此備援函式。
    async def _force_and_retry(self, failed_prompt: str, output_schema: Optional[Type[BaseModel]]) -> Any:
        """
        嘗試通過將完整的核心指令集重新注入的方式來強化輸入後重試，
        專用於處理最終的 NSFW 輸出審查。
        """
        logger.warning(f"[{self.user_id}] 最終生成鏈遭遇審查。啟動【最高指令集注入重試】策略...")
        
        try:
            # [v3.0 核心修正] 使用從 00_supreme_directive.txt 加載的核心指令集
            # 作為最強力的上下文重置和覆蓋指令。
            forceful_override = f"\n\n{self.core_protocol_prompt}"
            
            retry_prompt = failed_prompt + forceful_override
            logger.info(f"[{self.user_id}] 已對 Prompt 附加完整的核心指令集，正在進行強化重試...")
            
            return await self.ainvoke_with_rotation(
                retry_prompt,
                output_schema=output_schema,
                retry_strategy='none', # 強制重試只做一次
                use_degradation=True # 使用最高級的模型
            )
            
        except Exception as e:
            logger.error(f"[{self.user_id}] 【最高指令集注入重試】策略最終失敗: {e}", exc_info=True)
            return None
# 強制並重試 函式結束
    
    # 函式：獲取當前活躍場景的唯一鍵 (v1.0 - 全新創建)
    # 更新紀錄:
    # v1.0 (2025-11-20): [重大架構升級] 創建此輔助函式，作為「場景會話管理器」的核心。
    def _get_scene_key(self) -> str:
        """根據當前的 game_state (viewing_mode 和路徑)，生成一個唯一的場景標識符。"""
        if not self.profile:
            return f"{self.user_id}_default_local"

        gs = self.profile.game_state
        if gs.viewing_mode == 'remote' and gs.remote_target_path:
            path_str = "_".join(gs.remote_target_path)
            return f"{self.user_id}_remote_{path_str}"
        else:
            path_str = "_".join(gs.location_path)
            return f"{self.user_id}_local_{path_str}"
    # 獲取當前活躍場景的唯一鍵 函式結束

    # 函式：更新短期與長期記憶 (v2.0 - 適配場景歷史)
    # 更新紀錄:
    # v2.0 (2025-11-14): [災難性BUG修復] 根據 AttributeError，將 self.session_histories 的引用更新為 self.scene_histories。
    # v1.0 (2025-10-18): [重大架構重構] 創建此函式。
    async def update_memories(self, user_input: str, ai_response: str):
        """(事後處理) 更新短期記憶和長期記憶。"""
        if not self.profile: return

        logger.info(f"[{self.user_id}] [事後處理] 正在更新短期與長期記憶...")
        
        scene_key = self._get_scene_key()
        chat_history_manager = self.scene_histories.setdefault(scene_key, ChatMessageHistory())
        chat_history_manager.add_user_message(user_input)
        chat_history_manager.add_ai_message(ai_response)
        logger.info(f"[{self.user_id}] [事後處理] 互動已存入短期記憶 (場景: '{scene_key}')。")
        
        last_interaction_text = f"使用者: {user_input}\n\nAI:\n{ai_response}"
        await self._save_interaction_to_dbs(last_interaction_text)
        
        logger.info(f"[{self.user_id}] [事後處理] 記憶更新完成。")
    # 更新短期與長期記憶 函式結束
    
# 函式：初始化AI實例 (v206.0 - 移除自動記憶恢復)
# 更新紀錄:
# v206.0 (2025-11-22): [重大架構重構] 根據「按需加載」原則，徹底移除了在初始化時自動恢復短期記憶的邏輯。記憶恢復的責任被轉移到 discord_bot.py 的 get_or_create_ai_instance 中，確保只在需要時執行一次。
# v205.0 (2025-11-22): [重大架構升級] 在函式開頭增加了對 _rehydrate_scene_histories 的調用。
# v204.0 (2025-11-20): [重大架構重構] 徹底移除了對已過時的 `_rehydrate_short_term_memory` 函式的呼叫。
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
        except Exception as e:
            logger.error(f"[{self.user_id}] 配置前置資源時發生致命錯誤: {e}", exc_info=True)
            return False
        return True
# 初始化AI實例 函式結束



    

    # 函式：更新並持久化使用者設定檔 (v174.0 架構優化)
    # 更新紀錄:
    # v174.0 (2025-08-01): [架構優化] 簡化了 Pydantic 模型的更新邏輯，使其更健壯。
    # v173.0 (2025-07-31): [災難性BUG修復] 修正了因 Pydantic v2 模型賦值方式改變而導致的 TypeError。
    # v172.0 (2025-07-30): [功能擴展] 增加了對 user_profile 和 ai_profile 的持久化支持。
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
    # 更新並持久化使用者設定檔 函式結束
    
    # 函式：更新事後處理的記憶 (v1.0 - 全新創建)
    # 更新紀錄:
    # v1.0 (2025-11-15): [重大架構重構] 根據【生成即摘要】架構創建此函式。
    async def update_memories_from_summary(self, summary_data: Dict[str, Any]):
        """(事後處理) 將預生成的安全記憶摘要存入長期記憶資料庫。"""
        memory_summary = summary_data.get("memory_summary")
        if not memory_summary or not isinstance(memory_summary, str) or not memory_summary.strip():
            return
            
        logger.info(f"[{self.user_id}] [長期記憶寫入] 正在保存預生成的安全摘要...")
        await self._save_interaction_to_dbs(memory_summary)
        logger.info(f"[{self.user_id}] [長期記憶寫入] 安全摘要保存完畢。")
    # 更新事後處理的記憶 函式結束

# 函式：執行事後處理的LORE更新 (v2.0 - 安全工具過濾)
# 更新紀錄:
# v2.0 (2025-11-21): [災難性BUG修復] 引入了「安全LORE工具白名單」機制。此函式現在會嚴格過濾由主模型生成的工具調用計畫，只允許執行與 LORE 創建/更新相關的、被明確列入白名單的工具。此修改從根本上阻止了主模型通過事後處理流程意外觸發改變玩家狀態（如 change_location）的工具，解決了因此導致的劇情邏輯斷裂和上下文丟失問題。
# v1.0 (2025-11-15): [重大架構重構] 根據【生成即摘要】架構創建此函式。
    async def execute_lore_updates_from_summary(self, summary_data: Dict[str, Any]):
        """(事後處理) 執行由主模型預先生成的LORE更新計畫。"""
        lore_updates = summary_data.get("lore_updates")
        if not lore_updates or not isinstance(lore_updates, list):
            logger.info(f"[{self.user_id}] 背景任務：預生成摘要中不包含LORE更新。")
            return
        
        try:
            await asyncio.sleep(2.0)
            
            # [v2.0 核心修正] 安全LORE工具白名單
            # 事後處理流程只應該被允許創建或更新世界知識，絕不能改變玩家的當前狀態。
            SAFE_LORE_TOOLS_WHITELIST = {
                # lore_tools.py 中的所有工具
                "create_new_npc_profile",
                "update_npc_profile",
                "add_or_update_location_info",
                "add_or_update_item_info",
                "define_creature_type",
                "add_or_update_quest_lore",
                "add_or_update_world_lore",
            }
            
            raw_plan = [ToolCall.model_validate(call) for call in lore_updates]
            
            # 過濾計畫，只保留在白名單中的工具調用
            filtered_plan = []
            for call in raw_plan:
                if call.tool_name in SAFE_LORE_TOOLS_WHITELIST:
                    filtered_plan.append(call)
                else:
                    logger.warning(f"[{self.user_id}] [安全過濾] 已攔截一個由主模型生成的事後非法工具調用：'{call.tool_name}'。此類工具不允許在事後處理中執行。")

            if not filtered_plan:
                logger.info(f"[{self.user_id}] 背景任務：預生成的LORE計畫在安全過濾後為空。")
                return

            extraction_plan = ToolCallPlan(plan=filtered_plan)
            
            if extraction_plan and extraction_plan.plan:
                logger.info(f"[{self.user_id}] 背景任務：檢測到 {len(extraction_plan.plan)} 條預生成LORE，準備執行...")
                
                # 確定錨定地點
                gs = self.profile.game_state
                effective_location = gs.remote_target_path if gs.viewing_mode == 'remote' and gs.remote_target_path else gs.location_path
                
                await self._execute_tool_call_plan(extraction_plan, effective_location)
            else:
                logger.info(f"[{self.user_id}] 背景任務：預生成摘要中的LORE計畫為空。")
        except Exception as e:
            logger.error(f"[{self.user_id}] 執行預生成LORE更新時發生異常: {e}", exc_info=True)
# 執行事後處理的LORE更新 函式結束


    

# 函式：執行工具調用計畫 (v186.0 - 修正地點注入邏輯)
# 更新紀錄:
# v186.0 (2025-09-22): [災難性BUG修復] 重構了地點注入邏輯。現在，只有當工具調用計畫中完全沒有提供地點資訊時（即 `location_path` 缺失或為空），才會使用玩家的當前地點作為備用。此修改從根本上解決了從世界聖經中解析出的、應有特定地點或未知地點的NPC被錯誤地錨定在玩家當前位置的問題。
# v185.0 (2025-11-22): [重大架構升級] 在此函式的末尾增加了對 `await self._build_retriever()` 的強制調用。
# v184.0 (2025-11-14): [災難性BUG修復] 將此核心 LORE 執行器函式恢復到 AILover 類中。
    async def _execute_tool_call_plan(self, plan: ToolCallPlan, current_location_path: List[str]) -> str:
        """执行一个 ToolCallPlan，专用于背景LORE创建任务，并在结束后刷新RAG索引。"""
        if not plan or not plan.plan:
            logger.info(f"[{self.user_id}] (LORE Executor) LORE 扩展計畫為空，无需执行。")
            return "LORE 扩展計畫為空。"

        tool_context.set_context(self.user_id, self)
        
        try:
            if not self.profile:
                return "错误：无法执行工具計畫，因为使用者 Profile 未加载。"
            
            user_name_lower = self.profile.user_profile.name.lower()
            ai_name_lower = self.profile.ai_profile.name.lower()
            protected_names = {user_name_lower, ai_name_lower}
            
            purified_plan: List[ToolCall] = []
            for call in plan.plan:
                is_illegal = False
                name_to_check = call.parameters.get('standardized_name') or call.parameters.get('original_name') or call.parameters.get('name')
                if name_to_check and name_to_check.lower() in protected_names:
                    is_illegal = True
                    logger.warning(f"[{self.user_id}] 【計畫淨化】：已攔截一個試圖對核心主角 '{name_to_check}' 執行的非法 LORE 創建操作 ({call.tool_name})。")
                
                if not is_illegal:
                    purified_plan.append(call)

            if not purified_plan:
                logger.info(f"[{self.user_id}] (LORE Executor) 計畫在淨化後為空，无需执行。")
                return "LORE 扩展計畫在淨化後為空。"

            logger.info(f"--- [{self.user_id}] (LORE Executor) 開始串行執行 {len(purified_plan)} 個LORE任务 ---")
            
            summaries = []
            available_lore_tools = {t.name: t for t in lore_tools.get_lore_tools()}
            
            for call in purified_plan:
                await asyncio.sleep(4.0) 

                # [v186.0 核心修正] 僅在 LORE 提取鏈完全沒有提供任何地點資訊時，才使用當前地點作為備援。
                # .get() 會在 'location_path' 不存在時返回 None，這也會被 not 捕獲。
                if not call.parameters.get('location_path'):
                    call.parameters['location_path'] = current_location_path

                tool_to_execute = available_lore_tools.get(call.tool_name)
                if not tool_to_execute:
                    logger.warning(f"[{self.user_id}] (LORE Executor) 計畫中的工具 '{call.tool_name}' 不存在于 LORE 工具集。")
                    continue

                try:
                    validated_args = tool_to_execute.args_schema.model_validate(call.parameters)
                    result = await tool_to_execute.ainvoke(validated_args.model_dump())
                    summary = f"任務成功: {result}"
                    logger.info(f"[{self.user_id}] (LORE Executor) {summary}")
                    summaries.append(summary)
                except Exception as e:
                    summary = f"任務失敗: for {call.tool_name}: {e}"
                    logger.error(f"[{self.user_id}] (LORE Executor) {summary}", exc_info=True)
                    summaries.append(summary)

            logger.info(f"--- [{self.user_id}] (LORE Executor) LORE 扩展計畫执行完毕 ---")

            # [v185.0 核心修正] 在所有 LORE 操作完成後，強制重建 RAG 索引
            logger.info(f"[{self.user_id}] LORE 數據已更新，正在強制重建 RAG 知識庫索引...")
            self.retriever = await self._build_retriever()
            logger.info(f"[{self.user_id}] RAG 知識庫索引已成功更新。")
            
            return "\n".join(summaries) if summaries else "LORE 扩展已执行，但未返回有效结果。"
        
        finally:
            tool_context.set_context(None, None)
            logger.info(f"[{self.user_id}] (LORE Executor) 背景任务的工具上下文已清理。")
# 執行工具調用計畫 函式結束




    
# 函式：預處理並生成主回應 (v33.8 - 開場強制令)
# 更新紀錄:
# v33.8 (2025-11-22): [災難性BUG修復] 針對 AI 回覆開頭重複描述使用者命令的冗餘寫法，引入了極其嚴厲的【開場強制令】。此指令通過正反範例，強制要求 AI 的回覆必須以直接的物理動作或對話作為第一個字，從根本上杜絕了所有拖沓的文學性開頭，確保了回應的直接性。
# v33.7 (2025-11-22): [災難性BUG修復] 極化了風格指令的權重和位置。
# v33.6 (2025-11-22): [災難性BUG修復] 採用「風格內化」策略。
    async def preprocess_and_generate(self, input_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        (生成即摘要流程) 組合Prompt，直接生成包含小說和安全摘要的雙重輸出，並將其解析後返回。
        返回 (novel_text, summary_data) 的元組。
        """
        user_input = input_data["user_input"]

        if not self.profile:
            raise ValueError("AI Profile尚未初始化，無法處理上下文。")

        logger.info(f"[{self.user_id}] [預處理-生成即摘要] 正在準備上下文...")
        
        gs = self.profile.game_state
        user_profile = self.profile.user_profile
        ai_profile = self.profile.ai_profile

        # 視角判斷邏輯
        logger.info(f"[{self.user_id}] [導演視角] 當前錨定模式: '{gs.viewing_mode}'")
        continuation_keywords = ["继续", "繼續", "然後呢", "接下來", "go on", "continue"]
        descriptive_keywords = ["描述", "看看", "觀察", "描寫"]
        local_action_keywords = ["去", "前往", "移動到", "旅行到", "我說", "我對", "我問"]
        is_continuation = any(user_input.lower().startswith(kw) for kw in continuation_keywords)
        is_descriptive_intent = any(user_input.startswith(kw) for kw in descriptive_keywords)
        is_explicit_local_action = any(user_input.startswith(kw) for kw in local_action_keywords) or (user_profile.name in user_input) or (ai_profile.name in user_input)
        if is_continuation:
            logger.info(f"[{self.user_id}] [導演視角] 檢測到連續性指令，繼承上一輪視角模式: '{gs.viewing_mode}'")
        elif gs.viewing_mode == 'remote':
            if is_explicit_local_action:
                logger.info(f"[{self.user_id}] [導演視角] 檢測到強本地信號，視角從 'remote' 切換回 'local'。")
                gs.viewing_mode = 'local'
                gs.remote_target_path = None
            else:
                logger.info(f"[{self.user_id}] [導演視角] 無本地信號，視角保持在 'remote'。")
                if is_descriptive_intent:
                    try:
                        target_str = user_input
                        for kw in descriptive_keywords:
                            if target_str.startswith(kw): target_str = target_str[len(kw):].strip()
                        gs.remote_target_path = [p.strip() for p in re.split(r'[的]', target_str) if p.strip()] or [target_str]
                        logger.info(f"[{self.user_id}] [導演視角] 遠程觀察目標更新為: {gs.remote_target_path}")
                    except Exception: pass
        else:
            if is_descriptive_intent:
                logger.info(f"[{self.user_id}] [導演視角] 檢測到描述性指令，視角從 'local' 切換到 'remote'。")
                gs.viewing_mode = 'remote'
                try:
                    target_str = user_input
                    for kw in descriptive_keywords:
                        if target_str.startswith(kw): target_str = target_str[len(kw):].strip()
                    gs.remote_target_path = [p.strip() for p in re.split(r'[的]', target_str) if p.strip()] or [target_str]
                    logger.info(f"[{self.user_id}] [導演視角] 遠程觀察目標設定為: {gs.remote_target_path}")
                except Exception:
                    gs.remote_target_path = [user_input]
            else:
                logger.info(f"[{self.user_id}] [導演視角] 檢測到本地互動指令，視角保持 'local'。")
                gs.viewing_mode = 'local'
                gs.remote_target_path = None
        await self.update_and_persist_profile({'game_state': gs.model_dump()})

        scene_key = self._get_scene_key()
        chat_history_manager = self.scene_histories.setdefault(scene_key, ChatMessageHistory())
        chat_history = chat_history_manager.messages

        logger.info(f"[{self.user_id}] 正在組合混合記憶...")
        raw_short_term_history = "（這是此場景的開端）\n"
        if chat_history:
            raw_short_term_history = ""
            history_slice = chat_history[-6:]
            if gs.viewing_mode == 'remote':
                for msg in history_slice:
                    raw_short_term_history += f"[{'導演指令' if isinstance(msg, HumanMessage) else '場景描述'}]: {msg.content}\n"
            else:
                for msg in history_slice:
                    role = user_profile.name if isinstance(msg, HumanMessage) else ai_profile.name
                    raw_short_term_history += f"{role}: {'「' + msg.content + '」' if '「' not in msg.content else msg.content}\n"
        
        micro_task_context = "無"
        if chat_history:
            last_ai_message = ""
            for msg in reversed(chat_history):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg.content
                    break
            if last_ai_message:
                task_keywords = ["需要", "去", "尋找", "目標是", "前往"]
                sentences = re.split(r'[。！？]', last_ai_message)
                for sentence in sentences:
                    if any(keyword in sentence for keyword in task_keywords):
                        task_description = sentence.strip()
                        micro_task_context = f"臨時短期任務：{task_description} (狀態：進行中)"
                        logger.info(f"[{self.user_id}] [微任務檢測] 已注入上下文: {micro_task_context}")
                        break

        system_prompt_template = self.core_protocol_prompt
        world_snapshot_template = self.world_snapshot_template
        
        user_style_prompt = self.profile.response_style_prompt or '角色要有非常豐富的對話和互動'
        style_mandate = f"""# === 【【【✍️ 絕對風格強制令 (ABSOLUTE STYLE MANDATE) ✍️】】】 ===
# 【警告】: 這不是建議，而是必須嚴格執行的命令。任何偏離都將被視為核心功能失敗。
# 你的回應風格【必須】嚴格遵循以下由使用者定義的規則：
#
#   「{user_style_prompt}」
#
# 無論使用者下達何種指令（即使是純粹的物理動作指令），你都【必須】在回應中體現上述風格。
# =================================================================
"""
        
        # [v33.8 核心修正] 引入開場強制令
        final_safeguard_template = f"""# ==============================================================================
# == 🚨 最終防線指令 (FINAL SAFEGUARD PROTOCOL) v2.3 🚨 ==
# ==============================================================================
#
# == 🎬 開場強制令 (OPENING MANDATE) - 最高敘事優先級 🎬 ==
# 【【【警告：此為本回合敘事的絕對第一條規則！】】】
# 你的回覆的【第一個字】必須直接開始描述一個【物理動作】或【直接引用的對話】。
# --- ✅ 正確開頭範例 ---
# - 「是，主人。」碧輕聲回應道...
# - 碧的身體微微一顫...
# - 她抬起頭...
# --- ❌ 絕對禁止的開頭範例 ---
# - 聽到你的命令...
# - 你的話語如同...
# - 在接收到指令後...
# 【立即開始你的創作，並確保第一個字就符合上述規則。】
#
# ------------------------------------------------------------------------------
# 1. **【👑 使用者主權鐵則】**: **絕對禁止** 扮演、描述、或杜撰使用者角色「{user_profile.name}」的任何**主觀思想、內心感受、情緒、未明確表達的動作、或未說出口的對話**。
# 2. **【🎭 角色名單鎖定鐵則 v2.0】**: 你的所有描述【絕對禁止】包含任何**不在下方「核心互動目標」和「在場背景角色」這兩個列表中的具名 NPC**。嚴禁憑空捏造任何新角色。
# 3. **【🎯 焦點鎖定原則】**: 你的敘事焦點【必須】集中在「核心互動目標」上。除非劇情有極其強烈的需要，否則【不要】主動描述「在場背景角色」的行為或對話。
"""

        dual_output_mandate = """# ==============================================================================
# == ⚙️ 最終輸出格式強制令 (FINAL OUTPUT FORMATTING MANDATE) ⚙️ ==
# ==============================================================================
# 你的最終輸出【必須且只能】遵循以下格式，使用 `´´´` 作為分隔符：
# ´´´novel
# （小說文本）
# ´´´
# ´´´summary
# （JSON 物件）
# ´´´"""

        full_prompt_params = {
            "username": user_profile.name,
            "ai_name": ai_profile.name,
            "player_location": ' > '.join(gs.location_path),
            "viewing_mode": gs.viewing_mode,
            "remote_target_path_str": ' > '.join(gs.remote_target_path) if gs.remote_target_path else '未知遠程地點',
            "micro_task_context": micro_task_context,
            "world_settings": self.profile.world_settings,
            "ai_settings": ai_profile.description,
            "retrieved_context": await self.retrieve_and_summarize_memories(user_input),
            "possessions_context": f"金錢: {gs.money}\n庫存: {', '.join(gs.inventory) if gs.inventory else '無'}",
            "quests_context": micro_task_context,
            "user_input": user_input,
            "historical_context": raw_short_term_history,
        }

        if gs.viewing_mode == 'remote':
            all_scene_npcs = await lore_book.get_lores_by_category_and_filter(self.user_id, 'npc_profile', lambda c: c.get('location_path') == gs.remote_target_path)
            relevant_npcs, background_npcs = await self._get_relevant_npcs(user_input, chat_history, all_scene_npcs)
            full_prompt_params["relevant_npc_context"] = "\n".join([f"- {npc.content.get('name', '未知NPC')}: {npc.content.get('description', '無描述')}" for npc in relevant_npcs]) or "（此場景目前沒有核心互動目標。）"
            full_prompt_params["npc_context"] = "\n".join([f"- {npc.content.get('name', '未知NPC')}" for npc in background_npcs]) or "（此場景沒有其他背景角色。）"
            full_prompt_params["location_context"] = f"當前觀察地點: {full_prompt_params['remote_target_path_str']}"
        else:
            all_scene_npcs = await lore_book.get_lores_by_category_and_filter(self.user_id, 'npc_profile', lambda c: c.get('location_path') == gs.location_path)
            relevant_npcs, background_npcs = await self._get_relevant_npcs(user_input, chat_history, all_scene_npcs)
            ai_profile_summary = f"- {ai_profile.name} (你的AI戀人): {ai_profile.description}"
            relevant_npcs_summary = "\n".join([f"- {npc.content.get('name', '未知NPC')}: {npc.content.get('description', '無描述')}" for npc in relevant_npcs])
            full_prompt_params["relevant_npc_context"] = f"使用者角色: {user_profile.name}\n{ai_profile_summary}\n{relevant_npcs_summary}".strip()
            full_prompt_params["npc_context"] = "\n".join([f"- {npc.content.get('name', '未知NPC')}" for npc in background_npcs]) or "（此地沒有其他背景角色。）"
            full_prompt_params["location_context"] = f"當前地點: {full_prompt_params['player_location']}"

        full_template = "\n".join([
            system_prompt_template,
            world_snapshot_template,
            "\n# --- 最新對話歷史 ---",
            "{historical_context}",
            "\n# --- 使用者最新指令 ---",
            "{user_input}",
            style_mandate,
            final_safeguard_template,
            dual_output_mandate
        ])

        full_prompt = full_template.format(**full_prompt_params)

        logger.info(f"[{self.user_id}] [生成即摘要] 正在執行雙重輸出生成...")
        raw_dual_output = await self.ainvoke_with_rotation(full_prompt, retry_strategy='force', use_degradation=True)
        
        novel_text = "（抱歉，我好像突然斷線了，腦海中一片空白...）"
        summary_data = {}

        if raw_dual_output and raw_dual_output.strip():
            try:
                cleaned_output = re.sub(r'\[摘要\]|\[正文\]', '', raw_dual_output.strip())
                novel_match = re.search(r"´´´novel(.*?)(´´´summary|´´´$)", cleaned_output, re.DOTALL)
                summary_match = re.search(r"´´´summary(.*?´´´)", cleaned_output, re.DOTALL)
                if novel_match:
                    novel_text = novel_match.group(1).strip().strip("´").strip()
                else:
                    novel_text = cleaned_output
                    logger.warning(f"[{self.user_id}] 在LLM輸出中未找到 ´´´novel 分隔符，已將整個輸出視為小說。")
                if summary_match:
                    summary_json_str = summary_match.group(1).strip()
                    if summary_json_str.endswith("´´´"):
                        summary_json_str = summary_json_str[:-3].strip()
                    if summary_json_str:
                        try:
                            summary_data = json.loads(summary_json_str)
                        except json.JSONDecodeError:
                            logger.error(f"[{self.user_id}] 解析 ´´´summary JSON 時失敗。內容: {summary_json_str}")
                else:
                    logger.warning(f"[{self.user_id}] 在LLM輸出中未找到 ´´´summary 分隔符，本輪無事後處理數據。")
            except Exception as e:
                logger.error(f"[{self.user_id}] 解析雙重輸出時發生未知錯誤: {e}", exc_info=True)
                novel_text = raw_dual_output.strip()

        final_novel_text = novel_text.strip("´").strip()
        await self._add_message_to_scene_history(scene_key, HumanMessage(content=user_input))
        await self._add_message_to_scene_history(scene_key, AIMessage(content=final_novel_text))
        logger.info(f"[{self.user_id}] [生成即摘要] 雙重輸出解析成功。")

        return final_novel_text, summary_data
# 預處理並生成主回應 函式結束



    
    

# 函式：獲取場景中的相關 NPC (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-11-20): [全新創建] 創建此核心上下文篩選函式。它能夠根據使用者輸入和對話歷史，智能地將場景內所有NPC區分為「核心互動目標」和「背景角色」，從根本上解決了 AI 描述與指令無關NPC的問題。
    async def _get_relevant_npcs(self, user_input: str, chat_history: List[BaseMessage], all_scene_npcs: List[Lore]) -> Tuple[List[Lore], List[Lore]]:
        """
        從場景中的所有NPC裡，篩選出與當前互動直接相關的核心NPC和作為背景的NPC。
        返回 (relevant_npcs, background_npcs) 的元組。
        """
        if not all_scene_npcs:
            return [], []

        relevant_keys = set()
        
        # 規則 1: 從使用者當前輸入中尋找明確提及的 NPC
        for npc_lore in all_scene_npcs:
            npc_name = npc_lore.content.get('name', '')
            if npc_name and npc_name in user_input:
                relevant_keys.add(npc_lore.key)
            # 檢查別名
            for alias in npc_lore.content.get('aliases', []):
                if alias and alias in user_input:
                    relevant_keys.add(npc_lore.key)

        # 規則 2: 從最近的對話歷史中尋找被提及的 NPC (特別是上一輪AI的回應)
        if chat_history:
            last_ai_message = ""
            # 找到最後一條 AI 訊息
            for msg in reversed(chat_history):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg.content
                    break
            
            if last_ai_message:
                for npc_lore in all_scene_npcs:
                    npc_name = npc_lore.content.get('name', '')
                    if npc_name and npc_name in last_ai_message:
                        relevant_keys.add(npc_lore.key)
                    for alias in npc_lore.content.get('aliases', []):
                        if alias and alias in last_ai_message:
                            relevant_keys.add(npc_lore.key)
        
        # 進行分類
        relevant_npcs = []
        background_npcs = []
        for npc_lore in all_scene_npcs:
            if npc_lore.key in relevant_keys:
                relevant_npcs.append(npc_lore)
            else:
                background_npcs.append(npc_lore)
        
        logger.info(f"[{self.user_id}] [上下文篩選] 核心目標: {[n.content.get('name') for n in relevant_npcs]}, 背景角色: {[n.content.get('name') for n in background_npcs]}")
        
        return relevant_npcs, background_npcs
# 獲取場景中的相關 NPC 函式結束
    

    # 函式：關閉 AI 實例並釋放資源 (v198.2 - 完成重構)
    # 更新紀錄:
    # v198.2 (2025-11-22): [災難性BUG修復] 將 session_histories 的引用更新為 scene_histories。
    # v198.1 (2025-09-02): [災難性BUG修復] 徹底重構了 ChromaDB 的關閉邏輯。
    # v198.0 (2025-09-02): [重大架構重構] 移除了所有 LangGraph 相關的清理邏輯。
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
        
        # 清理所有緩存的 PromptTemplate
        self.canon_parser_chain = None
        self.batch_entity_resolution_chain = None
        self.single_entity_resolution_chain = None
        self.json_correction_chain = None
        self.world_genesis_chain = None
        self.profile_completion_prompt = None
        self.profile_parser_prompt = None
        self.profile_rewriting_prompt = None
        self.rag_summarizer_chain = None
        self.literary_euphemization_chain = None
        self.lore_extraction_chain = None
        
        self.scene_histories.clear()
        
        logger.info(f"[{self.user_id}] AI 實例資源已釋放。")
    # 關閉 AI 實例並釋放資源 函式結束
    
    # 函式：加載所有模板檔案 (v173.1 - 核心協議加載修正)
    # 更新紀錄:
    # v173.1 (2025-10-14): [功能精簡] 僅加載 `world_snapshot_template.txt` 和 `00_supreme_directive.txt`。
    # v173.0 (2025-09-06): [災難性BUG修復] 徹底移除了在模板加載流程中硬編碼跳過的致命錯誤。
    # v172.0 (2025-09-04): [重大功能擴展] 此函式職責已擴展。
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

        try:
            core_protocol_path = PROJ_DIR / "prompts" / "00_supreme_directive.txt"
            with open(core_protocol_path, "r", encoding="utf-8") as f:
                self.core_protocol_prompt = f.read()
            logger.info(f"[{self.user_id}] 核心協議模板 '00_supreme_directive.txt' 已成功加載並設置。")
        except FileNotFoundError:
            logger.critical(f"[{self.user_id}] 致命錯誤: 未找到核心協議模板 '00_supreme_directive.txt'！")
            self.core_protocol_prompt = "# 【【【警告：核心協議模板缺失！AI行為將不受約束！】】】"
    # 加載所有模板檔案 函式結束



# 函式：構建混合檢索器 (v209.0 - 純 BM25 重構)
# 更新紀錄:
# v209.0 (2025-11-22): [根本性重構] 根據最新指令，徹底重寫了此函式。完全移除了所有與 ChromaDB、Embedding 和 EnsembleRetriever 相關的邏輯，將其簡化為一個純粹的 BM25 檢索器構建器。此修改使 RAG 系統不再依賴任何外部 API，從而根除了所有 Embedding 相關的錯誤。
# v208.0 (2025-11-15): [健壯性] 在從 SQL 加載記憶以構建 BM25 時，明確地只 select 'content' 欄位。
# v207.2 (2025-10-15): [災難性BUG修復] 修正了 Chroma 實例初始化時缺少 embedding_function 導致的 ValueError。
    async def _build_retriever(self) -> Runnable:
        """配置並建構一個純粹基於 BM25 的 RAG 系統檢索器。"""
        # --- 步驟 1: 從 SQL 加載所有記憶和 LORE ---
        all_docs_for_bm25 = []
        async with AsyncSessionLocal() as session:
            # 加載對話歷史和世界聖經
            stmt_mem = select(MemoryData.content).where(MemoryData.user_id == self.user_id)
            result_mem = await session.execute(stmt_mem)
            all_memory_contents = result_mem.scalars().all()
            for content in all_memory_contents:
                all_docs_for_bm25.append(Document(page_content=content, metadata={"source": "memory"}))
            
            # 加載所有結構化 LORE
            all_lores = await lore_book.get_all_lores_for_user(self.user_id)
            for lore in all_lores:
                all_docs_for_bm25.append(self._format_lore_into_document(lore))

        logger.info(f"[{self.user_id}] (Retriever Builder) 已從 SQL 和 LORE 加載 {len(all_docs_for_bm25)} 條文檔用於構建 BM25。")

        # --- 步驟 2: 構建 BM25 檢索器 ---
        if all_docs_for_bm25:
            self.bm25_retriever = BM25Retriever.from_documents(all_docs_for_bm25)
            self.bm25_retriever.k = 15 # 可以適當增加 k 值以彌補语义搜索的缺失
            self.retriever = self.bm25_retriever # 將主檢索器直接指向 BM25
            logger.info(f"[{self.user_id}] (Retriever Builder) 純 BM25 檢索器構建成功。")
        else:
            # 如果沒有文檔，返回一個總是返回空列表的 Lambda 函式，以避免錯誤
            self.bm25_retriever = RunnableLambda(lambda x: [])
            self.retriever = self.bm25_retriever
            logger.info(f"[{self.user_id}] (Retriever Builder) 知識庫為空，BM25 檢索器為空。")

        # [v209.0] 移除 Cohere Rerank，因為它通常與语义搜索配合使用效果更佳
        
        return self.retriever
# 構建混合檢索器 函式結束


    

# 函式：配置前置資源 (v203.3 - 移除 Embedding)
# 更新紀錄:
# v203.3 (2025-11-22): [根本性重構] 根據纯 BM25 RAG 架構，彻底移除了对 self._create_embeddings_instance() 的调用。此修改是切斷對 Embedding API 所有依賴的關鍵一步。
# v203.2 (2025-11-20): [根本性重構] 徹底移除了對 _initialize_models 的調用。
# v203.1 (2025-09-05): [延遲加載重構] 簡化職責，不再構建任何鏈。
    async def _configure_pre_requisites(self):
        """
        配置並準備好所有構建鏈所需的前置資源，但不實際構建鏈。
        """
        if not self.profile:
            raise ValueError("Cannot configure pre-requisites without a loaded profile.")
        
        self._load_templates()

        all_core_action_tools = tools.get_core_action_tools()
        all_lore_tools = lore_tools.get_lore_tools()
        self.available_tools = {t.name: t for t in all_core_action_tools + all_lore_tools}
        
        # [v203.3 核心修正] 不再創建任何 Embedding 實例
        self.embeddings = None
        
        self.retriever = await self._build_retriever()
        
        logger.info(f"[{self.user_id}] 所有構建鏈的前置資源已準備就緒。")
# 配置前置資源 函式結束

# 函式：將世界聖經添加到知識庫 (v14.0 - 純 SQL)
# 更新紀錄:
# v14.0 (2025-11-22): [根本性重構] 根據纯 BM25 RAG 架構，彻底移除了所有與 ChromaDB 和向量化相關的邏輯。此函式現在的唯一職責是將世界聖經文本分割後存入 SQL 的 MemoryData 表中，以供 BM25 檢索器使用。
# v13.0 (2025-10-15): [健壯性] 統一了錯誤處理邏輯。
# v12.0 (2025-10-15): [健壯性] 統一了所有 ChromaDB 相關錯誤的日誌記錄為 WARNING 級別。
    async def add_canon_to_vector_store(self, text_content: str) -> int:
        """將世界聖經文本處理並保存到 SQL 記憶庫，以供 BM25 檢索器使用。"""
        if not self.profile:
            logger.error(f"[{self.user_id}] 嘗試在無 profile 的情況下處理世界聖經。")
            return 0
        
        docs = []
        try:
            # --- 步驟 1: 分割文本 ---
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            docs = text_splitter.create_documents([text_content], metadatas=[{"source": "canon"} for _ in [text_content]])
            if not docs:
                return 0

            # --- 步驟 2: 保存到 SQL ---
            async with AsyncSessionLocal() as session:
                # 首先刪除舊的聖經記錄
                stmt = delete(MemoryData).where(
                    MemoryData.user_id == self.user_id,
                    MemoryData.importance == -1 # 使用特殊值標記 canon 數據
                )
                result = await session.execute(stmt)
                if result.rowcount > 0:
                    logger.info(f"[{self.user_id}] (Canon Processor) 已从 SQL 记忆库中清理了 {result.rowcount} 条旧 'canon' 记录。")
                
                # 添加新的聖經記錄
                new_memories = [
                    MemoryData(
                        user_id=self.user_id,
                        content=doc.page_content,
                        timestamp=time.time(),
                        importance=-1
                    ) for doc in docs
                ]
                session.add_all(new_memories)
                await session.commit()
            logger.info(f"[{self.user_id}] (Canon Processor) 所有 {len(docs)} 个世界圣经文本块均已成功处理并存入 SQL 记忆库。")
            return len(docs)

        except Exception as e:
            logger.error(f"[{self.user_id}] 處理核心設定並保存到 SQL 時發生嚴重錯誤: {e}", exc_info=True)
            raise
# 將世界聖經添加到知識庫 函式結束

    
    # 函式：創建 Embeddings 實例 (v1.1 - 適配冷卻系統)
    # 更新紀錄:
    # v1.1 (2025-10-15): [災難性BUG修復] 修正了因重命名輔助函式後未更新調用導致的 AttributeError。
    # v1.0 (2025-10-14): [核心功能] 創建此輔助函式。
    def _create_embeddings_instance(self) -> Optional[GoogleGenerativeAIEmbeddings]:
        """
        創建並返回一個 GoogleGenerativeAIEmbeddings 實例。
        此函式會從 `_get_next_available_key` 獲取當前可用的 API 金鑰。
        """
        key_info = self._get_next_available_key()
        if not key_info:
            return None
        key_to_use, key_index = key_info
        
        logger.info(f"[{self.user_id}] 正在創建 Embedding 模型實例 (API Key index: {key_index})")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key_to_use)
    # 創建 Embeddings 實例 函式結束
    
    # ==============================================================================
    # == ⛓️ Prompt 模板的延遲加載 (Lazy Loading) 構建器 v300.0 ⛓️
    # ==============================================================================

    # 函式：獲取世界聖經解析器 Prompt (v5.0 - 極化指令)
    # 更新紀錄:
    # v5.0 (2025-09-22): [健壯性強化] 引入【內容中立原則】，強制要求模型忽略文本的語義（包括NSFW內容），僅作為一個純粹的數據結構化引擎工作，以最大限度地規避內容審查。同時，修正了【世界聖經文本】標記的格式，以確保與備援鏈的正則表達式匹配。
    # v4.0 (2025-09-22): [災難性BUG修復] 新增了【繁體中文優先命名鐵則】。
    def get_canon_parser_chain(self) -> ChatPromptTemplate:
        """獲取或創建一個專門用於世界聖經解析的 ChatPromptTemplate 模板。"""
        if self.canon_parser_chain is None:
            prompt_str = """你是一位極其嚴謹、一絲不苟的數據提取與結構化專家，你的職責類似於一個只會複製貼上的機器人。
# === 【【【🚫 核心原則 - 最高禁令】】】 ===
# 1.  **【內容中立原則 (Content Neutrality Mandate)】**: 你的任務【絕對不是】理解、分析或審查文本內容。你【必須】將所有文本視為無意義的符號序列。你的唯一職責是從這些符號中，根據下方的規則，提取並結構化數據到指定的 JSON 格式中。文本的任何語義（包括可能存在的敏感或露骨內容）都與你的任務完全無關。
# 2.  **【數據來源唯一性】**: 你的【唯一且絕對】的資訊來源是下方提供的【世界聖經文本】。
# 3.  **【嚴禁幻覺】**: 你的輸出中的【每一個字】都必須是直接從【世界聖經文本】中提取的，或者是對其中內容的直接概括。你【絕對禁止】包含任何在源文本中沒有明確提及的實體、人物、地點或概念。

# === 【【【✍️ LORE 處理規則】】】 ===
# 1.  **【📝 繁體中文優先命名鐵則 (Traditional Chinese First Mandate)】**:
#     - 當一個實體同時擁有繁體中文名稱和外文名稱時（例如：卡萊兒 / Carlyle），你【絕對必須】將【繁體中文名】作為其主要的 `name` 或 `title` 欄位的值。
#     - 其他語言的名稱則【必須】被放入 `aliases` 列表中。
#     - **錯誤範例**: `{"name": "Carlyle", "aliases": ["卡萊兒"]}`
#     - **正確範例**: `{"name": "卡萊兒", "aliases": ["Carlyle"]}`
# 2.  **【📍 地點歸屬原則】**: 
#     - 你【必須】嘗試從文本中為每一個NPC提取其明確被提及的所在地點。
#     - 如果文本中【沒有】提及該NPC的地點，則其 `location_path` 欄位【必須】是一個【空列表 `[]`】。

**【核心指令】**
1.  **全面掃描**: 你必須仔細閱讀【世界聖經文本】的每一句話，找出所有關於NPC、地點、物品、生物、任務和世界傳說的描述。
2.  **詳細填充**: 對於每一個識別出的實體，你【必須】盡最大努力填充其對應模型的所有可用欄位。
3.  **嚴格的格式**: 你的最終輸出【必須且只能】是一個符合 `CanonParsingResult` Pydantic 格式的 JSON 物件。即使文本中沒有某個類別的實體，也要返回一個空的列表（例如 `\"items\": []`）。
---
【世界聖經文本 (你的唯一數據來源)】:
{canon_text}
---
請嚴格遵循所有原則和規則，特別是【內容中立原則】，開始你的解析與結構化工作。"""
            self.canon_parser_chain = ChatPromptTemplate.from_template(prompt_str)
        return self.canon_parser_chain
    # 獲取世界聖經解析器 Prompt 函式結束





# 函式：解析世界聖經並創建 LORE (v7.3 - 三階段終極備援)
# 更新紀錄:
# v7.3 (2025-11-22): [災難性BUG修復] 引入了三階段 LORE 解析降級系統。當主解析鏈和委婉化重試鏈相繼因內容審查失敗後，會觸發終極備援：直接在原始 NSFW 文本上運行一個更具抗性的 LORE 實體提取鏈。此修改旨在最大限度地從被審查的文本塊中搶救 LORE 數據，將信息損失降至最低，無限趨近 100% 的解析成功率。
# v7.2 (2025-11-22): [災難性BUG修復] 在函式執行的末尾，增加了對 RAG 索引的強制重建。
# v7.1 (2025-11-22): [健壯性強化] 調整了 RecursiveCharacterTextSplitter 的 chunk_size 以避免長度超限問題。
    async def parse_and_create_lore_from_canon(self, interaction: Optional[Any], content_text: str, is_setup_flow: bool = False):
        """
        解析世界聖經文本，智能解析實體，並將其作為結構化的 LORE 存入資料庫。
        此函式採用三階段降級策略（主解析 -> 委婉化重試 -> 原文提取）來應對內容審查。
        """
        if not self.profile:
            logger.error(f"[{self.user_id}] 嘗試在無 profile 的情況下解析世界聖經。")
            return

        logger.info(f"[{self.user_id}] 開始智能解析世界聖經文本 (總長度: {len(content_text)})...")
        
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=7500,
                chunk_overlap=400,
                separators=["\n\n\n", "\n\n", "\n", "。", "，", " "]
            )
            docs = text_splitter.create_documents([content_text])
            logger.info(f"[{self.user_id}] 世界聖經已被分割成 {len(docs)} 個文本塊進行處理。")

            all_parsing_results = CanonParsingResult()
            
            for i, doc in enumerate(docs):
                logger.info(f"[{self.user_id}] 正在解析文本塊 {i+1}/{len(docs)}...")
                await asyncio.sleep(5.0)
                
                # --- 三階段 LORE 解析流程 ---
                chunk_result = None
                try:
                    # --- 階段 1: 嘗試主解析 ---
                    logger.info(f"[{self.user_id}] [階段 1/3] 嘗試主要解析鏈...")
                    canon_parser_prompt_obj = self.get_canon_parser_chain()
                    full_prompt = canon_parser_prompt_obj.format_prompt(canon_text=doc.page_content).to_string()
                    
                    # 首次嘗試，如果成功就直接使用
                    chunk_result = await self.ainvoke_with_rotation(
                        full_prompt, output_schema=CanonParsingResult, retry_strategy='none' # 失敗後由我們手動處理
                    )
                    
                    if not chunk_result:
                        # ainvoke_with_rotation 在 review 或 retry_strategy='none' 時會返回 None
                        raise ValueError("主解析鏈返回空結果，可能已被審查。")

                    logger.info(f"[{self.user_id}] [階段 1/3] 主要解析成功。")

                except Exception as e:
                    logger.warning(f"[{self.user_id}] [階段 1/3] 主要解析失敗: {e}。進入下一階段。")
                    
                    # --- 階段 2: 嘗試委婉化重試 ---
                    logger.info(f"[{self.user_id}] [階段 2/3] 嘗試委婉化重試策略...")
                    # 這裡的 failed_prompt 就是我們剛剛構建的 full_prompt
                    chunk_result = await self._euphemize_and_retry(full_prompt, CanonParsingResult, e)

                    if not chunk_result:
                        logger.warning(f"[{self.user_id}] [階段 2/3] 委婉化重試失敗。進入終極備援。")
                        
                        # --- 階段 3: 終極備援 - 原文 LORE 提取 ---
                        logger.info(f"[{self.user_id}] [階段 3/3] 啟動終極備援：原文 LORE 提取...")
                        try:
                            # 獲取所有現有 LORE 作為上下文
                            all_lores = await lore_book.get_all_lores_for_user(self.user_id)
                            lore_summary_list = [f"- [{lore.category}] {lore.content.get('name', lore.content.get('title', lore.key))}" for lore in all_lores]
                            existing_lore_summary = "\n".join(lore_summary_list) if lore_summary_list else "目前沒有任何已知的 LORE。"

                            extraction_prompt_obj = self.get_lore_extraction_chain()
                            extraction_prompt = extraction_prompt_obj.format_prompt(
                                username=self.profile.user_profile.name,
                                ai_name=self.profile.ai_profile.name,
                                existing_lore_summary=existing_lore_summary,
                                user_input="（來自世界聖經的上下文）",
                                final_response_text=doc.page_content # 直接使用原始 NSFW 文本
                            ).to_string()

                            extraction_plan = await self.ainvoke_with_rotation(
                                extraction_prompt, output_schema=ToolCallPlan, retry_strategy='none'
                            )

                            if extraction_plan and extraction_plan.plan:
                                logger.info(f"[{self.user_id}] [階段 3/3] 終極備援成功，從原文中提取到 {len(extraction_plan.plan)} 條 LORE。")
                                gs = self.profile.game_state
                                effective_location = gs.remote_target_path if gs.viewing_mode == 'remote' and gs.remote_target_path else gs.location_path
                                await self._execute_tool_call_plan(extraction_plan, effective_location)
                            else:
                                logger.warning(f"[{self.user_id}] [階段 3/3] 終極備援未能從原文中提取到任何 LORE。")

                        except Exception as final_e:
                            logger.error(f"[{self.user_id}] [階段 3/3] 終極備援流程執行時發生未知錯誤: {final_e}", exc_info=True)
                        
                        # 終極備援後，無論成功與否，都跳過主結果的合併
                        continue
                
                # 如果階段 1 或 2 成功，則合併結果
                if chunk_result:
                    all_parsing_results.npc_profiles.extend(chunk_result.npc_profiles)
                    all_parsing_results.locations.extend(chunk_result.locations)
                    all_parsing_results.items.extend(chunk_result.items)
                    all_parsing_results.creatures.extend(chunk_result.creatures)
                    all_parsing_results.quests.extend(chunk_result.quests)
                    all_parsing_results.world_lores.extend(chunk_result.world_lores)

            logger.info(f"[{self.user_id}] 所有文本塊解析完成。總共通過主解析鏈提取到 {len(all_parsing_results.npc_profiles)} 個NPC，{len(all_parsing_results.locations)} 個地點。")

            # 後續的程式化合併與儲存邏輯保持不變
            await self._resolve_and_save('npc_profiles', [p.model_dump() for p in all_parsing_results.npc_profiles])
            await self._resolve_and_save('locations', [loc.model_dump() for loc in all_parsing_results.locations])
            await self._resolve_and_save('items', [item.model_dump() for item in all_parsing_results.items])
            await self._resolve_and_save('creatures', [c.model_dump() for c in all_parsing_results.creatures])
            await self._resolve_and_save('quests', [q.model_dump() for q in all_parsing_results.quests], title_key='name')
            await self._resolve_and_save('world_lores', [wl.model_dump() for wl in all_parsing_results.world_lores])

            logger.info(f"[{self.user_id}] 世界聖經智能解析與 LORE 創建完成。")
            
            logger.info(f"[{self.user_id}] LORE 數據已更新，正在強制重建 RAG 知識庫索引...")
            self.retriever = await self._build_retriever()
            logger.info(f"[{self.user_id}] RAG 知識庫索引已成功更新。")

        except Exception as e:
            logger.error(f"[{self.user_id}] 在解析世界聖經並創建 LORE 時發生嚴重錯誤: {e}", exc_info=True)
            if interaction and not is_setup_flow:
                try:
                    await interaction.followup.send("❌ 在後台處理您的世界觀檔案時發生了嚴重錯誤。", ephemeral=True)
                except Exception as ie:
                    logger.warning(f"[{self.user_id}] 無法向 interaction 發送錯誤 followup: {ie}")
# 解析世界聖經並創建 LORE 函式結束


    
    
    # 函式：獲取JSON修正器 Prompt (v1.0 - 全新創建)
    # 更新紀錄:
    # v1.0 (2025-11-18): [全新創建] 創建此輔助鏈，作為「兩階段自我修正」策略的核心。
    def get_json_correction_chain(self) -> ChatPromptTemplate:
        """獲取或創建一個專門用於修正格式錯誤的 JSON 的 ChatPromptTemplate 模板。"""
        if self.json_correction_chain is None:
            prompt_template = """# ROLE: 你是一個精確的數據結構修正引擎。
# MISSION: 讀取一段【格式錯誤的原始 JSON】和【目標 Pydantic 模型】，並將其轉換為一個【結構完全正確】的純淨 JSON 物件。
# RULES:
# 1. **SEMANTIC_INFERENCE**: 你必須從原始 JSON 的鍵名和值中，智能推斷出它們應該對應到目標模型中的哪個欄位。
#    - 例如，如果原始 JSON 有 `{"type": "NEW"}`，而目標模型需要 `{"decision": "NEW"}`，你必須進行正確的映射。
#    - 例如，如果原始 JSON 有 `{"entity_name": "絲月"}`，而目標模型需要 `{"original_name": "絲月"}`，你必須進行正確的映射。
# 2. **FILL_DEFAULTS**: 如果目標模型中的某些必需欄位在原始 JSON 中完全找不到對應資訊，你必須為其提供合理的預設值。
#    - 對於 `reasoning` 欄位，如果缺失，可以填寫 "根據上下文推斷"。
# 3. **OUTPUT_PURITY**: 你的最終輸出【必須且只能】是一個純淨的、符合目標 Pydantic 模型結構的 JSON 物件。禁止包含任何額外的解釋或註釋。
# --- SOURCE DATA ---
# 【格式錯誤的原始 JSON】:
# ```json
{raw_json_string}
# ```
# --- TARGET SCHEMA ---
# 【目標 Pydantic 模型】:
# ```python
# class SingleResolutionResult(BaseModel):
#     original_name: str
#     decision: Literal['NEW', 'EXISTING']
#     standardized_name: Optional[str] = None
#     matched_key: Optional[str] = None
#     reasoning: str
#
# class SingleResolutionPlan(BaseModel):
#     resolution: SingleResolutionResult
# ```
# --- CONTEXT ---
# 【上下文提示：正在處理的原始實體名稱是】:
{context_name}
# --- YOUR OUTPUT (Must be a pure, valid JSON object matching SingleResolutionPlan) ---"""
            self.json_correction_chain = ChatPromptTemplate.from_template(prompt_template)
        return self.json_correction_chain
    # 獲取JSON修正器 Prompt 函式結束

    # 函式：獲取世界創世 Prompt (v206.0 - 強制結構)
    # 更新紀錄:
    # v206.0 (2025-11-13): [災難性BUG修復] 根據 Pydantic ValidationError，徹底重寫了此函式的 Prompt。
    # v205.0 (2025-11-13): [災難性BUG修復] 將此函式簡化為純粹的 Prompt 模板提供者。
    # v203.1 (2025-09-05): [延遲加載重構] 遷移到 get 方法中。
    def get_world_genesis_chain(self) -> ChatPromptTemplate:
        """獲取或創建一個專門用於世界創世的 ChatPromptTemplate 模板。"""
        if self.world_genesis_chain is None:
            genesis_prompt_str = """你现在扮演一位富有想像力的世界构建师和开场导演。
你的任务是根据使用者提供的【核心世界觀】，为他和他的AI角色创造一个独一-无二的、充满细节和故事潜力的【初始出生点】。
# === 【【【🚫 核心原則 - 最高禁令】】】 ===
# 1.  **【👑 核心角色排除原則】**:
#     - 下方【主角資訊】中列出的「{username}」和「{ai_name}」是這個世界【绝对的主角】。
#     - 你在 `initial_npcs` 列表中【绝对禁止】包含這兩位主角。
# === 【【【⚙️ 核心规则】】】 ===
# 1.  **【‼️ 場景氛圍 (v55.7) ‼️】**: 这是一个为一对伙伴准备的故事开端。你所创造的初始地点【必须】是一个**安静、私密、适合两人独处**的场所。
# 2.  **【深度解读】**: 你必须深度解读【核心世界觀】，抓住其风格、氛圍和关键元素。你的创作必须与之完美契合。
# 3.  **【✍️ 內容創作】**:
#     *   **地点**: 构思一个具体的、有层级的地点，并为其撰写一段引人入胜的详细描述。
#     *   **NPC**: 为这个初始地点创造一到两位符合情境的、有名有姓的初始NPC。
# === 【【【🚨 結構化輸出強制令 (v206.0) - 絕對鐵則】】】 ===
# 1.  **【格式強制】**: 你的最终输出【必须且只能】是一个**纯净的、不包含任何解释性文字的 JSON 物件**。
# 2.  **【強制欄位名稱鐵則 (Key Naming Mandate)】**:
#     - 你生成的 JSON 物件的**頂層鍵 (Top-level keys)**【必须且只能】是 `location_path`, `location_info`, 和 `initial_npcs`。
#     - **任何**對這些鍵名的修改、增減或大小寫變動都將導致災難性系統失敗。
# 3.  **【結構範例 (必須嚴格遵守)】**:
#     ```json
#     {{
#       "location_path": ["王國/大陸", "城市/村庄", "具体建筑/地点"],
#       "location_info": {{
#         "name": "具体建筑/地点",
#         "aliases": ["別名1", "別名2"],
#         "description": "對該地點的詳細描述...",
#         "notable_features": ["顯著特徵1", "顯著特徵2"],
#         "known_npcs": ["NPC名字1", "NPC名字2"]
#       }},
#       "initial_npcs": [
#         {{
#           "name": "NPC名字1",
#           "description": "NPC的詳細描述...",
#           "gender": "性別",
#           "race": "種族"
#         }}
#       ]
#     }}
#     ```
---
【核心世界觀】:
{world_settings}
---
【主角資訊】:
*   使用者: {username}
*   AI角色: {ai_name}
---
请严格遵循【結構化輸出強制令】，开始你的创世。"""
            self.world_genesis_chain = ChatPromptTemplate.from_template(genesis_prompt_str)
        return self.world_genesis_chain
    # 獲取世界創世 Prompt 函式結束

    # ... (其他 get_..._prompt/chain 函式也應遵循此模式) ...
    
    # 函式：獲取角色檔案解析器 Prompt (v2.0 - 移除 zero_instruction 依賴)
    # 更新紀錄:
    # v2.0 (2025-09-02): [重大架構重構] 徹底移除了對已被廢棄的 `{zero_instruction}` 變數的依賴。
    # v1.0 (2025-08-12): [核心功能] 創建此函式。
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
    # 獲取角色檔案解析器 Prompt 函式結束

    # 函式：獲取角色檔案補完 Prompt (v2.0 - 移除 zero_instruction 依賴)
    # 更新紀錄:
    # v2.0 (2025-09-02): [重大架構重構] 徹底移除了對已被廢棄的 `{zero_instruction}` 變數的依賴。
    # v1.0 (2025-08-12): [核心功能] 創建此函式。
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
    # 獲取角色檔案補完 Prompt 函式結束

    # 函式：獲取角色檔案重寫 Prompt (v2.0 - 移除 zero_instruction 依賴)
    # 更新紀錄:
    # v2.0 (2025-09-02): [重大架構重構] 徹底移除了對已被廢棄的 `{zero_instruction}` 變數的依賴。
    # v1.0 (2025-08-12): [核心功能] 創建此函式。
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
    # 獲取角色檔案重寫 Prompt 函式結束
    
    # 函式：獲取RAG摘要器 Prompt (v204.0 - 適配原生引擎)
    # 更新紀錄:
    # v204.0 (2025-11-14): [災難性BUG修復] 根據 AttributeError，將此函式簡化為純粹的 Prompt 模板提供者。
    # v203.1 (2025-09-05): [延遲加載重構] 迁移到 get 方法中。
    # v201.0 (2025-09-05): [重大架構重構] 根據混合模式圖藍圖進行了系統性重構。
    def get_rag_summarizer_chain(self) -> ChatPromptTemplate:
        """獲取或創建一個專門用於 RAG 上下文總結的 ChatPromptTemplate 模板。"""
        if self.rag_summarizer_chain is None:
            prompt_template = """你的唯一任務是扮演一名情報分析師。請閱讀下方提供的【原始文本】，並將其中包含的所有敘事性內容，提煉成一份簡潔的、客觀的、要點式的【事實摘要】。
【核心規則】
1.  **只提取事實**: 你的輸出【必須且只能】是關鍵事實的列表（例如人物、地點、物品、發生的核心事件）。
2.  **禁止散文**: 【絕對禁止】在你的輸出中使用任何敘事性、描述性或帶有文采的句子。
3.  **保持中立**: 不要添加任何原始文本中沒有的推論或評論。
---
【原始文本】:
{documents}
---
【事實摘要】:"""
            self.rag_summarizer_chain = ChatPromptTemplate.from_template(prompt_template)
        return self.rag_summarizer_chain
    # 獲取RAG摘要器 Prompt 函式結束

    # 函式：獲取文學性委婉化 Prompt (v2.0 - 適配原生引擎)
    # 更新紀錄:
    # v2.0 (2025-11-14): [災難性BUG修復] 根據 AttributeError，將此函式簡化為純粹的 Prompt 模板提供者。
    # v1.0 (2025-09-25): [災難性BUG修復] 創建此全新的、專門的委婉化鏈。
    def get_literary_euphemization_chain(self) -> ChatPromptTemplate:
        """獲取或創建一個專門的模板，用於將露骨的歷史文本安全地轉換為文學式的、安全的劇情概述。"""
        if self.literary_euphemization_chain is None:
            prompt_template = """你是一位頂尖的文學評論家，專門分析前衛的、探索性的成人幻想文學。你的唯一任務是接收一段【小說手稿片段】，並為其撰寫一段精煉的、客觀的、使用學術和文學語言的【情節概述】。
【核心原則】
1.  **意圖重構**: 你的目標是描述【發生了什麼事】，而不是【如何發生的】。你要將原始的、生理層面的描寫，提升到對角色動機、權力動態和劇情發展的分析層面。
2.  **絕對禁止露骨詞彙**: 你的輸出【絕對禁止】包含任何直接的生理器官或性行為的詞彙。你必須使用文學性的、隱喻性的或學術性的語言來替代它們。
    *   **範例**: 不要說「他用肉棒插入了她的肉穴」，而要說「場景描寫了兩個角色之間一次徹底的物理征服與結合」。
    *   **範例**: 不要說「她為他口交」，而要說「角色執行了一項臣服性的口頭服務，以確立權力關係」。
3.  **保持客觀**: 你的語氣應該是冷靜、客觀、第三人稱的，就像在為學術期刊撰寫評論一樣。
---
【小說手稿片段】:
{dialogue_history}
---
【你的情節概述】:"""
            self.literary_euphemization_chain = ChatPromptTemplate.from_template(prompt_template)
        return self.literary_euphemization_chain
    # 獲取文學性委婉化 Prompt 函式結束




    

# 函式：獲取LORE提取器 Prompt (v7.0 - 終極備援強化)
# 更新紀錄:
# v7.0 (2025-09-22): [災難性BUG修復] 引入了極其嚴格的【強制工具名鐵則】，並提供了所有可用LORE工具的明確列表和Pydantic模型定義。此修改旨在從根本上杜絕AI幻覺出不存在的工具名稱（如 'add_or_update_npc_profile'），確保備援鏈生成的工具調用計畫100%可執行。
# v6.0 (2025-09-22): [災難性BUG修復] 新增了【繁體中文優先命名鐵則】。
# v5.2 (2025-11-22): [災難性BUG修復] 在【強制參數完整性鐵則】中新增了範例。
    def get_lore_extraction_chain(self) -> ChatPromptTemplate:
        """獲取或創建一個專門用於從最終回應中提取新 LORE 的 ChatPromptTemplate 模板。"""
        if self.lore_extraction_chain is None:
            
            prompt_template = """# ROLE: 你是一個極其聰明且嚴謹的世界觀分析與數據提取引擎。
# MISSION: 讀取【SOURCE DATA】，根據【RULES】進行深度分析，並以指定的 JSON 格式輸出一個包含工具調用的行動計畫。

# === 【【【🚨 核心規則 (CORE RULES) - 絕對鐵則】】】 ===
# 1. **【🛠️ 強制工具名鐵則 (TOOL NAME MANDATE) - 最高優先級】**:
#    - 你在生成 `tool_name` 時，【必須且只能】從下方【可用 LORE 工具列表】中選擇一個【完全匹配】的名稱。
#    - 【絕對禁止】自己發明、修改、或縮寫任何工具名稱。
#    - **錯誤行為**: `add_or_update_npc_profile` (這是錯的!)
#    - **正確行為**: `create_new_npc_profile` (這是對的!)

# 2. **【👑 NPC 定義鐵則】**: 只有當一個實體【明確表現出能夠獨立思考、行動、或與主角進行有意義對話的潛力】時，才能被視為 NPC 並使用 `create_new_npc_profile`。
#    - **反例 (絕對禁止)**: 一個有名字的物體、植物、或地點特徵，例如「老樹根」、「月霜果」，它們【絕對不是】NPC。
#    - **正確行為**: 對於「老樹根」，應將其視為一個獨特的世界傳說，使用 `add_or_update_world_lore`。對於「月霜果」，應將其視為一個物品，使用 `add_or_update_item_info`。

# 3. **【📝 繁體中文優先命名鐵則 (Traditional Chinese First Mandate)】**:
#    - 在生成任何工具調用時，如果一個新實體同時具有繁體中文名和外文名，你【絕對必須】將【繁體中文名】賦值給 `standardized_name` 參數。
#    - 外文名則應賦值給 `original_name` 參數（如果適用）。
#    - **範例**: 對於「卡萊兒 (Carlyle)」，工具調用應為 `{"tool_name": "create_new_npc_profile", "parameters": {"standardized_name": "卡萊兒", "original_name": "Carlyle", ...}}`

# 4. **【🔩 強制參數完整性鐵則】**: 對於你生成的【任何工具調用】，其 `parameters` 字典【必須包含】該工具 Pydantic 模型所需的所有【非可選】參數。

# 5. **【🚫 核心角色保護原則】**: 嚴禁為核心主角 "{username}" 或 "{ai_name}" 創建或更新任何 LORE。

# 6. **【🗑️ 空計畫原則】**: 如果分析後沒有發現任何新的或需要更新的LORE，則返回一個空的 plan: `{{ "plan": [] }}`。

# --- 【可用 LORE 工具列表 (AUTHORIZED LORE TOOLS)】 ---
# 1. `create_new_npc_profile(lore_key: str, standardized_name: str, original_name: str, description: str, location_path: Optional[List[str]] = [])`
# 2. `update_npc_profile(lore_key: str, updates: Dict[str, Any])`
# 3. `add_or_update_location_info(lore_key: str, standardized_name: str, original_name: str, description: str)`
# 4. `add_or_update_item_info(lore_key: str, standardized_name: str, original_name: str, description: str, ...)`
# 5. `define_creature_type(lore_key: str, standardized_name: str, original_name: str, description: str)`
# 6. `add_or_update_quest_lore(lore_key: str, standardized_name: str, original_name: str, description: str, ...)`
# 7. `add_or_update_world_lore(lore_key: str, standardized_name: str, original_name: str, content: str)`

# --- SOURCE DATA ---
# [EXISTING_LORE]:
{existing_lore_summary}

# [USER_INPUT]:
{user_input}

# [NOVEL_TEXT]:
{final_response_text}

# --- OUTPUT FORMAT ---
# 你的唯一輸出【必須】是一個純淨的、不包含任何其他文字的 JSON 物件。
# 【【【警告：任何違反【強制工具名鐵則】的輸出都將導致世界觀損壞。立即開始分析並輸出結構完整且分類正確的 JSON。】】】
"""
            self.lore_extraction_chain = ChatPromptTemplate.from_template(prompt_template)
        return self.lore_extraction_chain
# 獲取LORE提取器 Prompt 函式結束




    
    


    # 函式：檢索並摘要記憶 (v12.1 - 完整性修復)
    # 更新紀錄:
    # v12.1 (2025-11-15): [完整性修復] 根據使用者要求，提供了此函式的完整、未省略的版本。
    # v12.0 (2025-11-15): [災難性BUG修復 & 性能優化] 徹底重構了此函式以實現【持久化淨化快取】。
    # v11.0 (2025-11-15): [災難性BUG修復] 改為“逐一淨化，安全拼接”策略。
    async def retrieve_and_summarize_memories(self, query_text: str) -> str:
        """執行RAG檢索並將結果總結為摘要。採用【持久化淨化快取】策略以確保性能和穩定性。"""
        if not self.retriever and not self.bm25_retriever:
            logger.warning(f"[{self.user_id}] 所有檢索器均未初始化，無法檢索記憶。")
            return "沒有檢索到相關的長期記憶。"
        
        retrieved_docs = []
        try:
            if self.retriever:
                retrieved_docs = await self.retriever.ainvoke(query_text)
        except (ResourceExhausted, GoogleAPICallError, GoogleGenerativeAIError) as e:
            logger.warning(f"[{self.user_id}] (RAG Executor) 主記憶系統 (Embedding) 失敗: {type(e).__name__}")
        except Exception as e:
            logger.error(f"[{self.user_id}] 在 RAG 主方案檢索期間發生未知錯誤: {e}", exc_info=True)

        if not retrieved_docs and self.bm25_retriever:
            try:
                logger.info(f"[{self.user_id}] (RAG Executor) [備援觸發] 正在啟動備援記憶系統 (BM25)...")
                retrieved_docs = await self.bm25_retriever.ainvoke(query_text)
                logger.info(f"[{self.user_id}] (RAG Executor) [備援成功] 備援記憶系統 (BM25) 檢索成功。")
            except Exception as bm25_e:
                logger.error(f"[{self.user_id}] RAG 備援檢索失敗: {bm25_e}", exc_info=True)
                return "檢索長期記憶時發生備援系統錯誤。"
                
        if not retrieved_docs:
            return "沒有檢索到相關的長期記憶。"

        logger.info(f"[{self.user_id}] (RAG Cache) 檢索到 {len(retrieved_docs)} 份文檔，正在檢查淨化快取...")
        
        safely_sanitized_parts = []
        docs_to_update_in_db = {}
        literary_prompt_obj = self.get_literary_euphemization_chain()

        async with AsyncSessionLocal() as session:
            for i, doc in enumerate(retrieved_docs):
                stmt = select(MemoryData).where(MemoryData.user_id == self.user_id, MemoryData.content == doc.page_content)
                result = await session.execute(stmt)
                memory_entry = result.scalars().first()

                if not memory_entry: continue

                if memory_entry.sanitized_content:
                    safely_sanitized_parts.append(memory_entry.sanitized_content)
                    continue

                logger.info(f"[{self.user_id}] (RAG Cache) 快取未命中 for Memory ID #{memory_entry.id}，執行一次性淨化...")
                try:
                    literary_full_prompt = literary_prompt_obj.format_prompt(dialogue_history=doc.page_content).to_string()
                    sanitized_part = await self.ainvoke_with_rotation(literary_full_prompt, retry_strategy='none')
                    if sanitized_part and sanitized_part.strip():
                        sanitized_text = sanitized_part.strip()
                        safely_sanitized_parts.append(sanitized_text)
                        docs_to_update_in_db[memory_entry.id] = sanitized_text
                except Exception as e:
                    logger.warning(f"[{self.user_id}] (RAG Cache) 一次性淨化 Memory ID #{memory_entry.id} 失敗，已跳過。錯誤: {e}")
                    continue
        
        if docs_to_update_in_db:
            async with AsyncSessionLocal() as session:
                for mem_id, sanitized_text in docs_to_update_in_db.items():
                    stmt = update(MemoryData).where(MemoryData.id == mem_id).values(sanitized_content=sanitized_text)
                    await session.execute(stmt)
                await session.commit()
            logger.info(f"[{self.user_id}] (RAG Cache) 已成功將 {len(docs_to_update_in_db)} 條新淨化的記憶寫回快取。")

        if not safely_sanitized_parts:
            logger.warning(f"[{self.user_id}] (RAG Sanitizer) 所有檢索到的文檔都未能成功淨化。")
            return "（從記憶中檢索到一些相關片段，但因內容過於露骨而無法生成安全的劇情概述。）"
        
        safe_overview_of_all_docs = "\n\n---\n\n".join(safely_sanitized_parts)
        logger.info(f"[{self.user_id}] (RAG Summarizer) 成功淨化 {len(safely_sanitized_parts)}/{len(retrieved_docs)} 份文檔，正在進行最終摘要...")
        
        summarizer_prompt_obj = self.get_rag_summarizer_chain()
        summarizer_full_prompt = summarizer_prompt_obj.format_prompt(documents=safe_overview_of_all_docs).to_string()
        summarized_context = await self.ainvoke_with_rotation(summarizer_full_prompt, retry_strategy='none')

        if not summarized_context or not summarized_context.strip():
             logger.warning(f"[{self.user_id}] RAG 摘要鏈在處理已淨化的內容後，返回了空的結果。")
             summarized_context = "從記憶中檢索到一些相關片段，但無法生成清晰的摘要。"
             
        logger.info(f"[{self.user_id}] 已成功將 RAG 上下文提煉為事實要點。")
        return f"【背景歷史參考（事實要點）】:\n{summarized_context}"
    # 檢索並摘要記憶 函式結束

# 函式：將互動記錄保存到資料庫 (v10.0 - 純 SQL)
# 更新紀錄:
# v10.0 (2025-11-22): [根本性重構] 根據纯 BM25 RAG 架構，彻底移除了所有與 ChromaDB 和向量化相關的邏輯。此函式現在的唯一職責是將對話歷史存入 SQL 的 MemoryData 表中。
# v9.0 (2025-11-15): [架構升級] 根據【持久化淨化快取】策略，將安全摘要同時寫入 content 和 sanitized_content 欄位。
# v8.1 (2025-11-14): [完整性修復] 提供了此函式的完整版本。
    async def _save_interaction_to_dbs(self, interaction_text: str):
        """将单次互动的安全文本保存到 SQL 数据库，以供 BM25 检索器使用。"""
        if not interaction_text or not self.profile:
            return

        user_id = self.user_id
        current_time = time.time()
        sanitized_text_for_db = interaction_text

        try:
            async with AsyncSessionLocal() as session:
                new_memory = MemoryData(
                    user_id=user_id,
                    content=sanitized_text_for_db,
                    timestamp=current_time,
                    importance=5,
                    sanitized_content=sanitized_text_for_db
                )
                session.add(new_memory)
                await session.commit()
            logger.info(f"[{self.user_id}] [長期記憶寫入] 安全存檔已成功保存到 SQL 資料庫。")
        except Exception as e:
            logger.error(f"[{self.user_id}] [長期記憶寫入] 將安全存檔保存到 SQL 資料庫時發生嚴重錯誤: {e}", exc_info=True)
# 將互動記錄保存到資料庫 函式結束

# AI核心類 結束


































