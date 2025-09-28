# schemas.py v3.0 (關係結構化 & 致命BUG修復)
# 更新紀錄:
# v3.0 (2025-09-27): [災難性BUG修復] 補全了缺失的 LoreClassificationResult 和 BatchClassificationResult 類定義，並將 WorldLore 的 'title' 統一為 'name' 以解決 ValidationError。
# v2.0 (2025-09-27): [重大架構升級] 新增了 RelationshipDetail 模型，並將 CharacterProfile.relationships 升級為結構化字典，同時增加了向下兼容的驗證器。
# v1.4 (2025-09-26): [災難性BUG修復] 在文件頂部增加了所有運行FastAPI Web伺服器所需的、缺失的import語句。

import json
import re
from typing import Optional, Dict, List, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, AliasChoices

# --- 基礎驗證器 ---
def _validate_string_to_list(value: Any) -> Any:
    if isinstance(value, str):
        items = re.split(r'[，,、;\n]', value)
        return [item.strip() for item in items if item and item.strip()]
    if isinstance(value, list):
        processed_list = []
        for item in value:
            if isinstance(item, dict) and 'name' in item:
                processed_list.append(item['name'])
            elif isinstance(item, str):
                processed_list.append(item)
        return processed_list
    return value

def _validate_string_to_dict(value: Any) -> Any:
    if isinstance(value, str):
        if value.strip().lower() in ["無", "未知", "", "none", "null"]:
            return {}
        try:
            return json.loads(value.replace("'", '"'))
        except json.JSONDecodeError:
            return {"summary": value}
    return value

# src/schemas.py 的 CharacterSkeleton 和 ExtractionResult 類別 (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-09-28): [全新創建] 根據「多階段混合解析管線」架構，創建這兩個模型。`CharacterSkeleton` 用於定義第一階段（骨架提取）的最小化數據結構（僅含名稱和原始描述），而 `ExtractionResult` 則用於包裹批量的骨架提取結果。此舉將複雜的解析任務分解，確保了第一階段的快速、高效和高成功率。
class CharacterSkeleton(BaseModel):
    """用於混合 NLP 第一階段，表示一個角色的最基本骨架信息。"""
    name: str = Field(description="角色的名字。必須是文本中明確提到的、最常用的名字。")
    raw_description: str = Field(description="一段包含了所有與該角色相關的、從世界聖經原文中提取出的、未經處理的完整文本片段。")

class ExtractionResult(BaseModel):
    """包裹第一階段實體骨架提取結果的模型。"""
    characters: List[CharacterSkeleton] = Field(description="從文本中提取出的所有潛在角色實體的骨架列表。")

# [v2.0 新增] 用於結構化關係的模型
class RelationshipDetail(BaseModel):
    """儲存一個角色對另一個角色的詳細關係資訊"""
    type: str = Field(default="社交關係", description="關係的類型，例如 '家庭', '主從', '敵對', '戀愛', '社交關係'。")
    roles: List[str] = Field(default_factory=list, description="對方在此關係中扮演的角色或稱謂列表，支持多重身份。例如 ['女兒', '學生']。")

class CharacterProfile(BaseModel):
    name: str = Field(description="角色的標準化、唯一的官方名字。")
    aliases: List[str] = Field(default_factory=list, description="此角色的其他已知稱呼或別名。")
    alternative_names: List[str] = Field(default_factory=list, description="一个由AI预先生成的、用于在主名称冲突时备用的名称列表。")
    gender: str = Field(default="未設定", description="角色的性別。")
    age: str = Field(default="未知", description="角色的年齡或年齡段。")
    race: str = Field(default="未知", description="角色的種族。")
    appearance: str = Field(default="", description="角色的外貌特徵的總體描述。")
    appearance_details: Dict[str, Any] = Field(default_factory=dict, description="角色的具體外貌細節，值可以是字串或列表。")
    likes: List[str] = Field(default_factory=list, description="角色喜歡的事物列表。")
    dislikes: List[str] = Field(default_factory=list, description="角色不喜歡的事物列表。")
    equipment: List[str] = Field(default_factory=list, description="角色【當前穿戴或持有】的裝備列表。")
    skills: List[str] = Field(default_factory=list, description="角色掌握的技能列表。")
    description: str = Field(default="", description="角色的性格、背景故事、行為模式等綜合簡介。")
    location: str = Field(default="", description="角色當前所在的城市或主要區域。")
    location_path: List[str] = Field(default_factory=list, description="角色當前所在的層級式地點路徑。")
    affinity: int = Field(default=0, description="此角色對使用者的好感度。")
    # [v2.0 核心修正] 升級 relationships 欄位
    relationships: Dict[str, RelationshipDetail] = Field(default_factory=dict, description="記錄此角色與其他角色的結構化關係。例如：{'莉莉絲': {'type': '家庭', 'roles': ['女兒']}}")
    status: str = Field(default="健康", description="角色的當前健康或狀態。")
    current_action: str = Field(default="站著", description="角色當前正在進行的、持續性的動作或所處的姿態。")

    @field_validator('aliases', 'likes', 'dislikes', 'equipment', 'skills', 'location_path', 'alternative_names', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        if isinstance(value, str) and (' > ' in value or '/' in value):
            return [part.strip() for part in re.split(r'\s*>\s*|/', value)]
        return _validate_string_to_list(value)

    @field_validator('appearance_details', mode='before')
    @classmethod
    def _validate_string_to_dict_fields(cls, value: Any) -> Any:
        return _validate_string_to_dict(value)

    # [v2.0 核心修正] 增加強大的向下兼容驗證器
    @field_validator('relationships', mode='before')
    @classmethod
    def _validate_and_normalize_relationships(cls, value: Any) -> Dict[str, Any]:
        """
        一個強大的驗證器，用於處理和規範化 'relationships' 字典。
        它可以向下兼容舊的扁平化字串格式，並將其自動轉換為新的 RelationshipDetail 結構。
        """
        if isinstance(value, str):
            value = _validate_string_to_dict(value)
        if not isinstance(value, dict):
            return {}
            
        normalized_dict = {}
        for k, v in value.items():
            # 兼容舊格式: {"莉莉絲": "女兒"}
            if isinstance(v, str):
                normalized_dict[str(k)] = RelationshipDetail(roles=[v])
            # 兼容舊格式: {"卡爾": 100} (好感度)
            elif isinstance(v, int):
                 normalized_dict[str(k)] = RelationshipDetail(type="好感度", roles=[str(v)])
            # 處理新格式或已經是字典的格式
            elif isinstance(v, dict):
                try:
                    # 嘗試直接驗證為新模型
                    normalized_dict[str(k)] = RelationshipDetail.model_validate(v)
                except Exception:
                    # 如果失敗，則作為通用字典處理
                    roles = [str(role) for role in v.get("roles", [])]
                    type_str = v.get("type", "社交關係")
                    normalized_dict[str(k)] = RelationshipDetail(type=type_str, roles=roles)
            # 其他類型，盡力轉換
            else:
                normalized_dict[str(k)] = RelationshipDetail(roles=[str(v)])
        return normalized_dict

class BatchRefinementResult(BaseModel):
    """包裹第二階段批量深度精煉結果的模型。"""
    refined_profiles: List[CharacterProfile] = Field(description="一個包含所有被成功精煉後的角色檔案的列表。")

class Quest(BaseModel):
    name: str = Field(description="任務的標準化、唯一的官方名稱。")
    aliases: List[str] = Field(default_factory=list, description="此任務的其他已知稱呼或別名。")
    description: str = Field(default="", description="任務的詳細描述和目標。")
    status: str = Field(default="active", description="任務的當前狀態，例如 'active', 'completed', 'failed'。")
    quest_giver: Optional[str] = Field(default=None, description="此任務的發布者（NPC名字）。")
    suggested_level: Optional[int] = Field(default=None, description="建議執行此任務的角色等級。")
    rewards: Dict[str, Any] = Field(default_factory=dict, description="完成任務的獎勵，例如 {{'金錢': 100, '物品': ['治療藥水']}}。")

    @field_validator('aliases', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        return _validate_string_to_list(value)
        
    @field_validator('rewards', mode='before')
    @classmethod
    def _validate_string_to_dict_fields(cls, value: Any) -> Any:
        return _validate_string_to_dict(value)

    @field_validator('suggested_level', mode='before')
    @classmethod
    def _parse_int_from_string(cls, value: Any) -> Optional[int]:
        if isinstance(value, str):
            match = re.search(r'\d+', value)
            if match:
                try:
                    return int(match.group(0))
                except (ValueError, TypeError):
                    return None
        return value

class LocationInfo(BaseModel):
    name: str = Field(description="地點的標準化、唯一的官方名稱。")
    aliases: List[str] = Field(default_factory=list, description="此地點的其他已知稱呼或別名。")
    description: str = Field(default="", description="對該地點的詳細描述，包括環境、氛圍、建築風格等。")
    notable_features: List[str] = Field(default_factory=list, description="該地點的顯著特徵或地標列表。")
    known_npcs: List[str] = Field(default_factory=list, description="已知居住或出現在此地點的 NPC 名字列表。")

    @field_validator('aliases', 'notable_features', 'known_npcs', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

class ItemInfo(BaseModel):
    name: str = Field(description="道具的標準化、唯一的官方名稱。")
    aliases: List[str] = Field(default_factory=list, description="此物品的其他已知稱呼或別名。")
    description: str = Field(default="", description="對該道具的詳細描述，包括其歷史、材質、背景故事等。")
    item_type: str = Field(default="未知", description="道具的類型，例如 '消耗品', '武器', '關鍵物品', '盔甲', '服裝'。")
    effect: str = Field(default="無", description="道具的使用效果描述。")
    rarity: str = Field(default="普通", description="道具的稀有度，例如 '普通', '稀有', '史詩', '傳說'。")
    visual_description: Optional[str] = Field(default="", description="對道具外觀的詳細、生動的描寫。")
    origin: Optional[str] = Field(default="", description="關於該道具來源或製造者的簡短傳說。")

    @field_validator('aliases', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

class CreatureInfo(BaseModel):
    name: str = Field(description="生物/魔物的標準化、唯一的官方種類名稱（例如 '水晶雞'）。")
    aliases: List[str] = Field(default_factory=list, description="此生物的其他已知稱呼或別名。")
    description: str = Field(default="", description="對該生物/魔物的詳細描述，包括外貌、習性、生態地位等。")
    abilities: List[str] = Field(default_factory=list, description="該生物/魔物的特殊能力列表。")
    habitat: List[str] = Field(default_factory=list, description="該生物/魔物的主要棲息地列表。")

    @field_validator('aliases', 'abilities', 'habitat', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

# src/schemas.py 的 WorldLore 類別 (v2.1 - 規則繼承)
# 更新紀錄:
# v2.1 (2025-11-22): [架構擴展] 新增了 template_keys 欄位。這是實現「LORE繼承與規則注入系統」的應用層基礎，用於定義一條世界傳說可以被哪些身份（aliases）所繼承。
# v2.0 (2025-09-27): [重大架構升級] 新增了 RelationshipDetail 模型，並將 CharacterProfile.relationships 升級為結構化字典。
# v1.4 (2025-09-26): [災難性BUG修復] 在文件頂部增加了所有運行FastAPI Web伺服器所需的、缺失的import語句。
class WorldLore(BaseModel):
    # [v3.0 核心修正] 將 'title' 統一為 'name' 並保持兼容性
    name: str = Field(description="這條傳說、神話或歷史事件的標準化、唯一的官方標題。", validation_alias=AliasChoices('name', 'title'))
    aliases: List[str] = Field(default_factory=list, description="此傳說的其他已知稱呼或別名。")
    content: str = Field(default="", description="詳細的內容描述。")
    category: str = Field(default="未知", description="Lore 的分類，例如 '神話', '歷史', '地方傳聞', '物品背景', '角色設定'。")
    key_elements: List[str] = Field(default_factory=list, description="與此 Lore 相關的關鍵詞或核心元素列表。")
    related_entities: List[str] = Field(default_factory=list, description="與此 Lore 相關的角色、地點或物品的名稱列表。")
    # [v2.1 核心修正] 新增 template_keys 欄位
    template_keys: Optional[List[str]] = Field(default=None, description="一個可選的關鍵詞列表。任何身份(alias)匹配此列表的角色，都將繼承本條LORE的content作為其行為準則。")

    @field_validator('aliases', 'key_elements', 'related_entities', 'template_keys', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        return _validate_string_to_list(value)
# src/schemas.py 的 WorldLore 類別結束

class EntityValidationResult(BaseModel):
    decision: Literal['CREATE', 'MERGE', 'IGNORE'] = Field(description="驗證後的最終決定。")
    reasoning: str = Field(description="做出此判斷的簡短、清晰的理由。")
    matched_key: Optional[str] = Field(default=None, description="如果判斷為'MERGE'，此欄位必須包含匹配到的實體的完整 key。")

    @model_validator(mode='after')
    def check_consistency(self) -> 'EntityValidationResult':
        if self.decision == 'MERGE' and not self.matched_key:
            raise ValueError("如果 decision 是 'MERGE'，則 matched_key 欄位是必需的。")
        return self

class SynthesisTask(BaseModel):
    name: str
    original_description: str
    new_information: str

class SynthesizedDescription(BaseModel):
    name: str = Field(description="與輸入任務完全相同的角色名稱。")
    description: str = Field(description="由LLM重寫並整合後的全新描述文本。")

class BatchSynthesisResult(BaseModel):
    synthesized_descriptions: List[SynthesizedDescription] = Field(description="一個包含所有被成功合成描述的角色的結果列表。")

class FactCheckResult(BaseModel):
    is_consistent: bool = Field(description="判斷提議的更新是否與對話上下文完全一致。")
    conflicting_info: Optional[str] = Field(default=None, description="如果不一致，說明衝突之處。")
    suggestion: Optional[Dict[str, Any]] = Field(default=None, description="一個修正後的 `updates` 字典。")

class BatchResolutionResult(BaseModel):
    original_name: str = Field(description="與輸入列表中完全相同的原始實體名稱。")
    decision: str = Field(description="您的最終判斷，通常是 'CREATE', 'NEW', 'MERGE', 或 'EXISTING' 之一。")
    reasoning: str = Field(description="您做出此判斷的簡短、清晰的理由。")
    matched_key: Optional[str] = Field(default=None, description="如果判斷為合併類型，此欄位必須包含匹配到的實體的完整 key。")
    standardized_name: str = Field(description="最終應使用的標準化名稱。")

    @model_validator(mode='after')
    def check_consistency(self) -> 'BatchResolutionResult':
        if self.decision.upper() in ['MERGE', 'EXISTING'] and not self.matched_key:
            raise ValueError("如果 decision 是 'MERGE' 或 'EXISTING'，則 matched_key 欄位是必需的。")
        return self

class BatchResolutionPlan(BaseModel):
    resolutions: List[BatchResolutionResult] = Field(description="一個包含對每一個待解析實體的判斷結果的列表。")

# 類別：工具調用 (v1.1 - 增加別名容錯)
# 更新紀錄:
# v1.1 (2025-09-28): [災難性BUG修復] 為 `tool_name` 欄位增加了 `validation_alias=AliasChoices('tool_name', 'tool_code')`。此修改旨在解決因LLM未能嚴格遵守Schema、錯誤地生成了`tool_code`鍵名而導致的致命ValidationError，通過使Pydantic模型能夠兼容兩種鍵名，極大地增強了事後分析流程的健壯性。
class ToolCall(BaseModel):
    tool_name: str = Field(..., description="要呼叫的工具的名稱。", validation_alias=AliasChoices('tool_name', 'tool_code'))
    parameters: Dict[str, Any] = Field(..., description="要傳遞給工具的參數字典。")

    @field_validator('parameters', mode='before')
    @classmethod
    def parse_parameters_from_string(cls, value):
        if isinstance(value, str):
            try:
                corrected_string = value.replace("'", '"')
                return json.loads(corrected_string)
            except json.JSONDecodeError:
                return value
        return value
# 類別：工具調用

class WorldGenesisResult(BaseModel):
    location_path: List[str] = Field(description="新生成的出生點的層級式路徑。")
    location_info: LocationInfo = Field(description="對該出生點的詳細描述。")
    initial_npcs: List[CharacterProfile] = Field(default_factory=list, description="伴隨出生點生成的初始NPC列表。")

class CanonParsingResult(BaseModel):
    npc_profiles: List[CharacterProfile] = Field(default_factory=list, description="從文本中解析出的所有 NPC 的完整個人檔案列表。")
    locations: List[LocationInfo] = Field(default_factory=list, description="從文本中解析出的所有地點的詳細資訊列表。")
    items: List[ItemInfo] = Field(default_factory=list, description="從文本中解析出的所有物品的詳細資訊列表。")
    creatures: List[CreatureInfo] = Field(default_factory=list, description="從文本中解析出的所有生物或物種的詳細資訊列表。")
    quests: List[Quest] = Field(default_factory=list, description="從文本中解析出的所有任務的詳細資訊列表。")
    world_lores: List[WorldLore] = Field(default_factory=list, description="從文本中解析出的所有世界傳說、歷史或背景故事的列表。")


# [v3.0 核心修正] 補全缺失的類定義
class LoreClassificationResult(BaseModel):
    """用於混合 NLP 流程，表示單個候選實體的分類結果。"""
    entity_name: str = Field(description="與輸入完全相同的候選實體名稱。")
    lore_category: Literal['npc_profile', 'location_info', 'item_info', 'creature_info', 'quest', 'world_lore', 'ignore'] = Field(description="對此實體的 LORE 類別判斷。如果是不重要或無法識別的實體，則判斷為 'ignore'。")
    reasoning: str = Field(description="做出此分類判斷的簡短理由。")

class BatchClassificationResult(BaseModel):
    """包裹批量分類結果的模型。"""
    classifications: List[LoreClassificationResult] = Field(description="一個包含對每一個候選實體的分類結果的列表。")


# [v1.4 新增] 單個實體解析模型，確保 ai_core.py 可以成功導入
class SingleResolutionResult(BaseModel):
    """單個實體名稱的解析結果。"""
    original_name: str = Field(description="LLM 在計畫中生成的原始實體名稱。")
    decision: Literal['NEW', 'EXISTING'] = Field(description="判斷結果：'NEW' 代表這是一個全新的實體，'EXISTING' 代表它指向一個已存在的實體。")
    standardized_name: Optional[str] = Field(None, description="如果判斷為'NEW'，AI 應為其生成一個更標準、更正式的名稱。如果判斷為'EXISTING'，此欄位可為空。")
    matched_key: Optional[str] = Field(None, description="如果判斷為'EXISTING'，此欄位必須包含匹配到的、已存在的實體的唯一主鍵 (lore_key)。")
    reasoning: str = Field(description="AI 做出此判斷的簡短理由。")

class SingleResolutionPlan(BaseModel):
    """單個實體名稱的完整解析計畫。"""
    resolution: SingleResolutionResult

class ToolCallPlan(BaseModel):
    plan: List[ToolCall] = Field(..., description="一個包含多個工具呼叫計畫的列表。")

class TurnPlan(BaseModel):
    thought: Optional[str] = Field(default=None, description="您作為世界導演的整體思考過程。")
    narration: Optional[str] = Field(default="", description="導演的場景設定旁白。")
    character_actions: List[Dict] = Field(default_factory=list, description="本回合所有核心角色的關鍵表演列表。")
    execution_rejection_reason: Optional[str] = Field(default=None, description="當指令因不合邏輯而無法執行時的解釋。")
    template_id: Optional[str] = Field(default=None, description="[系統專用] 模板ID。")

class UserInputAnalysis(BaseModel):
    input_type: Literal['dialogue_or_command', 'narration', 'continuation'] = Field(description="使用者輸入的類型。")
    summary_for_planner: str = Field(description="為規劃器提供的指令摘要。")
    narration_for_turn: str = Field(description="如果使用者提供了旁白，此處為完整內容。")

class SceneCastingResult(BaseModel):
    newly_created_npcs: List[CharacterProfile] = Field(default_factory=list, description="新創建的核心角色列表。")
    supporting_cast: List[CharacterProfile] = Field(default_factory=list, description="用於互動的臨時配角列表。")
    implied_location: Optional[LocationInfo] = Field(default=None, description="從上下文中推斷出的場景地點。")

class SceneAnalysisResult(BaseModel):
    viewing_mode: Literal['local', 'remote'] = Field(description="使用者當前的行動/觀察視角。")
    reasoning: str = Field(description="做出此判斷的理由。")
    target_location_path: Optional[List[str]] = Field(default=None, description="遠程觀察的目標地點路徑。")
    focus_entity: Optional[str] = Field(default=None, description="觀察的核心對象。")
    action_summary: str = Field(description="對使用者意圖的簡潔總結。")

    @model_validator(mode='after')
    def check_target_location_if_remote(self) -> 'SceneAnalysisResult':
        if self.viewing_mode == 'remote' and not self.target_location_path:
            raise ValueError("如果 viewing_mode 是 'remote'，則 target_location_path 是必需的。")
        return self

class ValidationResult(BaseModel):
    is_violating: bool = Field(description="如果文本違反了使用者主權原則，則為 true。")

class ExtractedEntities(BaseModel):
    names: List[str] = Field(description="從文本中提取出的所有專有名詞和關鍵實體名稱的列表。")

class ExpansionDecision(BaseModel):
    should_expand: bool = Field(description="如果當前對話適合進行世界構建，則為 true。")
    reasoning: str = Field(description="做出此決定的理由。")

class IntentClassificationResult(BaseModel):
    intent_type: Literal['sfw', 'nsfw_interactive', 'nsfw_descriptive'] = Field(description="對使用者輸入意圖的最終分類。")
    reasoning: str = Field(description="做出此分類的理由。")

class StyleAnalysisResult(BaseModel):
    dialogue_requirement: str = Field(description="對本回合對話的具體要求。")
    narration_level: str = Field(description="對旁白詳細程度的要求。")
    proactive_suggestion: Optional[str] = Field(default=None, description="用於推動劇情的行動建議。")

# src/schemas.py 的 NarrativeExtractionResult 類別 (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-11-22): [全新創建] 根據「智能敘事RAG注入」策略，創建此模型，用於驗證從世界聖經中提取出的純敘事文本。
class NarrativeExtractionResult(BaseModel):
    """用於包裹從世界聖經中提取出的純敘事文本的模型。"""
    narrative_text: str = Field(description="所有被提取出的劇情摘要、背景故事和歷史事件的組合文本。")
# src/schemas.py 的 NarrativeExtractionResult 類別 結束


# src/schemas.py 的 NarrativeDirective 類別 (v1.0 - 全新創建)
# 更新紀錄:
# v1.1 (2025-09-28): [架構升級] 根據「最終防線協議」對模型進行了重構。移除了孤立的`mandatory_action`，並將`scene_summary_for_generation`設為核心必需字段。這將AI導演的職責從提供「建議」升級為輸出一個融合了所有邏輯的、完整的「劇本大綱」，確保了下游主生成模型的指令單一性和執行可靠性。
# v1.0 (2025-09-28): [全新創建] 根據「AI導演」架構，創建此模型。
class NarrativeDirective(BaseModel):
    """用於包裹「AI導演」決策結果的模型。"""
    scene_summary_for_generation: str = Field(description="對使用者原始意圖和推斷出的強制動作的簡潔整合，作為給主生成模型的最終、最高優先級的創作指令。")
# src/schemas.py 的 NarrativeDirective 類別結束



# src/schemas.py 的 RagFactSheet 類別 (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-09-28): [全新創建] 根據「RAG事實清單」策略，創建此模型。它的作用是將RAG摘要任務從一個高風險的「創造性總結」轉變為一個低風險的「結構化數據提取」，通過強制LLM輸出事實列表而非敘事段落，從根本上解決RAG摘要流程被語義安全過濾器攔截的問題。
class RagFactSheet(BaseModel):
    """用於包裹從RAG檢索結果中提取出的結構化事實清單的模型。"""
    involved_characters: List[str] = Field(default_factory=list, description="在文本中涉及到的所有核心角色的名字列表。")
    key_locations: List[str] = Field(default_factory=list, description="事件發生的關鍵地點列表。")
    significant_objects: List[str] = Field(default_factory=list, description="在事件中扮演重要角色的物品列表。")
    core_events: List[str] = Field(default_factory=list, description="對核心事件的、極度中性且客觀的要點式描述列表。")



# src/schemas.py 的 PostGenerationAnalysisResult 類別 (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-11-22): [全新創建] 根據「生成後分析」架構，創建此模型。它用於包裹事後分析鏈返回的記憶摘要和LORE更新計畫，確保數據結構的穩定性。
class PostGenerationAnalysisResult(BaseModel):
    """用於包裹事後分析鏈返回的記憶摘要和LORE更新計畫的模型。"""
    memory_summary: Optional[str] = Field(default=None, description="對本回合對話的簡潔、無害化總結，用於長期記憶。")
    lore_updates: List[ToolCall] = Field(default_factory=list, description="一個包含多個用於創建或更新LORE的工具呼叫計畫的列表。")
# src/schemas.py 的 PostGenerationAnalysisResult 類別結束


# src/schemas.py 的 SceneLocationExtraction 類別 (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-09-28): [全新創建] 根據「場景範疇界定」架構，創建此模型。它的作用是驗證一個前置的、輕量級LLM調用的輸出，該調用專門用於從使用者指令中判斷並提取一個明確的“敘事意圖地點”。這使得系統能夠區分玩家的“客觀位置”和他们希望讲述的“故事場景”，從根源上解決因地點衝突導致的場景漂移問題。
class SceneLocationExtraction(BaseModel):
    """用於包裹从使用者指令中提取出的叙事意图地点的模型。"""
    has_explicit_location: bool = Field(description="如果使用者指令中包含一個明確的地點或场景描述，则为 true。")
    location_path: Optional[List[str]] = Field(default=None, description="如果 has_explicit_location 为 true，则此處為提取出的、層級化的地點路徑列表。")
# src/schemas.py 的 SceneLocationExtraction 類別結束










# --- 確保所有模型都已更新 ---
CharacterProfile.model_rebuild()
Quest.model_rebuild()
LocationInfo.model_rebuild()
ItemInfo.model_rebuild()
CreatureInfo.model_rebuild()
WorldLore.model_rebuild()
ToolCall.model_rebuild()
WorldGenesisResult.model_rebuild()
CanonParsingResult.model_rebuild()
BatchResolutionResult.model_rebuild()
BatchResolutionPlan.model_rebuild()
BatchRefinementResult.model_rebuild()
EntityValidationResult.model_rebuild()
SynthesisTask.model_rebuild()
SynthesizedDescription.model_rebuild()
BatchSynthesisResult.model_rebuild()
FactCheckResult.model_rebuild()
ExtractionResult.model_rebuild()
CharacterSkeleton.model_rebuild()
TurnPlan.model_rebuild()
UserInputAnalysis.model_rebuild()
SceneCastingResult.model_rebuild()
SceneAnalysisResult.model_rebuild()
ValidationResult.model_rebuild()
ExtractedEntities.model_rebuild()
ExpansionDecision.model_rebuild()
IntentClassificationResult.model_rebuild()
StyleAnalysisResult.model_rebuild()
SingleResolutionPlan.model_rebuild()
SingleResolutionResult.model_rebuild()
# [v2.0 新增] 確保新模型也被重建
RelationshipDetail.model_rebuild()
# [v3.0 新增] 確保新模型也被重建
LoreClassificationResult.model_rebuild()
BatchClassificationResult.model_rebuild()

# [v1.0 新增] 確保敘事提取模型也被重建
NarrativeExtractionResult.model_rebuild()

# [v1.0 新增] 確保事後分析模型也被重建
PostGenerationAnalysisResult.model_rebuild()







