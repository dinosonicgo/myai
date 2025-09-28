# schemas.py v3.1 (完整版)
# 更新紀錄:
# v3.1 (2025-09-28): [完整性修復] 根據使用者要求，提供了包含所有近期新增模型（CharacterSkeleton, ExtractionResult, RagFactSheet, NarrativeDirective, SceneLocationExtraction）的完整文件內容。
# v3.0 (2025-09-27): [災難性BUG修復] 補全了缺失的 LoreClassificationResult 和 BatchClassificationResult 類定義，並將 WorldLore 的 'title' 統一為 'name' 以解決 ValidationError。
# v2.0 (2025-09-27): [重大架構升級] 新增了 RelationshipDetail 模型，並將 CharacterProfile.relationships 升級為結構化字典，同時增加了向下兼容的驗證器。

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

# --- 多階段混合解析管線所需模型 ---
class CharacterSkeleton(BaseModel):
    """用於混合 NLP 第一階段，表示一個角色的最基本骨架信息。"""
    name: str = Field(description="角色的名字。必須是文本中明確提到的、最常用的名字。")
    raw_description: str = Field(description="一段包含了所有與該角色相關的、從世界聖經原文中提取出的、未經處理的完整文本片段。")

class ExtractionResult(BaseModel):
    """包裹第一階段實體骨架提取結果的模型。"""
    characters: List[CharacterSkeleton] = Field(description="從文本中提取出的所有潛在角色實體的骨架列表。")

# --- 核心 LORE 結構 ---
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

    @field_validator('relationships', mode='before')
    @classmethod
    def _validate_and_normalize_relationships(cls, value: Any) -> Dict[str, Any]:
        if isinstance(value, str):
            value = _validate_string_to_dict(value)
        if not isinstance(value, dict):
            return {}
            
        normalized_dict = {}
        for k, v in value.items():
            if isinstance(v, str):
                normalized_dict[str(k)] = RelationshipDetail(roles=[v])
            elif isinstance(v, int):
                 normalized_dict[str(k)] = RelationshipDetail(type="好感度", roles=[str(v)])
            elif isinstance(v, dict):
                try:
                    normalized_dict[str(k)] = RelationshipDetail.model_validate(v)
                except Exception:
                    roles = [str(role) for role in v.get("roles", [])]
                    type_str = v.get("type", "社交關係")
                    normalized_dict[str(k)] = RelationshipDetail(type=type_str, roles=roles)
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

class WorldLore(BaseModel):
    name: str = Field(description="這條傳說、神話或歷史事件的標準化、唯一的官方標題。", validation_alias=AliasChoices('name', 'title'))
    aliases: List[str] = Field(default_factory=list, description="此傳說的其他已知稱呼或別名。")
    content: str = Field(default="", description="詳細的內容描述。")
    category: str = Field(default="未知", description="Lore 的分類，例如 '神話', '歷史', '地方傳聞', '物品背景', '角色設定'。")
    key_elements: List[str] = Field(default_factory=list, description="與此 Lore 相關的關鍵詞或核心元素列表。")
    related_entities: List[str] = Field(default_factory=list, description="與此 Lore 相關的角色、地點或物品的名稱列表。")
    template_keys: Optional[List[str]] = Field(default=None, description="一個可選的關鍵詞列表。任何身份(alias)匹配此列表的角色，都將繼承本條LORE的content作為其行為準則。")

    @field_validator('aliases', 'key_elements', 'related_entities', 'template_keys', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

class CanonParsingResult(BaseModel):
    npc_profiles: List[CharacterProfile] = Field(default_factory=list, description="從文本中解析出的所有 NPC 的完整個人檔案列表。")
    locations: List[LocationInfo] = Field(default_factory=list, description="從文本中解析出的所有地點的詳細資訊列表。")
    items: List[ItemInfo] = Field(default_factory=list, description="從文本中解析出的所有物品的詳細資訊列表。")
    creatures: List[CreatureInfo] = Field(default_factory=list, description="從文本中解析出的所有生物或物種的詳細資訊列表。")
    quests: List[Quest] = Field(default_factory=list, description="從文本中解析出的所有任務的詳細資訊列表。")
    world_lores: List[WorldLore] = Field(default_factory=list, description="從文本中解析出的所有世界傳說、歷史或背景故事的列表。")

# --- 輔助與流程控制模型 ---
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

class PostGenerationAnalysisResult(BaseModel):
    """用於包裹事後分析鏈返回的記憶摘要和LORE更新計畫的模型。"""
    memory_summary: Optional[str] = Field(default=None, description="對本回合對話的簡潔、無害化總結，用於長期記憶。")
    lore_updates: List[ToolCall] = Field(default_factory=list, description="一個包含多個用於創建或更新LORE的工具呼叫計畫的列表。")

class RagFactSheet(BaseModel):
    """用於包裹從RAG檢索結果中提取出的結構化事實清單的模型。"""
    involved_characters: List[str] = Field(default_factory=list, description="在文本中涉及到的所有核心角色的名字列表。")
    key_locations: List[str] = Field(default_factory=list, description="事件發生的關鍵地點列表。")
    significant_objects: List[str] = Field(default_factory=list, description="在事件中扮演重要角色的物品列表。")
    core_events: List[str] = Field(default_factory=list, description="對核心事件的、極度中性且客觀的要點式描述列表。")

class NarrativeDirective(BaseModel):
    """用於包裹「AI導演」決策結果的模型。"""
    scene_summary_for_generation: str = Field(description="對使用者原始意圖和推斷出的強制動作的簡潔整合，作為給主生成模型的最終、最高優先級的創作指令。")

class SceneLocationExtraction(BaseModel):
    """用於包裹从使用者指令中提取出的叙事意图地点的模型。"""
    has_explicit_location: bool = Field(description="如果使用者指令中包含一個明確的地點或场景描述，则为 true。")
    location_path: Optional[List[str]] = Field(default=None, description="如果 has_explicit_location 为 true，则此處為提取出的、層級化的地點路徑列表。")

# --- 舊版或特定用途模型 ---
class LoreClassificationResult(BaseModel):
    entity_name: str = Field(description="與輸入完全相同的候選實體名稱。")
    lore_category: Literal['npc_profile', 'location_info', 'item_info', 'creature_info', 'quest', 'world_lore', 'ignore'] = Field(description="對此實體的 LORE 類別判斷。")
    reasoning: str = Field(description="做出此分類判斷的簡短理由。")

class BatchClassificationResult(BaseModel):
    classifications: List[LoreClassificationResult] = Field(description="一個包含對每一個候選實體的分類結果的列表。")

# --- 確保所有模型都已更新 ---
CharacterSkeleton.model_rebuild()
ExtractionResult.model_rebuild()
RelationshipDetail.model_rebuild()
CharacterProfile.model_rebuild()
BatchRefinementResult.model_rebuild()
Quest.model_rebuild()
LocationInfo.model_rebuild()
ItemInfo.model_rebuild()
CreatureInfo.model_rebuild()
WorldLore.model_rebuild()
CanonParsingResult.model_rebuild()
ToolCall.model_rebuild()
PostGenerationAnalysisResult.model_rebuild()
RagFactSheet.model_rebuild()
NarrativeDirective.model_rebuild()
SceneLocationExtraction.model_rebuild()
LoreClassificationResult.model_rebuild()
BatchClassificationResult.model_rebuild()
