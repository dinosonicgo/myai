# schemas.py v4.0 (严格流水线 Schema)
# 更新纪录:
# v4.0 (2025-09-30): [重大架构重构] 根据资讯遗失问题，为专职解析流水线创建了一整套全新的、层次分明的 Pydantic 模型。新增了 `IdentifiedEntity`、`AliasItem`、`AppearanceItem`、`CoreInfoItem` 等模型，以及包裹它们的批量结果模型。此修改为 LLM 提供了极度明确的输出结构契约，旨在从根本上解决因输出格式模糊导致的灾难性数据合并失败问题。
# v3.0 (2025-09-27): [灾难性BUG修复] 补全了缺失的 LoreClassificationResult 和 BatchClassificationResult 类定义。
# v2.3 (2025-09-30): [重大架构升级] 新增了 AppearanceDetails 模型，用于结构化存储角色的详细外观特征。

import json
import re
from typing import Optional, Dict, List, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, AliasChoices

# --- 基础验证器 ---
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

# --- 流水线核心模型 ---

# [v4.0 新增] 阶段一：实体识别
class IdentifiedEntity(BaseModel):
    """代表一个从原文中识别出的、待处理的 LORE 实体"""
    name: str = Field(description="实体的专有名称")
    category: Literal['npc_profile', 'location_info', 'item_info', 'creature_info', 'quest', 'world_lore'] = Field(description="实体的 LORE 类别")

class BatchIdentifiedEntitiesResult(BaseModel):
    """包裹批量实体识别结果的模型"""
    entities: List[IdentifiedEntity]

# [v4.0 新增] 阶段三：专职流水线输出模型
class AliasItem(BaseModel):
    """包裹单个角色别名/身份列表的模型"""
    character_name: str
    aliases: List[str]

class BatchAliasesResult(BaseModel):
    """包裹批量别名/身份提取结果的模型"""
    results: List[AliasItem]

class AppearanceItem(BaseModel):
    """包裹单个角色结构化外观细节的模型"""
    character_name: str
    appearance_details: "AppearanceDetails"

class BatchAppearanceResult(BaseModel):
    """包裹批量外观细节提取结果的模型"""
    results: List[AppearanceItem]
    
# schemas.py 的 CoreInfoItem 模型 (v4.1 - 容錯 description)
# 更新紀錄:
# v4.1 (2025-09-30): [災難性BUG修復] 根據 ValidationError，將 description 欄位的類型從 str 修改為 Optional[str]。此修改允許 LLM 在找不到描述時返回 null，從而避免驗證失敗，提高了 LORE 解析管線的健壯性。
# v4.0 (2025-09-30): [重大架構重構] 根據資訊遺失問題，為專職解析流水線創建了此模型。
class CoreInfoItem(BaseModel):
    """包裹单个角色核心信息（描述、技能、关系）的模型"""
    character_name: str
    description: Optional[str] = "" # [v4.1 核心修正] 允許 description 為 None
    skills: List[str] = Field(default_factory=list)
    relationships: Dict[str, "RelationshipDetail"] = Field(default_factory=dict)
# schemas.py 的 CoreInfoItem 模型

class BatchCoreInfoResult(BaseModel):
    """包裹批量核心信息提取结果的模型"""
    results: List[CoreInfoItem]


class LocationItem(BaseModel):
    """包裹单个地点信息的模型"""
    name: str
    location_info: "LocationInfo"

class BatchLocationsResult(BaseModel):
    """包裹批量地点信息提取结果的模型"""
    results: List[LocationItem]

class ItemItem(BaseModel):
    """包裹单个物品信息的模型"""
    name: str
    item_info: "ItemInfo"

class BatchItemsResult(BaseModel):
    """包裹批量物品信息提取结果的模型"""
    results: List[ItemItem]

class CreatureItem(BaseModel):
    """包裹单个生物信息的模型"""
    name: str
    creature_info: "CreatureInfo"

class BatchCreaturesResult(BaseModel):
    """包裹批量生物信息提取结果的模型"""
    results: List[CreatureItem]

class QuestItem(BaseModel):
    """包裹单个任务信息的模型"""
    name: str
    quest_info: "Quest"

class BatchQuestsResult(BaseModel):
    """包裹批量任务信息提取结果的模型"""
    results: List[QuestItem]

class WorldLoreItem(BaseModel):
    """包裹单个世界传说信息的模型"""
    name: str
    world_lore_info: "WorldLore"

class BatchWorldLoresResult(BaseModel):
    """包裹批量世界传说信息提取结果的模型"""
    results: List[WorldLoreItem]
# --- LORE 基础模型 ---

class AppearanceDetails(BaseModel):
    """角色的结构化外观细节"""
    height: Optional[str] = Field(default=None, description="身高")
    body_type: Optional[str] = Field(default=None, description="体型或身材描述")
    hair_style: Optional[str] = Field(default=None, description="发型与发色")
    eye_color: Optional[str] = Field(default=None, description="瞳色与眼神特征")
    skin_tone: Optional[str] = Field(default=None, description="肤色与皮肤特征")
    distinctive_features: List[str] = Field(default_factory=list, description="显著特征，如疤痕、纹身、胎记等")
    age_appearance: Optional[str] = Field(default=None, description="外观看上去的年龄")
    clothing_style: Optional[str] = Field(default=None, description="通常的服装风格")
    overall_impression: Optional[str] = Field(default=None, description="给人的整体印象或气质")

    @field_validator('distinctive_features', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

class RelationshipDetail(BaseModel):
    """储存一个角色对另一个角色的详细关系资讯"""
    type: str = Field(default="社交关系", description="关系的类型，例如 '家庭', '主从', '敌对', '恋爱', '社交关系'。")
    roles: List[str] = Field(default_factory=list, description="对方在此关系中扮演的角色或称谓列表，支持多重身份。例如 ['女儿', '学生']。")

# schemas.py 的 CharacterProfile 模型 (v2.4 - 全面 Optional 容錯)
# 更新紀錄:
# v2.4 (2025-10-01): [災難性BUG修復] 根據持續的 ValidationError，將模型中所有可能被 LLM 省略的字串欄位（如 gender, age, race, appearance 等）的類型從 `str` 系統性地修改為 `Optional[str]`。此修改使模型能夠容忍 LLM 返回 `null` 值，從根本上解決了因數據缺失導致的驗證失敗問題。
# v2.3 (2025-09-30): [重大架構升級] 新增了 AppearanceDetails 模型。
class CharacterProfile(BaseModel):
    name: str = Field(description="角色的标准化、唯一的官方名字。")
    aliases: List[str] = Field(default_factory=list, description="此角色的所有其他已知称呼、身份标签、头衔或绰号。")
    alternative_names: List[str] = Field(default_factory=list, description="一个由AI预先生成的、用于在主名称冲突时备用的名称列表。")
    gender: Optional[str] = Field(default="未设定", description="角色的性别。")
    age: Optional[str] = Field(default="未知", description="角色的年龄或年龄段。")
    race: Optional[str] = Field(default="未知", description="角色的种族。")
    appearance: Optional[str] = Field(default="", description="角色的外貌特征的总体描述。")
    appearance_details: AppearanceDetails = Field(default_factory=AppearanceDetails, description="角色的结构化具体外貌细节。")
    likes: List[str] = Field(default_factory=list, description="角色喜欢的事物列表。")
    dislikes: List[str] = Field(default_factory=list, description="角色不喜欢的食物列表。")
    equipment: List[str] = Field(default_factory=list, description="角色【当前穿戴或持有】的装备列表。")
    skills: List[str] = Field(default_factory=list, description="角色掌握的技能列表。")
    description: Optional[str] = Field(default="", description="角色的性格、背景故事、行为模式等综合简介。")
    location: Optional[str] = Field(default="", description="角色当前所在的城市或主要区域。")
    location_path: List[str] = Field(default_factory=list, description="角色当前所在的层级式地点路径。")
    affinity: int = Field(default=0, description="此角色对使用者的好感度。")
    relationships: Dict[str, RelationshipDetail] = Field(default_factory=dict, description="记录此角色与其他角色的结构化关系。例如：{'莉莉丝': {'type': '家庭', 'roles': ['女儿']}}")
    status: Optional[str] = Field(default="健康", description="角色的当前健康或状态。")
    current_action: Optional[str] = Field(default="站着", description="角色当前正在进行的、持续性的动作或所处的姿态。")

    @field_validator('aliases', 'likes', 'dislikes', 'equipment', 'skills', 'location_path', 'alternative_names', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        if isinstance(value, str) and (' > ' in value or '/' in value):
            return [part.strip() for part in re.split(r'\s*>\s*|/', value)]
        return _validate_string_to_list(value)

    @field_validator('appearance_details', mode='before')
    @classmethod
    def _validate_string_to_dict_fields(cls, value: Any) -> Any:
        if isinstance(value, AppearanceDetails):
            return value
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
                    type_str = v.get("type", "社交关系")
                    normalized_dict[str(k)] = RelationshipDetail(type=type_str, roles=roles)
            else:
                normalized_dict[str(k)] = RelationshipDetail(roles=[str(v)])
        return normalized_dict

    @field_validator('affinity', mode='before')
    @classmethod
    def _coerce_affinity_to_int(cls, value: Any) -> Any:
        if isinstance(value, float):
            return int(value)
        return value
# schemas.py 的 CharacterProfile 模型

class Quest(BaseModel):
    name: str = Field(description="任务的标准化、唯一的官方名称。")
    aliases: List[str] = Field(default_factory=list, description="此任务的其他已知称呼或别名。")
    description: str = Field(default="", description="任务的详细描述和目标。")
    status: str = Field(default="active", description="任务的当前状态，例如 'active', 'completed', 'failed'。")
    quest_giver: Optional[str] = Field(default=None, description="此任务的发布者（NPC名字）。")
    suggested_level: Optional[int] = Field(default=None, description="建议执行此任务的角色等级。")
    rewards: Dict[str, Any] = Field(default_factory=dict, description="完成任务的奖励，例如 {{'金钱': 100, '物品': ['治疗药水']}}。")

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
    name: str = Field(description="地点的标准化、唯一的官方名称。")
    aliases: List[str] = Field(default_factory=list, description="此地点的其他已知称呼或别名。")
    description: str = Field(default="", description="对该地点的详细描述，包括环境、氛围、建筑风格等。")
    notable_features: List[str] = Field(default_factory=list, description="该地点的显著特征或地标列表。")
    known_npcs: List[str] = Field(default_factory=list, description="已知居住或出现在此地点的 NPC 名字列表。")

    @field_validator('aliases', 'notable_features', 'known_npcs', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

class ItemInfo(BaseModel):
    name: str = Field(description="道具的标准化、唯一的官方名称。")
    aliases: List[str] = Field(default_factory=list, description="此物品的其他已知称呼或别名。")
    description: str = Field(default="", description="对该道具的详细描述，包括其历史、材质、背景故事等。")
    item_type: str = Field(default="未知", description="道具的类型，例如 '消耗品', '武器', '关键物品', '盔甲', '服装'。")
    effect: str = Field(default="无", description="道具的使用效果描述。")
    rarity: str = Field(default="普通", description="道具的稀有度，例如 '普通', '稀有', '史诗', '传说'。")
    visual_description: Optional[str] = Field(default="", description="对道具外观的详细、生动的描寫。")
    origin: Optional[str] = Field(default="", description="关于该道具来源或制造者的简短传说。")

    @field_validator('aliases', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

class CreatureInfo(BaseModel):
    name: str = Field(description="生物/魔物的标准化、唯一的官方种类名称（例如 '水晶鸡'）。")
    aliases: List[str] = Field(default_factory=list, description="此生物的其他已知称呼或别名。")
    description: str = Field(default="", description="对该生物/魔物的详细描述，包括外貌、习性、生态地位等。")
    abilities: List[str] = Field(default_factory=list, description="该生物/魔物的特殊能力列表。")
    habitat: List[str] = Field(default_factory=list, description="该生物/魔物的主要栖息地列表。")

    @field_validator('aliases', 'abilities', 'habitat', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

class WorldLore(BaseModel):
    name: str = Field(description="这条传说、神话或历史事件的标准化、唯一的官方标题。", validation_alias=AliasChoices('name', 'title'))
    aliases: List[str] = Field(default_factory=list, description="此传说的其他已知称呼或别名。")
    content: str = Field(default="", description="详细的内容描述。")
    category: str = Field(default="未知", description="Lore 的分类，例如 '神话', '历史', '地方传闻', '物品背景', '角色设定'。")
    key_elements: List[str] = Field(default_factory=list, description="与此 Lore 相关的关键词或核心元素列表。")
    related_entities: List[str] = Field(default_factory=list, description="与此 Lore 相关的角色、地点或物品的名称列表。")
    template_keys: Optional[List[str]] = Field(default=None, description="一个可选的关键词列表。任何身份(alias)匹配此列表的角色，都将继承本条LORE的content作为其行为准则。")

    @field_validator('aliases', 'key_elements', 'related_entities', 'template_keys', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

class ToolCall(BaseModel):
    tool_name: str = Field(..., description="要呼叫的工具的名称。", validation_alias=AliasChoices('tool_name', 'tool_code'))
    parameters: Dict[str, Any] = Field(..., description="要传递给工具的参数字典。")

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

# --- 旧的、用于兼容的各种模型 ---

class CharacterSkeleton(BaseModel):
    name: str
    description: str

class ExtractionResult(BaseModel):
    characters: List[CharacterSkeleton]

class BatchRefinementResult(BaseModel):
    refined_profiles: List[CharacterProfile]

class EntityValidationResult(BaseModel):
    decision: Literal['CREATE', 'MERGE', 'IGNORE']
    reasoning: str
    matched_key: Optional[str] = None

    @model_validator(mode='after')
    def check_consistency(self) -> 'EntityValidationResult':
        if self.decision == 'MERGE' and not self.matched_key:
            raise ValueError("如果 decision 是 'MERGE'，则 matched_key 字段是必需的。")
        return self

class SynthesisTask(BaseModel):
    name: str
    original_description: str
    new_information: str

class SynthesizedDescription(BaseModel):
    name: str
    description: str

class BatchSynthesisResult(BaseModel):
    synthesized_descriptions: List[SynthesizedDescription]

class FactCheckResult(BaseModel):
    is_consistent: bool
    conflicting_info: Optional[str] = None
    suggestion: Optional[Dict[str, Any]] = None

class BatchResolutionResult(BaseModel):
    original_name: str
    decision: str
    reasoning: str
    matched_key: Optional[str] = None
    standardized_name: str

    @model_validator(mode='after')
    def check_consistency(self) -> 'BatchResolutionResult':
        if self.decision.upper() in ['MERGE', 'EXISTING'] and not self.matched_key:
            raise ValueError("如果 decision 是 'MERGE' 或 'EXISTING'，则 matched_key 栏位是必需的。")
        return self

class BatchResolutionPlan(BaseModel):
    resolutions: List[BatchResolutionResult]

class WorldGenesisResult(BaseModel):
    location_path: List[str]
    location_info: LocationInfo
    initial_npcs: List[CharacterProfile] = Field(default_factory=list)

class CanonParsingResult(BaseModel):
    npc_profiles: List[CharacterProfile] = Field(default_factory=list)
    locations: List[LocationInfo] = Field(default_factory=list)
    items: List[ItemInfo] = Field(default_factory=list)
    creatures: List[CreatureInfo] = Field(default_factory=list)
    quests: List[Quest] = Field(default_factory=list)
    world_lores: List[WorldLore] = Field(default_factory=list)

class LoreClassificationResult(BaseModel):
    entity_name: str
    lore_category: Literal['npc_profile', 'location_info', 'item_info', 'creature_info', 'quest', 'world_lore', 'ignore']
    reasoning: str

class BatchClassificationResult(BaseModel):
    classifications: List[LoreClassificationResult]

class SingleResolutionResult(BaseModel):
    original_name: str
    decision: Literal['NEW', 'EXISTING']
    standardized_name: Optional[str] = None
    matched_key: Optional[str] = None
    reasoning: str

class SingleResolutionPlan(BaseModel):
    resolution: SingleResolutionResult

class ToolCallPlan(BaseModel):
    plan: List[ToolCall]

class TurnPlan(BaseModel):
    thought: Optional[str] = None
    narration: Optional[str] = ""
    character_actions: List[Dict] = Field(default_factory=list)
    execution_rejection_reason: Optional[str] = None
    template_id: Optional[str] = None

class UserInputAnalysis(BaseModel):
    input_type: Literal['dialogue_or_command', 'narration', 'continuation']
    summary_for_planner: str
    narration_for_turn: str

class SceneCastingResult(BaseModel):
    newly_created_npcs: List[CharacterProfile] = Field(default_factory=list)
    supporting_cast: List[CharacterProfile] = Field(default_factory=list)
    implied_location: Optional[LocationInfo] = None

class SceneAnalysisResult(BaseModel):
    viewing_mode: Literal['local', 'remote']
    reasoning: str
    target_location_path: Optional[List[str]] = None
    focus_entity: Optional[str] = None
    action_summary: str

    @model_validator(mode='after')
    def check_target_location_if_remote(self) -> 'SceneAnalysisResult':
        if self.viewing_mode == 'remote' and not self.target_location_path:
            raise ValueError("如果 viewing_mode 是 'remote'，则 target_location_path 是必需的。")
        return self

class ValidationResult(BaseModel):
    is_violating: bool

class ExtractedEntities(BaseModel):
    names: List[str]

class ExpansionDecision(BaseModel):
    should_expand: bool
    reasoning: str

class IntentClassificationResult(BaseModel):
    intent_type: Literal['sfw', 'nsfw_interactive', 'nsfw_descriptive']
    reasoning: str

class StyleAnalysisResult(BaseModel):
    dialogue_requirement: str
    narration_level: str
    proactive_suggestion: Optional[str] = None

class NarrativeExtractionResult(BaseModel):
    narrative_text: str

class NarrativeDirective(BaseModel):
    scene_summary_for_generation: str

class RagFactSheet(BaseModel):
    involved_characters: List[str] = Field(default_factory=list)
    key_locations: List[str] = Field(default_factory=list)
    significant_objects: List[str] = Field(default_factory=list)
    core_events: List[str] = Field(default_factory=list)

class PostGenerationAnalysisResult(BaseModel):
    memory_summary: Optional[str] = None
    lore_updates: List[ToolCall] = Field(default_factory=list)

class SceneLocationExtraction(BaseModel):
    has_explicit_location: bool
    location_path: Optional[List[str]] = None

# --- 手动重建模型依赖 ---
AppearanceDetails.model_rebuild()
RelationshipDetail.model_rebuild()
CharacterProfile.model_rebuild()
AliasItem.model_rebuild()
AppearanceItem.model_rebuild()
CoreInfoItem.model_rebuild()
BatchAliasesResult.model_rebuild()
BatchAppearanceResult.model_rebuild()
BatchCoreInfoResult.model_rebuild()
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
LoreClassificationResult.model_rebuild()
BatchClassificationResult.model_rebuild()
NarrativeExtractionResult.model_rebuild()
PostGenerationAnalysisResult.model_rebuild()
IdentifiedEntity.model_rebuild()
BatchIdentifiedEntitiesResult.model_rebuild()

LocationItem.model_rebuild()
BatchLocationsResult.model_rebuild()
ItemItem.model_rebuild()
BatchItemsResult.model_rebuild()
CreatureItem.model_rebuild()
BatchCreaturesResult.model_rebuild()
QuestItem.model_rebuild()
BatchQuestsResult.model_rebuild()
WorldLoreItem.model_rebuild()
BatchWorldLoresResult.model_rebuild()



