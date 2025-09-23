import json
import re
from typing import Optional, Dict, List, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

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



# 函式：基础 LORE 數據模型 - CharacterProfile (v2.2 - 防禦性默認值)
# 更新紀錄:
# v2.2 (2025-09-23): [災難性BUG修復] 為多個容易被LLM省略的字段（如 appearance, description, skills 等）提供了空的默認值（例如 `default=""` 或 `default_factory=list`）。這使得 Pydantic 模型更具防禦性，即使LLM返回的JSON中缺少這些鍵，模型也能成功驗證並使用安全的空值填充，從根本上解決了大量的 ValidationError。
# v2.1 (2025-09-23): [災難性BUG修復] 為 `relationships` 欄位的 description 新增了轉義。
class CharacterProfile(BaseModel):
    name: str = Field(description="角色的標準化、唯一的官方名字。")
    aliases: List[str] = Field(default_factory=list, description="此角色的其他已知稱呼或別名。")
    alternative_names: List[str] = Field(default_factory=list, description="一个由AI预先生成的、用于在主名称冲突时备用的名称列表。")
    gender: Optional[str] = Field(default="未設定", description="角色的性別。")
    age: Optional[str] = Field(default="未知", description="角色的年齡或年齡段。")
    race: Optional[str] = Field(default="未知", description="角色的種族。")
    appearance: str = Field(default="", description="角色的外貌特徵的總體描述。")
    appearance_details: Dict[str, Any] = Field(default_factory=dict, description="角色的具體外貌細節，值可以是字串或列表。")
    likes: List[str] = Field(default_factory=list, description="角色喜歡的事物列表。")
    dislikes: List[str] = Field(default_factory=list, description="角色不喜歡的事物列表。")
    equipment: List[str] = Field(default_factory=list, description="角色【當前穿戴或持有】的裝備列表。")
    skills: List[str] = Field(default_factory=list, description="角色掌握的技能列表。")
    description: str = Field(default="", description="角色的性格、背景故事、行為模式等綜合簡介。")
    location: Optional[str] = Field(default=None, description="角色當前所在的城市或主要區域。")
    location_path: List[str] = Field(default_factory=list, description="角色當前所在的層級式地點路徑。")
    affinity: int = Field(default=0, description="此角色對使用者的好感度。")
    relationships: Dict[str, Any] = Field(default_factory=dict, description="記錄此角色與其他角色的關係。例如：{{'莉莉絲': '女兒', '卡爾': '丈夫'}}")
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
    def _validate_and_normalize_relationships(cls, value: Any) -> Dict[str, str]:
        if isinstance(value, str):
            value = _validate_string_to_dict(value)
        if not isinstance(value, dict):
            return {}
        normalized_dict = {}
        for k, v in value.items():
            if isinstance(v, int):
                normalized_dict[str(k)] = f"關係值: {v}"
            elif isinstance(v, str):
                normalized_dict[str(k)] = v
            else:
                normalized_dict[str(k)] = str(v)
        return normalized_dict
# 函式：基础 LORE 數據模型 - CharacterProfile (v2.2 - 防禦性默認值)

# [v1.0 新增] 用於批量LORE精煉的包裹模型
class BatchRefinementResult(BaseModel):
    refined_profiles: List[CharacterProfile] = Field(description="一個包含所有被成功精煉後的角色檔案的列表。")









class Quest(BaseModel):
    name: str = Field(description="任務的標準化、唯一的官方名稱。")
    aliases: List[str] = Field(default_factory=list, description="此任務的其他已知稱呼或別名。")
    description: str = Field(default="", description="任務的詳細描述和目標。")
    status: str = Field(default="active", description="任務的當前狀態，例如 'active', 'completed', 'failed'。")
    quest_giver: Optional[str] = Field(default=None, description="此任務的發布者（NPC名字）。")
    suggested_level: Optional[int] = Field(default=None, description="建議執行此任務的角色等級。")
    # [v1.0 核心修正] 轉義了 description 字符串中的大括號
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
    title: str = Field(description="這條傳說、神話或歷史事件的標準化、唯一的官方標題。")
    aliases: List[str] = Field(default_factory=list, description="此傳說的其他已知稱呼或別名。")
    content: str = Field(default="", description="詳細的內容描述。")
    category: str = Field(default="未知", description="Lore 的分類，例如 '神話', '歷史', '地方傳聞', '物品背景'。")
    key_elements: List[str] = Field(default_factory=list, description="與此 Lore 相關的關鍵詞或核心元素列表。")
    related_entities: List[str] = Field(default_factory=list, description="與此 Lore 相關的角色、地點或物品的名稱列表。")

    @field_validator('aliases', 'key_elements', 'related_entities', mode='before')
    @classmethod
    def _validate_string_to_list_fields(cls, value: Any) -> Any:
        return _validate_string_to_list(value)

# --- AI 思考/行動相關模型 ---
class ActionIntent(BaseModel):
    action_type: Literal['physical', 'verbal', 'magical', 'observation', 'other'] = Field(description="將使用者指令分類為：'physical'(物理動作), 'verbal'(對話), 'magical'(魔法), 'observation'(觀察), 或 'other'(其他)。")
    primary_target: Optional[str] = Field(default=None, description="動作的主要目標是誰或什麼？（例如 NPC 的名字）")
    action_summary_for_status: str = Field(description="用一句話總結這個動作，以便將其記錄為角色的 `current_action` 狀態。例如：'正在與碧進行口交'、'坐下'、'正在攻擊哥布林'。")

class ToolCall(BaseModel):
    tool_name: str = Field(..., description="要呼叫的工具的名稱。")
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

class CharacterAction(BaseModel):
    character_name: str = Field(description="執行此行動的角色的【確切】名字。")
    reasoning: str = Field(description="【必需】解釋該角色【為什麼】要採取這個行動。此理由必須與其性格、好感度、當前情境和目標緊密相關。")
    action_description: Optional[str] = Field(default=None, description="對該角色將要執行的【具體物理動作】的清晰、簡潔的描述。如果行動主要是對話，此欄位可為空。")
    dialogue: Optional[str] = Field(default=None, description="如果該角色在行動中或行動後會說話，請在此處提供確切的對話內容。")
    tool_call: Optional[ToolCall] = Field(default=None, description="如果此行動需要呼叫一個工具來改變世界狀態（如移動、使用物品），請在此處定義工具呼叫。") 
    template_id: Optional[str] = Field(default=None, description="[系統專用] 用於標識此動作來源於哪個預設模板。")

    @model_validator(mode='after')
    def check_action_or_dialogue_exists(self) -> 'CharacterAction':
        if not self.action_description and not self.dialogue:
            raise ValueError("一個 CharacterAction 必須至少包含 action_description 或 dialogue 其中之一。")
        return self

class TurnPlan(BaseModel):
    """一回合行動的完整結構化計畫。"""
    thought: Optional[str] = Field(default=None, description="您作為世界導演的整體思考過程。首先分析情境，然後為每個活躍的 AI/NPC 角色生成行動動機，最終制定出本回合的完整計畫。")
    narration: Optional[str] = Field(default="", description="【導演的場景設定 (Director's Scene Setting)】一個綜合性的、用於搭建舞台的旁白。它應描寫場景的整體氛圍、光影、聲音、以及任何與核心角色無關的背景活動，為接下來的核心表演奠定基調。")
    character_actions: List[CharacterAction] = Field(default_factory=list, description="一個包含本回合所有【核心角色】的【關鍵表演 (Key Performances)】的列表。")
    execution_rejection_reason: Optional[str] = Field(default=None, description="當且僅當指令因不合 lógica而無法執行時，此欄位包含以角色口吻給出的解釋。")
    template_id: Optional[str] = Field(default=None, description="[系統專用] 用於標識此計畫是否來源於某個預設模板。")

    @model_validator(mode='before')
    @classmethod
    def repair_missing_character_names(cls, data: Any) -> Any:
        if isinstance(data, dict) and 'character_actions' in data and isinstance(data['character_actions'], list):
            last_valid_name = None
            for action in data['character_actions']:
                if isinstance(action, dict):
                    if 'character_name' in action and action['character_name']:
                        last_valid_name = action['character_name']
                    elif 'character_name' not in action and last_valid_name:
                        action['character_name'] = last_valid_name
        return data

    @model_validator(mode='after')
    def check_plan_logic(self) -> 'TurnPlan':
        """[v15.0 自我修復模式] 驗證並修復計畫的邏輯一致性。"""
        has_actions = bool(self.character_actions)
        has_rejection = bool(self.execution_rejection_reason and self.execution_rejection_reason.strip())

        if has_actions and has_rejection:
            object.__setattr__(self, 'execution_rejection_reason', None)
            has_rejection = False
            
        has_thought_or_actions = bool(self.thought) or has_actions or (self.narration and self.narration.strip())
        if not has_thought_or_actions and not has_rejection:
            raise ValueError("一個 TurnPlan 必須至少包含 'thought'、'narration'、'character_actions' 或 'execution_rejection_reason' 中的一項。")
            
        return self

class ToolCallPlan(BaseModel):
    plan: List[ToolCall] = Field(..., description="一個包含多個工具呼叫計畫的列表。")

# 類別：世界創世結果
# 更新紀錄:
# v1.1 (2025-09-23): [架構調整] 根據“按需生成”原則，將 initial_npcs 欄位設為可選，因為創世階段現在只專注於生成地點。
class WorldGenesisResult(BaseModel):
    location_path: List[str] = Field(description="新生成的出生點的層級式路徑。例如：['艾瑟利亞王國', '首都晨風城', '城南的寧靜小巷']。")
    location_info: LocationInfo = Field(description="對該出生點的詳細描述，符合 LocationInfo 模型。")
    initial_npcs: List[CharacterProfile] = Field(default_factory=list, description="伴隨出生點生成的一到兩位初始NPC的完整角色檔案列表。")
# 類別：世界創世結果
class CanonParsingResult(BaseModel):
    npc_profiles: List[CharacterProfile] = Field(default_factory=list, description="從文本中解析出的所有 NPC 的完整個人檔案列表。")
    locations: List[LocationInfo] = Field(default_factory=list, description="從文本中解析出的所有地點的詳細資訊列表。")
    items: List[ItemInfo] = Field(default_factory=list, description="從文本中解析出的所有物品的詳細資訊列表。")
    creatures: List[CreatureInfo] = Field(default_factory=list, description="從文本中解析出的所有生物或物種的詳細資訊列表。")
    quests: List[Quest] = Field(default_factory=list, description="從文本中解析出的所有任務的詳細資訊列表。")
    world_lores: List[WorldLore] = Field(default_factory=list, description="從文本中解析出的所有世界傳說、歷史或背景故事的列表。")

class BatchResolutionResult(BaseModel):
    original_name: str = Field(description="與輸入列表中完全相同的原始實體名稱。")
    decision: Literal['EXISTING', 'NEW'] = Field(description="您的最終判斷：'EXISTING'表示此名稱指向一個已存在的實體，'NEW'表示這是一個全新的實體。")
    reasoning: str = Field(description="您做出此判斷的簡短、清晰的理由。")
    matched_key: Optional[str] = Field(default=None, description="如果判斷為'EXISTING'，此欄位【必須】包含來自現有實體列表中的、與之匹配的那個實體的【完整、未經修改的 `key`】。")
    standardized_name: Optional[str] = Field(default=None, description="如果判斷為'NEW'，請提供一個對新實體名稱進行清理和標準化後的版本。如果判斷為'EXISTING'，則返回匹配到的實體的主要名稱。")

    @model_validator(mode='after')
    def check_consistency_and_autofill(self) -> 'BatchResolutionResult':
        if self.decision == 'NEW' and not self.standardized_name:
            self.standardized_name = self.original_name
        if self.decision == 'EXISTING' and not self.matched_key:
            raise ValueError("如果 decision 是 'EXISTING'，則 matched_key 欄位是必需的。")
        
        if self.decision == 'EXISTING' and not self.standardized_name:
            if self.matched_key:
                self.standardized_name = self.matched_key.split(' > ')[-1]
            else:
                self.standardized_name = self.original_name
        
        if not self.standardized_name:
            self.standardized_name = self.original_name

        return self

class BatchResolutionPlan(BaseModel):
    resolutions: List[BatchResolutionResult] = Field(description="一個包含對每一個待解析實體的判斷結果的列表。")

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

class UserInputAnalysis(BaseModel):
    """[第一層分析] 用於結構化地表示對使用者輸入的初步意圖分析結果。"""
    input_type: Literal['dialogue_or_command', 'narration', 'continuation'] = Field(description="判斷使用者輸入的類型：'dialogue_or_command'（對話或指令）、'narration'（場景描述）、或 'continuation'（要求接續上文）。")
    summary_for_planner: str = Field(description="為後續的規劃器（Planner）提供一句簡潔、清晰的指令摘要。")
    narration_for_turn: str = Field(description="如果 `input_type` 是 'narration' 且由使用者主動提供，此欄位包含完整的原始旁白。否則為空字符串。")

class SceneCastingResult(BaseModel):
    """用於結構化地表示場景中預生成的新 NPC 的結果，包括核心角色和互動配角。"""
    newly_created_npcs: List[CharacterProfile] = Field(
        description="一個新創建的、需要被賦予身份的核心角色列表。",
        default_factory=list
    )
    supporting_cast: List[CharacterProfile] = Field(
        description="一個為核心角色創造的、用於互動的臨時配角列表（例如顧客、同伴等）。",
        default_factory=list
    )
    implied_location: Optional[LocationInfo] = Field(
        default=None,
        description="如果能從上下文中推斷出一個合理的、符合世界觀的場景地點，則在此處提供該地點的詳細信息。"
    )

class SceneAnalysisResult(BaseModel):
    """[第二層分析] 用於結構化地表示對使用者意圖和場景視角的分析結果，解決'遠程觀察'問題。"""
    viewing_mode: Literal['local', 'remote'] = Field(description="判斷使用者當前的行動/觀察視角。'local' 表示在當前場景行動；'remote' 表示正在觀察一個遠程場景。")
    reasoning: str = Field(description="做出此判斷的簡潔理由。")
    target_location_path: Optional[List[str]] = Field(default=None, description="如果 viewing_mode 為 'remote'，此欄位必須包含使用者意圖觀察的目標地點的層級路徑。")
    focus_entity: Optional[str] = Field(default=None, description="如果指令是觀察某個地點裡的特定事物或人（例如‘市場裡的魚販’），此欄位應包含那個核心觀察對象的名稱，例如 '魚販'。")
    action_summary: str = Field(description="對使用者意圖的簡潔總結，例如：'使用者想要看看王城市場現在是什麼樣子' 或 '使用者向 AI 角色搭話'。")

    @model_validator(mode='after')
    def check_target_location_if_remote(self) -> 'SceneAnalysisResult':
        if self.viewing_mode == 'remote' and not self.target_location_path:
            raise ValueError("如果 viewing_mode 是 'remote'，則 target_location_path 是必需的。")
        return self

class CharacterQuantificationResult(BaseModel):
    """用於結構化地表示從使用者輸入中量化出的角色描述列表。"""
    character_descriptions: List[str] = Field(description="一個包含所有需要被創建的角色的具體描述的列表。例如：['男性獸人戰士', '男性獸人戰士', '女性精靈法師']。")

class RawSceneAnalysis(BaseModel):
    """一個沒有複雜內部驗證的 Pydantic 模型，專門用於安全地接收來自 LLM 的、可能存在邏輯矛盾的初步場景分析結果。"""
    viewing_mode: Literal['local', 'remote'] = Field(description="對視角的初步判斷。")
    reasoning: str = Field(description="做出判斷的理由。")
    target_location_path: Optional[List[str]] = Field(default=None, description="初步提取的地點路徑。")
    focus_entity: Optional[str] = Field(default=None, description="初步提取的核心實體。")
    action_summary: str = Field(description="對使用者意圖的摘要。")

class ValidationResult(BaseModel):
    is_violating: bool = Field(description="如果文本違反了使用者主權原則，則為 true，否則為 false。")

class ExtractedEntities(BaseModel):
    names: List[str] = Field(description="從文本中提取出的所有專有名詞和關鍵實體名稱的列表。")

class ExpansionDecision(BaseModel):
    """用于结构化地表示关于是否应在本回合进行世界LORE扩展的决定。"""
    should_expand: bool = Field(description="如果当前对话轮次适合进行世界构建和LORE扩展，则为 true；如果对话是简单的、重复的或与已知实体的互动，则为 false。")
    reasoning: str = Field(description="做出此决定的简短理由。")

class IntentClassificationResult(BaseModel):
    """用於結構化地表示對使用者輸入意圖的語意分類結果。"""
    intent_type: Literal['sfw', 'nsfw_interactive', 'nsfw_descriptive'] = Field(description="""對使用者意圖的最終分類：
- 'sfw': 安全的、適合所有觀眾的內容。
- 'nsfw_interactive': 包含露骨或敏感內容的、使用者與 AI/NPC 的直接互動請求。
- 'nsfw_descriptive': 包含露骨或敏感內容的、要求對遠程或第三方場景進行描述的請求。""")
    reasoning: str = Field(description="做出此分類的簡短理由。")

class StyleAnalysisResult(BaseModel):
    """用於結構化地表示對使用者自訂風格的分析結果，以便為規劃器提供具體指令。"""
    dialogue_requirement: str = Field(description="根據風格指令，對本回合對話的具體要求。例如：'AI角色必須說話' 或 '無需對話'。")
    narration_level: str = Field(description="對旁白詳細程度的要求。例如：'低', '中等', '高'。")
    proactive_suggestion: Optional[str] = Field(default=None, description="根據風格和情境，給出一個可選的、用於推動劇情的行動建議。")

CharacterAction.model_rebuild()




