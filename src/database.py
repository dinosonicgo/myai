# src/database.py 的中文註釋(v5.1 - LORE繼承支持)
# 更新紀錄:
# v5.1 (2025-09-24): [架構擴展] 在 Lore 模型中新增了 template_keys 欄位。此欄位用於實現LORE的繼承和模板化，允許一個LORE條目（如一個具體NPC）繼承另一個概念LORE（如一個角色職業）的屬性，是解決角色設定不一致問題的關鍵數據庫層支持。
# v5.0 (2025-11-22): [重大架構升級] 新增了 SceneHistoryData 模型。此修改旨在將之前純記憶體的短期場景對話歷史進行資料庫持久化，從根本上解決因程式重啟或實例重建導致的上下文丟失和劇情斷裂問題。
# v4.0 (2025-11-15): [架構升級] 根據【持久化淨化快取】策略，增加了 sanitized_content 欄位。

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Integer, Float, JSON, TEXT
import time

from src.config import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# 類別：用戶核心數據模型
class UserData(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True)
    username = Column(String)
    ai_name = Column(String)
    ai_settings = Column(String, nullable=True)
    affinity = Column(Integer, default=0)
    game_state = Column(JSON)
    one_instruction = Column(String, nullable=True)
    world_settings = Column(String, nullable=True)
    user_profile = Column(JSON, nullable=True)
    ai_profile = Column(JSON, nullable=True)
    response_style_prompt = Column(String, nullable=True)
# 用戶核心數據模型 類別結束

# 類別：長期記憶數據模型
class MemoryData(Base):
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    content = Column(String)
    timestamp = Column(Float)
    importance = Column(Integer)
    sanitized_content = Column(String, nullable=True)
# 長期記憶數據模型 類別結束

# 類別：LORE (世界設定) 數據模型
class Lore(Base):
    __tablename__ = "lore_book"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    category = Column(String, index=True, nullable=False)
    key = Column(String, index=True, nullable=False)
    content = Column(JSON, nullable=False)
    timestamp = Column(Float, nullable=False)
    source = Column(String, index=True, nullable=True)
    template_keys = Column(JSON, nullable=True) # [v5.1 核心新增]
# LORE (世界設定) 數據模型 類別結束

# 類別：短期場景歷史數據模型 (v5.0 新增)
class SceneHistoryData(Base):
    __tablename__ = "scene_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    scene_key = Column(String, index=True, nullable=False)
    message_json = Column(JSON, nullable=False)
    timestamp = Column(Float, default=time.time, nullable=False)
# 短期場景歷史數據模型 類別結束

# 函式：初始化資料庫
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
# 初始化資料庫 函式結束
        
# 函式：獲取資料庫會話
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
# 獲取資料庫會話 函式結束
