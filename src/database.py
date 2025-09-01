# database.py 的中文註釋(v3.0 - 模型定義集中化)
# 更新紀錄:
# v3.0 (2025-08-16):
# 1. [重大架構重構] 將 `Lore` 資料庫模型的定義從 `lore_book.py` 移動到此檔案中。
# 2. [健壯性] 此修改將所有 SQLAlchemy 的模型定義集中到一個基礎檔案中，使其不依賴任何其他專案模組，從根本上解決了潛在的循環導入（Circular Import）問題。
# v2.3 (2025-08-09):
# 1. [功能擴展] 在 `UserData` 表中新增了 `response_style_prompt` 欄位。

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Integer, Float, JSON, TEXT

from src.config import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class UserData(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True)
    # 舊欄位，保留用於資料遷移
    username = Column(String)
    ai_name = Column(String)
    ai_settings = Column(String, nullable=True)
    
    # 核心資料
    affinity = Column(Integer, default=0)
    game_state = Column(JSON)
    one_instruction = Column(String, nullable=True)
    world_settings = Column(String, nullable=True)

    # [v2.2 新增] 新的主角檔案儲存欄位
    user_profile = Column(JSON, nullable=True)
    ai_profile = Column(JSON, nullable=True)
    
    # [v2.3 新增] 自訂回覆風格
    response_style_prompt = Column(String, nullable=True)
    
class MemoryData(Base):
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    content = Column(String)
    timestamp = Column(Float)
    importance = Column(Integer)

# [v3.0 新增] Lore (衍生設定) 資料庫模型
class Lore(Base):
    __tablename__ = "lore_book"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    category = Column(String, index=True, nullable=False)
    key = Column(String, index=True, nullable=False)
    content = Column(JSON, nullable=False)
    timestamp = Column(Float, nullable=False)
    source = Column(String, index=True, nullable=True)

# 初始化資料庫
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
# 初始化資料庫
        
# 獲取資料庫會話
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
# 獲取資料庫會話