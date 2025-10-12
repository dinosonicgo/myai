# src/database.py 的中文註釋(v6.0 - 強制啟用WAL模式)
# 更新紀錄:
# v6.0 (2025-10-13): [災難性BUG修復] 引入了SQLAlchemy事件監聽器，在每次數據庫連接時強制執行`PRAGMA journal_mode=WAL;`。此修改將SQLite的日誌模式切換為預寫式日誌（WAL），極大地提高了併發讀寫性能，旨在從根本上解決因aiosqlite驅動在高負載啟動時發生I/O競爭而導致的永久性阻塞（卡死）問題。
# v5.4 (2025-10-04): [災難性BUG修復] 將檔案頂部的 `from src.config import settings` 修正為相對導入。
# v5.3 (2025-09-24): [災難性BUG修復] 在文件頂部增加了 `import asyncio`。

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Integer, Float, JSON, TEXT
# [v6.0 核心修正] 導入 SQLAlchemy 事件模塊
from sqlalchemy import event
from sqlalchemy.engine import Engine
import time
import asyncio

from .config import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=False)

# [v6.0 核心修正] 設定事件監聽器以啟用 WAL 模式
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """在每個連接上啟用 WAL 模式以提高併發性"""
    # 僅對 SQLite 連接執行此操作
    if "sqlite" in engine.url.drivername:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
        finally:
            cursor.close()

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

# 類別：長期記憶數據模型 (v6.0 - 雙重持久化)
class MemoryData(Base):
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    content = Column(TEXT)
    timestamp = Column(Float)
    importance = Column(Integer)
    sanitized_content = Column(TEXT, nullable=True)
# 長期記憶數據模型 類別結束

# 類別：LORE (世界設定) 數據模型 (v6.0 - 混合式LORE)
class Lore(Base):
    __tablename__ = "lore_book"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    category = Column(String, index=True, nullable=False)
    key = Column(String, index=True, nullable=False)
    structured_content = Column(JSON, nullable=True)
    narrative_content = Column(TEXT, nullable=True)
    timestamp = Column(Float, nullable=False)
    source = Column(String, index=True, nullable=True)
    template_keys = Column(JSON, nullable=True)
# LORE (世界設定) 數據模型 類別結束

# 類別：短期場景歷史數據模型
class SceneHistoryData(Base):
    __tablename__ = "scene_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    scene_key = Column(String, index=True, nullable=False)
    message_json = Column(JSON, nullable=False)
    timestamp = Column(Float, default=time.time, nullable=False)
# 短期場景歷史數據模型 類別結束

# 函式：初始化資料庫
async def init_db(db_ready_event: asyncio.Event):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    db_ready_event.set()
    print("✅ 數據庫初始化完成，並已發出就緒信號。")
# 初始化資料庫 函式結束
        
# 函式：獲取資料庫會話
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
# 獲取資料庫會話 函式結束
