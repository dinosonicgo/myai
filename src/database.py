# src/database.py 的中文註釋(v5.3 - 導入修正)
# 更新紀錄:
# v5.3 (2025-09-24): [災難性BUG修復] 在文件頂部增加了 `import asyncio`，以解決因在 `init_db` 函式簽名中使用 `asyncio.Event` 類型提示而導致的 `NameError`。
# v5.2 (2025-09-24): [健壯性強化] 增加了對 asyncio.Event 的支持，以解決啟動時的競爭條件問題。
# v5.1 (2025-09-24): [架構擴展] 在 Lore 模型中新增了 template_keys 欄位。

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Integer, Float, JSON, TEXT
import time
import asyncio

from src.config import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# database.py 的 UserData 類別 (v5.4 - 新增快照持久化欄位)
# 更新紀錄:
# v5.4 (2025-11-24): [健壯性強化] 新增了 context_snapshot_json 欄位。此修改旨在將上一輪對話生成的、用於恢復上下文的快照持久化到資料庫，從根本上解決了因程式重啟導致記憶體中快照丟失，從而使「繼續」等指令失效的問題。
# v5.3 (2025-09-24): [災難性BUG修復] 在文件頂部增加了 `import asyncio`。
# v5.2 (2025-09-24): [健壯性強化] 增加了對 asyncio.Event 的支持。
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
    # [v5.4 核心修正] 新增上下文快照欄位
    context_snapshot_json = Column(JSON, nullable=True)
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

# v5.1 (2025-09-24): [架構擴展] 在 Lore 模型中新增了 template_keys 欄位。這是實現「LORE繼承與規則注入系統」的資料庫層基礎，用於標識哪些LORE條目可以作為其他角色的行為模板。
# v5.0 (2025-09-24): [災難性BUG修復] 在文件頂部增加了 `import asyncio`。
# v4.2 (2025-09-24): [健壯性強化] 增加了對 asyncio.Event 的支持。
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
    # [v5.1 核心修正] 新增 template_keys 欄位
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

# database.py 的 init_db 函式 (v6.1 - 增加日誌清晰度)
# 更新紀錄:
# v6.1 (2025-11-25): [健壯性強化] 增加了更詳細的日誌輸出，以便在應用程式啟動時，能明確地觀察到資料庫遷移檢查是否被觸發和執行，從而簡化問題診斷流程。
# v6.0 (2025-11-24): [災難性BUG修復] 引入了輕量級的資料庫遷移機制以解決 `OperationalError: no such column` 的問題。
# v5.3 (2025-09-24): [災難性BUG修復] 在文件頂部增加了 `import asyncio`。
async def init_db(db_ready_event: asyncio.Event):
    """
    初始化資料庫。
    首先確保所有表格都已創建，然後執行輕量級的遷移檢查，
    確保現有表格的結構與最新的模型定義保持一致。
    """
    print("--- 正在初始化資料庫與執行結構驗證 ---")
    async with engine.begin() as conn:
        # 步驟 1: 確保所有在 Base 中定義的表格都存在
        print("   [DB Init] 步驟 1/3: 確保所有資料表已創建...")
        await conn.run_sync(Base.metadata.create_all)
        print("   [DB Init] 步驟 1/3: 資料表創建檢查完成。")
        
        # [v6.1 核心修正] 步驟 2: 執行輕量級的資料庫遷移
        print("   [DB Init] 步驟 2/3: 檢查 'users' 表結構是否需要升級...")
        try:
            from sqlalchemy import inspect, text

            # 創建一個 Inspector 來檢查資料庫的實際結構
            inspector = inspect(conn)
            
            # 異步獲取 'users' 表的所有欄位資訊
            columns = await conn.run_sync(inspector.get_columns, "users")
            
            # 將欄位資訊轉換為一個簡單的名稱集合，以便快速查找
            column_names = {c['name'] for c in columns}

            # 檢查 'context_snapshot_json' 欄位是否存在
            if 'context_snapshot_json' not in column_names:
                print("   ⚠️ [資料庫遷移] 檢測到 'users' 表缺少 'context_snapshot_json' 欄位，正在自動新增...")
                # 如果不存在，則執行 ALTER TABLE 命令來新增它
                await conn.execute(text('ALTER TABLE users ADD COLUMN context_snapshot_json JSON'))
                print("   ✅ [資料庫遷移] 'context_snapshot_json' 欄位已成功新增。")
            else:
                print("   [DB Init] 步驟 2/3: 'users' 表結構已是最新，無需升級。")

        except Exception as e:
            # 如果在遷移過程中發生任何錯誤，記錄下來但不要讓整個程式崩潰
            print(f"   🔥 [資料庫遷移] 在嘗試升級 'users' 表結構時發生嚴重錯誤: {e}")

    # 步驟 3: 發出資料庫就緒信號
    db_ready_event.set()
    print("✅ 數據庫初始化與結構驗證完成，並已發出就緒信號。")
# 初始化資料庫 函式結束



        
# 函式：獲取資料庫會話
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
# 獲取資料庫會話 函式結束




