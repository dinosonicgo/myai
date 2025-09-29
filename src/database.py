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

# database.py 的 init_db 函式 (v6.2 - 修正异步检查错误)
# 更新紀錄:
# v6.2 (2025-11-25): [灾难性BUG修复] 彻底重构了轻量级迁移的实现方式，遵循 SQLAlchemy 的异步编程规范。将所有需要同步连接的 `inspect` 操作封装在一个独立的同步函式中，并通过 `conn.run_sync` 来安全地调用，从而解决了 `Inspection on an AsyncConnection is not supported` 的致命错误。
# v6.1 (2025-11-25): [健壮性强化] 增加了更详细的日誌输出。
# v6.0 (2025-11-24): [灾难性BUG修复] 引入了轻量级资料库迁移机制。
async def init_db(db_ready_event: asyncio.Event):
    """
    初始化资料库。
    首先确保所有表格都已创建，然后执行轻量级的迁移检查，
    确保现有表格的结构与最新的模型定义保持一致。
    """
    print("--- 正在初始化资料库与执行结构验证 ---")

    # [v6.2 核心修正] 定义一个同步函式来执行所有需要同步连接的检查操作
    def _inspect_and_migrate_sync(connection):
        """
        在同步上下文中执行资料库结构检查和迁移。
        """
        from sqlalchemy import inspect, text

        print("   [DB Init] 步骤 2/3: 检查 'users' 表结构是否需要升级...")
        try:
            # 在同步连接上创建 Inspector
            inspector = inspect(connection)
            
            # 获取 'users' 表的所有栏位资讯
            columns = inspector.get_columns("users")
            
            # 将栏位资讯转换为一个简单的名称集合，以便快速查找
            column_names = {c['name'] for c in columns}

            # 检查 'context_snapshot_json' 栏位是否存在
            if 'context_snapshot_json' not in column_names:
                print("   ⚠️ [资料库迁移] 检测到 'users' 表缺少 'context_snapshot_json' 栏位，正在自动新增...")
                # 在同步事务中，可以直接使用 connection 执行命令
                connection.execute(text('ALTER TABLE users ADD COLUMN context_snapshot_json JSON'))
                print("   ✅ [资料库迁移] 'context_snapshot_json' 栏位已成功新增。")
            else:
                print("   [DB Init] 步骤 2/3: 'users' 表结构已是最新，无需升级。")

        except Exception as e:
            # 如果在迁移过程中发生任何错误，记录下来但不要让整个程式崩溃
            print(f"   🔥 [资料库迁移] 在尝试升级 'users' 表结构时发生严重错误: {e}")

    async with engine.begin() as conn:
        # 步骤 1: 确保所有在 Base 中定义的表格都存在
        print("   [DB Init] 步骤 1/3: 确保所有资料表已创建...")
        await conn.run_sync(Base.metadata.create_all)
        print("   [DB Init] 步骤 1/3: 资料表创建检查完成。")
        
        # [v6.2 核心修正] 使用 conn.run_sync 来安全地执行同步的检查函式
        await conn.run_sync(_inspect_and_migrate_sync)

    # 步骤 3: 发出资料库就绪信号
    db_ready_event.set()
    print("✅ 数据库初始化与结构验证完成，并已发出就绪信号。")
# 初始化资料库 函式结束



        
# 函式：獲取資料庫會話
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
# 獲取資料庫會話 函式結束





