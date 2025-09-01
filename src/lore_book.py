# src/lore_book.py 的中文註釋(v3.0 - 數據訪問層重構)
# 更新紀錄:
# v3.0 (2025-08-16):
# 1. [重大架構重構] 移除了本地的 `Lore` 模型定義，改為從 `database.py` 導入。
# 2. [健壯性] 此修改使本檔案成為一個純粹的數據訪問層（Data Access Layer），解決了循環導入問題，使資料庫架構更加清晰和穩定。
# v2.0 (2025-08-12):
# 1. [功能擴展] 在 `Lore` 模型中新增了 `source` 欄位。

import time
from typing import List, Dict, Any, Optional, Callable
from sqlalchemy import select, delete
from sqlalchemy.future import select as future_select
from pydantic import Field

# [v3.0 修改] 從 database 導入模型和會話
from .database import AsyncSessionLocal, Lore

# 異步函式，新增或更新一條 Lore 記錄
async def add_or_update_lore(user_id: str, category: str, key: str, content: Dict[str, Any], source: Optional[str] = None) -> Lore:
    """
    新增或更新一条 Lore 记录。
    如果具有相同 user_id, category 和 key 的记录已存在，则更新其 content 和 timestamp。
    否则，创建一条新记录。
    可以选择性地传入 source 来标记数据来源。
    """
    async with AsyncSessionLocal() as session:
        # 查詢現有記錄
        stmt = select(Lore).where(
            Lore.user_id == user_id,
            Lore.category == category,
            Lore.key == key
        )
        result = await session.execute(stmt)
        existing_lore = result.scalars().first()
        
        current_time = time.time()
        
        if existing_lore:
            # 更新現有記錄
            existing_lore.content = content
            existing_lore.timestamp = current_time
            # 只有在明確傳入 source 時才更新 source，避免意外覆蓋
            if source is not None:
                existing_lore.source = source
            lore_entry = existing_lore
        else:
            # 創建新記錄
            lore_entry = Lore(
                user_id=user_id,
                category=category,
                key=key,
                content=content,
                timestamp=current_time,
                source=source
            )
            session.add(lore_entry)
            
        await session.commit()
        await session.refresh(lore_entry)
        return lore_entry
# 異步函式，新增或更新一條 Lore 記錄

# 異步函式，根據 category 和 key 查詢單條 Lore 記錄
async def get_lore(user_id: str, category: str, key: str) -> Optional[Lore]:
    """
    根據 user_id, category 和 key 查詢單條 Lore 記錄。
    """
    async with AsyncSessionLocal() as session:
        stmt = select(Lore).where(
            Lore.user_id == user_id,
            Lore.category == category,
            Lore.key == key
        )
        result = await session.execute(stmt)
        return result.scalars().first()
# 異步函式，根據 category 和 key 查詢單條 Lore 記錄

# 異步函式，根據 category 查詢多條 Lore 記錄，並可選地進行過濾
async def get_lores_by_category_and_filter(
    user_id: str, 
    category: str, 
    filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
) -> List[Lore]:
    """
    根據 user_id 和 category 查詢多條 Lore 記錄。
    如果提供了 filter_func，則會對查詢結果的 content 字段進行進一步的篩選。
    """
    async with AsyncSessionLocal() as session:
        stmt = select(Lore).where(
            Lore.user_id == user_id,
            Lore.category == category
        )
        result = await session.execute(stmt)
        all_lores = result.scalars().all()
        
        if filter_func:
            # 在 Python 層面對 JSON content 進行過濾
            return [lore for lore in all_lores if filter_func(lore.content)]
        else:
            return list(all_lores)
# 異步函式，根據 category 查詢多條 Lore 記錄，並可選地進行過濾

# [v2.0 新增] 根據來源刪除 Lore 記錄
async def delete_lores_by_source(user_id: str, source: str) -> int:
    """
    根據 user_id 和 source 刪除所有匹配的 Lore 記錄。
    返回被刪除的記錄數量。
    """
    async with AsyncSessionLocal() as session:
        stmt = delete(Lore).where(
            Lore.user_id == user_id,
            Lore.source == source
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount
# 根據來源刪除 Lore 記錄

# [v2.0 新增] 根據來源獲取所有 Lore 記錄
async def get_all_lores_by_source(user_id: str, source: str) -> List[Lore]:
    """
    根據 user_id 和 source 獲取所有匹配的 Lore 記錄。
    """
    async with AsyncSessionLocal() as session:
        stmt = select(Lore).where(
            Lore.user_id == user_id,
            Lore.source == source
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())
# 根據來源獲取所有 Lore 記錄

# [v2.0 新增] 根據主鍵列表刪除 Lore 記錄
async def delete_lores_by_keys(user_id: str, keys: List[str]) -> int:
    """
    根據 user_id 和一個主鍵列表 (key) 刪除匹配的 Lore 記錄。
    返回被刪除的記錄數量。
    """
    if not keys:
        return 0
    async with AsyncSessionLocal() as session:
        stmt = delete(Lore).where(
            Lore.user_id == user_id,
            Lore.key.in_(keys)
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount
# 根據主鍵列表刪除 Lore 記錄