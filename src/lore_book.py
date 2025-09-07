# src/lore_book.py 的中文註釋(v3.1 - 新增全局查詢)
# 更新紀錄:
# v3.1 (2025-09-03): [健壯性] 新增了 `get_all_lores_for_user` 函式，将获取用户所有 LORE 的数据库查询逻辑从 ai_core.py 中分离并封装于此。此修改遵循了代码复用和职责分离的原则，使 LORE 系统的底层交互更加模块化和可维护。
# v3.0 (2025-08-16): [重大架構重構] 移除了本地的 `Lore` 模型定義，改為從 `database.py` 導入。
# v2.0 (2025-08-12): [功能擴展] 在 `Lore` 模型中新增了 `source` 欄位。

import time
from typing import List, Dict, Any, Optional, Callable
from sqlalchemy import select, delete, or_
from sqlalchemy.future import select as future_select
from pydantic import Field, BaseModel

# [v3.0 修改] 從 database 導入模型和會話
from .database import AsyncSessionLocal, Lore

# 異步函式，新增或更新一條 Lore 記錄
async def add_or_update_lore(user_id: str, category: str, key: str, content: Dict[str, Any], source: Optional[str] = None, merge: bool = False) -> Lore:
    """
    新增或更新一条 Lore 记录。
    如果具有相同 user_id, category 和 key 的记录已存在，则更新其 content 和 timestamp。
    否则，创建一条新记录。
    可以选择性地传入 source 来标记数据来源。
    如果 merge 为 True，则会将传入的 content 字典与现有的 content 字典合并，而不是完全替换。
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
            if merge and isinstance(existing_lore.content, dict):
                # 合并字典
                updated_content = existing_lore.content.copy()
                updated_content.update(content)
                existing_lore.content = updated_content
            else:
                # 完全替换
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

# 函式：獲取指定使用者的所有 LORE (v1.0 - 全新创建)
# 更新纪录:
# v1.0 (2025-09-03): [健壮性] 新增此函数，将获取用户所有 LORE 的数据库查询逻辑从 ai_core.py 中分离并封装于此。此修改遵循了代码复用和职责分离的原则，使 LORE 系统的底层交互更加模块化和可维护。
async def get_all_lores_for_user(user_id: str) -> List[Lore]:
    """获取指定用户的所有LORE条目。"""
    async with AsyncSessionLocal() as session:
        stmt = select(Lore).where(Lore.user_id == user_id)
        result = await session.execute(stmt)
        return result.scalars().all()
# 函式：獲取指定使用者的所有 LORE (v1.0 - 全新创建)
