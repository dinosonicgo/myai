# src/discord_bot.py 的中文註釋(v63.0 - 結構分離)
# 更新紀錄:
# v63.0 (2025-10-04): [災難性BUG修復-終極方案] 創建此簡化版 Bot 檔案。所有指令、UI 和事件監聽器已被遷移至 `cogs/core_cog.py`。此檔案現在只負責 Bot 的實例化和加載擴展，徹底解決了模組初始化悖論問題。
# v62.1 (2025-10-04): [災難性BUG修復] 修正了導入路徑。
# v62.0 (2025-10-02): [功能擴展] 新增了管理員調試指令。

import discord
from discord.ext import commands
import asyncio
import sys

from .config import settings
from .logger import logger

# --- Bot 類別定義 ---

class AILoverBot(commands.Bot):
    # 函式：初始化 AILoverBot
    def __init__(self, shutdown_event: asyncio.Event, git_lock: asyncio.Lock, is_ollama_available: bool):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True
        
        super().__init__(command_prefix='/', intents=intents, activity=discord.Game(name="與你共度時光"))
        
        self.shutdown_event = shutdown_event
        self.git_lock = git_lock
        self.is_ready_once = False
        self.is_ollama_available = is_ollama_available
    # 函式：初始化 AILoverBot 結束

    # 函式：Discord 機器人設置鉤子
    async def setup_hook(self):
        """
        此鉤子在 bot 登錄前執行，是加載擴展（Cogs）的最佳位置。
        """
        logger.info("正在加載核心 Cog (core_cog)...")
        try:
            # 我們告訴 bot 去加載 `src/cogs/core_cog.py` 這個擴展
            await self.load_extension("src.cogs.core_cog")
            logger.info("✅ 核心 Cog 加載成功。")
        except Exception as e:
            logger.error(f"🔥 核心 Cog 加載失敗: {e}", exc_info=True)
            # 在發生嚴重錯誤時，可以選擇關閉 bot
            await self.close()
            return
        
        # 同步指令的邏輯保持不變
        try:
            if settings.TEST_GUILD_ID:
                guild = discord.Object(id=int(settings.TEST_GUILD_ID))
                self.tree.copy_global_to(guild=guild)
                logger.info(f"正在將應用程式指令同步到測試伺服器 (ID: {settings.TEST_GUILD_ID})...")
                await self.tree.sync(guild=guild)
                logger.info(f"✅ 指令已同步到測試伺服器！")
            else:
                logger.info("正在全域同步應用程式指令...")
                await self.tree.sync()
                logger.info("✅ 指令已全域同步！")
        except Exception as e:
            logger.error(f"🔥 應用程式指令同步失敗: {e}", exc_info=True)
            
        logger.info("Discord Bot setup hook finished!")
    # 函式：Discord 機器人設置鉤子 結束

    # 函式：機器人準備就緒時的事件處理器
    async def on_ready(self):
        if not self.is_ready_once:
            self.is_ready_once = True
            logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
            if settings.ADMIN_USER_ID:
                try:
                    admin_user = self.get_user(int(settings.ADMIN_USER_ID)) or await self.fetch_user(int(settings.ADMIN_USER_ID))
                    await admin_user.send(f"✅ **系統啟動成功！(Bot v63.0)**")
                    logger.info(f"已成功發送啟動成功通知給管理員。")
                except Exception as e:
                    logger.error(f"發送啟動成功通知給管理員時發生未知錯誤: {e}", exc_info=True)
    # 函式：機器人準備就緒時的事件處理器 結束
