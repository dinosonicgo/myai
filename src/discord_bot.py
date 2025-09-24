# src/discord_bot.py 的中文註釋(v57.0 - 完整檔案整合)
# 更新紀錄:
# v57.0 (2025-11-17): [完整性修復] 根據使用者要求，提供包含所有近期修正（重新生成、指令同步、結構校正）的完整檔案，並為所有函式添加了標準化中文註釋。
# v56.0 (2025-11-17): [災難性BUG修復] 提供了結構絕對正確的類別定義，以修復因縮排錯誤導致的指令註冊失敗問題。
# v55.0 (2025-11-16): [功能整合] 整合了「重新生成」功能。

import discord
from discord import app_commands, Embed
from discord.ext import commands, tasks
import asyncio
import json
import shutil
from pathlib import Path
from sqlalchemy import select, delete, or_
import math 
import re
from typing import Optional, Literal, List, Dict, Any, Tuple
from collections import defaultdict
import os
import sys
import subprocess
import gc
import subprocess
import datetime
from pathlib import Path

from .logger import logger
from .ai_core import AILover
from . import lore_book
from .lore_book import Lore
from .database import AsyncSessionLocal, UserData, MemoryData, init_db, SceneHistoryData
from .schemas import CharacterProfile, LocationInfo, WorldGenesisResult
from .models import UserProfile, GameState
from src.config import settings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

PROJ_DIR = Path(__file__).resolve().parent.parent

# 函式：檢查使用者是否為管理員
async def is_admin(interaction: discord.Interaction) -> bool:
    if not settings.ADMIN_USER_ID: return False
    return str(interaction.user.id) == settings.ADMIN_USER_ID
# 函式：檢查使用者是否為管理員

LORE_CATEGORIES = [
    app_commands.Choice(name="👤 NPC 檔案 (npc_profile)", value="npc_profile"),
    app_commands.Choice(name="📍 地點資訊 (location_info)", value="location_info"),
    app_commands.Choice(name="📦 物品資訊 (item_info)", value="item_info"),
    app_commands.Choice(name="🐾 生物/物種 (creature_info)", value="creature_info"),
    app_commands.Choice(name="📜 任務 (quest)", value="quest"),
    app_commands.Choice(name="🌍 世界傳說 (world_lore)", value="world_lore"),
]

# 函式：使用者自動完成
async def user_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    choices = []
    async with AsyncSessionLocal() as session:
        stmt = select(UserData).where(or_(UserData.username.ilike(f"%{current}%"), UserData.user_id.ilike(f"%{current}%"))).limit(25)
        result = await session.execute(stmt)
        users = result.scalars().all()
        for user in users:
            choices.append(app_commands.Choice(name=f"{user.username} ({user.user_id})", value=user.user_id))
    return choices
# 函式：使用者自動完成

# 函式：Lore Key 自動完成
async def lore_key_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    target_user_id = str(interaction.namespace.target_user)
    category = interaction.namespace.category
    if not target_user_id or not category:
        return [app_commands.Choice(name="請先選擇使用者和類別...", value="")]
    
    choices = []
    async with AsyncSessionLocal() as session:
        stmt = select(Lore).where(
            Lore.user_id == target_user_id, 
            Lore.category == category, 
            Lore.key.ilike(f"%{current}%")
        ).limit(25)
        result = await session.execute(stmt)
        lores = result.scalars().all()

        for lore in lores:
            if category == 'npc_profile':
                content = lore.content
                name = content.get('name', '未知名稱')
                description = content.get('description', '未知職業')
                profession_part = re.split(r'[，。]', description)[0]
                profession = (profession_part[:15] + '…') if len(profession_part) > 15 else profession_part
                location_path = content.get('location_path', [])
                location = location_path[-1] if location_path else content.get('location', '未知地點')
                display_name = f"{name} ({profession}) @ {location}"
                choices.append(app_commands.Choice(name=display_name[:100], value=lore.key))
            else:
                key = lore.key
                display_name = key.split(' > ')[-1]
                if category in ['location_info'] or display_name == key:
                    display_name = key
                choices.append(app_commands.Choice(name=display_name, value=key))
                
    return choices
# 函式：Lore Key 自動完成

# --- 持久化視圖與 Modals ---

# 類別：/start 指令的初始設定視圖
class StartSetupView(discord.ui.View):
    # 函式：初始化 StartSetupView
    def __init__(self, *, cog: "BotCog"):
        super().__init__(timeout=None)
        self.cog = cog
    # 函式：初始化 StartSetupView

    # 函式：處理「開始設定」按鈕點擊事件
    @discord.ui.button(label="🚀 開始設定", style=discord.ButtonStyle.success, custom_id="persistent_start_setup_button")
    async def start_setup_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) Persistent 'StartSetupView' button clicked.")
        world_modal = WorldSettingsModal(self.cog, current_world="這是一個魔法與科技交織的幻想世界。", is_setup_flow=True, original_interaction_message_id=interaction.message.id)
        await interaction.response.send_modal(world_modal)
    # 函式：處理「開始設定」按鈕點擊事件
# 類別：/start 指令的初始設定視圖

# 類別：繼續到使用者角色設定的視圖
class ContinueToUserSetupView(discord.ui.View):
    # 函式：初始化 ContinueToUserSetupView
    def __init__(self, *, cog: "BotCog"):
        super().__init__(timeout=None)
        self.cog = cog
    # 函式：初始化 ContinueToUserSetupView

    # 函式：處理「下一步：設定您的角色」按鈕點擊事件
    @discord.ui.button(label="下一步：設定您的角色", style=discord.ButtonStyle.primary, custom_id="persistent_continue_to_user_setup")
    async def continue_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) Persistent 'ContinueToUserSetupView' button clicked.")
        ai_instance = await self.cog.get_or_create_ai_instance(user_id, is_setup_flow=True)
        profile_data = ai_instance.profile.user_profile.model_dump() if ai_instance and ai_instance.profile else {}
        modal = CharacterSettingsModal(self.cog, title="步驟 2/3: 您的角色設定", profile_data=profile_data, profile_type='user', is_setup_flow=True, original_interaction_message_id=interaction.message.id)
        await interaction.response.send_modal(modal)
    # 函式：處理「下一步：設定您的角色」按鈕點擊事件
# 類別：繼續到使用者角色設定的視圖

# 類別：繼續到 AI 角色設定的視圖
class ContinueToAiSetupView(discord.ui.View):
    # 函式：初始化 ContinueToAiSetupView
    def __init__(self, *, cog: "BotCog"):
        super().__init__(timeout=None)
        self.cog = cog
    # 函式：初始化 ContinueToAiSetupView

    # 函式：處理「最後一步：設定 AI 戀人」按鈕點擊事件
    @discord.ui.button(label="最後一步：設定 AI 戀人", style=discord.ButtonStyle.primary, custom_id="persistent_continue_to_ai_setup")
    async def continue_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) Persistent 'ContinueToAiSetupView' button clicked.")
        ai_instance = await self.cog.get_or_create_ai_instance(str(interaction.user.id), is_setup_flow=True)
        profile_data = ai_instance.profile.ai_profile.model_dump() if ai_instance and ai_instance.profile else {}
        modal = CharacterSettingsModal(self.cog, title="步驟 3/3: AI 戀人設定", profile_data=profile_data, profile_type='ai', is_setup_flow=True, original_interaction_message_id=interaction.message.id)
        await interaction.response.send_modal(modal)
    # 函式：處理「最後一步：設定 AI 戀人」按鈕點擊事件
# 類別：繼續到 AI 角色設定的視圖

# 類別：繼續到世界聖經設定的視圖
# 更新紀錄:
# v1.5 (2025-09-25): [災難性BUG修復] 修正了UI生命週期管理。將 self.stop() 從 finally 塊中移出，確保在長時異步任務 finalize_setup 完全結束後才停止視圖，從而防止 interaction 失效導致最終訊息發送失敗。
# v1.4 (2025-09-25): [災難性BUG修復] 修正了调用 finalize_setup 時的關鍵字參數名稱。
class ContinueToCanonSetupView(discord.ui.View):
    # 函式：初始化 ContinueToCanonSetupView
    def __init__(self, *, cog: "BotCog"):
        super().__init__(timeout=None)
        self.cog = cog
    # 函式：初始化 ContinueToCanonSetupView

    # 函式：處理「貼上世界聖經」按鈕點擊事件
    @discord.ui.button(label="📄 貼上世界聖經 (文字)", style=discord.ButtonStyle.success, custom_id="persistent_paste_canon")
    async def paste_canon(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) Persistent 'ContinueToCanonSetupView' paste button clicked.")
        modal = WorldCanonPasteModal(self.cog, is_setup_flow=True, original_interaction_message_id=interaction.message.id)
        await interaction.response.send_modal(modal)
    # 函式：處理「貼上世界聖經」按鈕點擊事件

    # 處理「上傳世界聖經」按鈕點擊事件
    @discord.ui.button(label="📄 上傳世界聖經 (.txt)", style=discord.ButtonStyle.success, custom_id="persistent_upload_canon")
    async def upload_canon(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) Persistent 'ContinueToCanonSetupView' upload button clicked.")
        
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(content="**請在 5 分鐘內，直接在此對話中發送您的 `.txt` 世界聖經檔案...**", view=self)

        def check(message: discord.Message):
            return (message.author.id == interaction.user.id and 
                    message.channel.id == interaction.channel.id and 
                    message.attachments and 
                    message.attachments[0].filename.lower().endswith('.txt'))

        try:
            user_message_with_file = await self.cog.bot.wait_for('message', check=check, timeout=300.0)
            attachment = user_message_with_file.attachments[0]
            
            if attachment.size > 5 * 1024 * 1024:
                await interaction.followup.send("❌ 檔案過大！請重新點擊 `/start` 開始。", ephemeral=True)
                self.stop()
                return

            content_bytes = await attachment.read()
            content_text = content_bytes.decode('utf-8', errors='ignore')
            
            # [v1.5 核心修正] 先執行長時任務，執行完畢後才停止視圖
            await self.cog.finalize_setup(interaction, canon_text=content_text)
            self.stop()

        except asyncio.TimeoutError:
            await interaction.followup.send("⏳ 操作已超時。請重新點擊 `/start` 開始。", ephemeral=True)
            self.stop()
        except Exception as e:
            logger.error(f"[{user_id}] 在等待檔案上傳時發生錯誤: {e}", exc_info=True)
            await interaction.followup.send(f"處理您的檔案時發生錯誤: `{e}`。請重新點擊 `/start` 開始。", ephemeral=True)
            self.stop()
    # 處理「上傳世界聖經」按鈕點擊事件

    # 函式：處理「完成設定」按鈕點擊事件
    @discord.ui.button(label="✅ 完成設定並開始冒險 (跳過聖經)", style=discord.ButtonStyle.primary, custom_id="persistent_finalize_setup")
    async def finalize(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) Persistent 'ContinueToCanonSetupView' finalize button clicked.")
        for item in self.children: item.disabled = True
        
        if interaction.message:
            try:
                await interaction.message.edit(view=self)
            except discord.errors.NotFound:
                pass 

        await interaction.response.defer(ephemeral=True)
        # [v1.5 核心修正] 先執行長時任務，執行完畢後才停止視圖
        await self.cog.finalize_setup(interaction, canon_text=None)
        self.stop()
    # 函式：處理「完成設定」按鈕點擊事件
# 類別：繼續到世界聖經設定的視圖




# 類別：重新生成或撤銷回覆的視圖
# 更新紀錄:
# v1.1 (2025-09-23): [功能擴展] 新增了“撤銷”按鈕。此按鈕允許使用者徹底移除上一回合的對話（包括使用者的輸入和AI的回覆），並將短期記憶回滾到上一個狀態，從而實現了類似網頁版AI的“刪除回合”功能。
# v1.0 (2025-11-17): [全新創建] 創建此視圖以支持重新生成功能。
class RegenerateView(discord.ui.View):
    # 函式：初始化 RegenerateView
    def __init__(self, *, cog: "BotCog"):
        super().__init__(timeout=None)
        self.cog = cog
    # 函式：初始化 RegenerateView

    # 函式：處理「重新生成」按鈕點擊事件
    @discord.ui.button(label="🔄 重新生成", style=discord.ButtonStyle.secondary, custom_id="persistent_regenerate_button")
    async def regenerate(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        
        await interaction.response.defer()

        ai_instance = await self.cog.get_or_create_ai_instance(user_id)
        if not ai_instance or not ai_instance.last_user_input:
            await interaction.followup.send("❌ 錯誤：找不到上一回合的對話記錄，無法重新生成。", ephemeral=True)
            return

        try:
            scene_key = ai_instance._get_scene_key()
            if scene_key in ai_instance.scene_histories:
                history = ai_instance.scene_histories[scene_key]
                if len(history.messages) >= 2:
                    # 移除 AI 的回覆和使用者的輸入
                    history.messages.pop()
                    history.messages.pop()
                    logger.info(f"[{user_id}] [重新生成] 已從場景 '{scene_key}' 的短期記憶中撤銷上一回合。")

            # 刪除觸發此操作的 AI 回覆訊息
            await interaction.message.delete()

            logger.info(f"[{user_id}] [重新生成] 正在使用上次輸入重新生成回應...")
            input_data = {"user_input": ai_instance.last_user_input}
            
            final_response, summary_data = await ai_instance.preprocess_and_generate(input_data)

            if final_response and final_response.strip():
                view = self
                for i in range(0, len(final_response), 2000):
                    current_view = view if i + 2000 >= len(final_response) else None
                    await interaction.channel.send(final_response[i:i+2000], view=current_view)
                
                if summary_data:
                    logger.info(f"[{user_id}] [重新生成] 新回應已發送，正在啟動事後處理任務...")
                    asyncio.create_task(ai_instance.update_memories_from_summary(summary_data))
                    asyncio.create_task(ai_instance.execute_lore_updates_from_summary(summary_data))
                else:
                    logger.info(f"[{user_id}] [重新生成] 新回應無摘要數據，跳過事後處理。")
            else:
                await interaction.followup.send("（抱歉，我重新思考了一下，但腦海還是一片空白...）", ephemeral=True)

        except Exception as e:
            logger.error(f"[{user_id}] [重新生成] 流程執行時發生異常: {e}", exc_info=True)
            await interaction.followup.send(f"重新生成時發生了一個嚴重的內部錯誤: `{type(e).__name__}`", ephemeral=True)
    # 函式：處理「重新生成」按鈕點擊事件

    # [v1.1 新增] 函式：處理「撤銷」按鈕點擊事件
    @discord.ui.button(label="🗑️ 撤銷", style=discord.ButtonStyle.danger, custom_id="persistent_undo_button")
    async def undo(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        
        await interaction.response.defer(ephemeral=True)

        ai_instance = await self.cog.get_or_create_ai_instance(user_id)
        if not ai_instance:
            await interaction.followup.send("❌ 錯誤：找不到您的 AI 實例。", ephemeral=True)
            return

        try:
            scene_key = ai_instance._get_scene_key()
            history = ai_instance.scene_histories.get(scene_key)

            if not history or len(history.messages) < 2:
                await interaction.followup.send("❌ 錯誤：沒有足夠的歷史記錄可供撤銷。", ephemeral=True)
                return

            # 步驟 1: 從短期記憶中移除上一回合
            history.messages.pop()  # 移除 AI 的回覆
            last_user_message = history.messages.pop() # 移除使用者的輸入
            logger.info(f"[{user_id}] [撤銷] 已成功從場景 '{scene_key}' 的短期記憶中撤銷上一回合。")

            # 步驟 2: 刪除 Discord 上的訊息
            # 刪除觸發此操作的 AI 回覆訊息
            await interaction.message.delete()
            
            # 嘗試尋找並刪除使用者的上一條訊息
            # 注意：這在私訊中可能無法完美工作，但在頻道中通常有效
            try:
                async for msg in interaction.channel.history(limit=10):
                    if msg.author.id == interaction.user.id and msg.content == last_user_message.content:
                        await msg.delete()
                        logger.info(f"[{user_id}] [撤銷] 已成功刪除使用者的上一條指令訊息。")
                        break
            except (discord.errors.Forbidden, discord.errors.NotFound) as e:
                logger.warning(f"[{user_id}] [撤銷] 刪除使用者訊息時發生非致命錯誤: {e}")
            
            # 步驟 3: 更新 last_user_input 為空，防止重新生成出錯
            ai_instance.last_user_input = None

            await interaction.followup.send("✅ 上一回合已成功撤銷。", ephemeral=True)

        except Exception as e:
            logger.error(f"[{user_id}] [撤銷] 流程執行時發生異常: {e}", exc_info=True)
            await interaction.followup.send(f"撤銷時發生了一個嚴重的內部錯誤: `{type(e).__name__}`", ephemeral=True)
    # [v1.1 新增] 函式：處理「撤銷」按鈕點擊事件
# 類別：重新生成或撤銷回覆的視圖

# 類別：貼上世界聖經的 Modal
class WorldCanonPasteModal(discord.ui.Modal, title="貼上您的世界聖經文本"):
    canon_text = discord.ui.TextInput(label="請將您的世界觀/角色背景故事貼於此處", style=discord.TextStyle.paragraph, placeholder="在此貼上您的 .txt 檔案內容或直接編寫...", required=True, max_length=4000)
    
    # 函式：初始化 WorldCanonPasteModal
    def __init__(self, cog: "BotCog", is_setup_flow: bool = False, original_interaction_message_id: int = None):
        super().__init__(timeout=600.0)
        self.cog = cog
        self.is_setup_flow = is_setup_flow
        self.original_interaction_message_id = original_interaction_message_id
    # 函式：初始化 WorldCanonPasteModal
    
    # 函式：處理 Modal 提交事件
    # 更新紀錄:
    # v1.3 (2025-09-25): [災難性BUG修復] 確保在 await finalize_setup 之後再停止 modal/view，防止 interaction 失效。
    # v1.2 (2025-09-25): [災難性BUG修復] 修正了调用 finalize_setup 時的關鍵字參數名稱。
    async def on_submit(self, interaction: discord.Interaction):
        original_message = None
        if self.original_interaction_message_id:
            try:
                original_message = await interaction.channel.fetch_message(self.original_interaction_message_id)
                view = discord.ui.View.from_message(original_message)
                for item in view.children: item.disabled = True
                await original_message.edit(view=view)
            except (discord.errors.NotFound, AttributeError): pass
        
        if self.is_setup_flow:
            await interaction.response.defer(ephemeral=True)
            await self.cog.finalize_setup(interaction, canon_text=self.canon_text.value)
            # [v1.3 核心修正] 任務完成後再停止相關視圖
            if original_message:
                view = discord.ui.View.from_message(original_message)
                if hasattr(view, 'stop'):
                    view.stop()

        else:
            await interaction.response.send_message("✅ 指令已接收！正在後台為您處理世界聖經...", ephemeral=True)
            asyncio.create_task(self.cog._background_process_canon(interaction=interaction, content_text=self.canon_text.value, is_setup_flow=self.is_setup_flow))

    # 函式：處理 Modal 提交事件
# 類別：貼上世界聖經的 Modal






# 類別：LORE 瀏覽器分頁視圖 (v1.0 - 全新創建)
# 更新紀錄:
# v1.0 (2025-09-23): [全新創建] 創建此類別以支持 /admin_browse_lores 指令。它提供了一個帶有“上一頁”和“下一頁”按鈕的交互式界面，用於分頁顯示大量的LORE條目，解決了Discord自動完成最多只能顯示25個選項的限制。
class LorePaginatorView(discord.ui.View):
    def __init__(self, *, lores: List[Lore], user_id: str, category: str, items_per_page: int = 10):
        super().__init__(timeout=300.0)
        self.lores = lores
        self.user_id = user_id
        self.category = category
        self.items_per_page = items_per_page
        self.current_page = 0
        self.total_pages = (len(self.lores) - 1) // self.items_per_page

    async def _create_embed(self) -> Embed:
        start_index = self.current_page * self.items_per_page
        end_index = start_index + self.items_per_page
        page_lores = self.lores[start_index:end_index]

        embed = Embed(
            title=f"📜 LORE 瀏覽器: {self.category}",
            description=f"正在顯示使用者 `{self.user_id}` 的 LORE 條目。",
            color=discord.Color.gold()
        )

        for lore in page_lores:
            name = lore.content.get('name', lore.content.get('title', lore.key.split(' > ')[-1]))
            description = lore.content.get('description', '無描述。')
            value = (description[:70] + '...') if len(description) > 70 else description
            embed.add_field(name=f"`{name}`", value=f"```{value}```\n🔑 **Key:** `{lore.key}`", inline=False)

        embed.set_footer(text=f"第 {self.current_page + 1} / {self.total_pages + 1} 頁 | 總計 {len(self.lores)} 條")
        return embed

    async def update_message(self, interaction: discord.Interaction):
        self.prev_page.disabled = self.current_page == 0
        self.next_page.disabled = self.current_page == self.total_pages
        embed = await self._create_embed()
        await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(label="⬅️ 上一頁", style=discord.ButtonStyle.secondary)
    async def prev_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_page > 0:
            self.current_page -= 1
            await self.update_message(interaction)

    @discord.ui.button(label="下一頁 ➡️", style=discord.ButtonStyle.secondary)
    async def next_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_page < self.total_pages:
            self.current_page += 1
            await self.update_message(interaction)
# 類別：LORE 瀏覽器分頁視圖 結束






# 類別：設定角色檔案的 Modal
class CharacterSettingsModal(discord.ui.Modal):
    # 函式：初始化 CharacterSettingsModal
    def __init__(self, cog: "BotCog", title: str, profile_data: dict, profile_type: str, is_setup_flow: bool = False, original_interaction_message_id: int = None):
        super().__init__(title=title, timeout=600.0)
        self.cog = cog
        self.profile_type = profile_type
        self.is_setup_flow = is_setup_flow
        self.original_interaction_message_id = original_interaction_message_id
        self.name = discord.ui.TextInput(label="名字 (必填)", default=profile_data.get('name', ''))
        self.gender = discord.ui.TextInput(label="性別 (必填)", default=profile_data.get('gender', ''), placeholder="男 / 女 / 其他")
        self.description = discord.ui.TextInput(label="性格、背景、種族、年齡等綜合描述", style=discord.TextStyle.paragraph, default=profile_data.get('description', ''), max_length=1000)
        self.appearance = discord.ui.TextInput(label="外觀描述 (髮型/瞳色/身材等)", style=discord.TextStyle.paragraph, default=profile_data.get('appearance', ''), required=False, max_length=1000)
        self.add_item(self.name)
        self.add_item(self.gender)
        self.add_item(self.description)
        self.add_item(self.appearance)
    # 函式：初始化 CharacterSettingsModal
        
    # 函式：處理 Modal 提交事件
    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) CharacterSettingsModal submitted for profile_type: '{self.profile_type}', is_setup_flow: {self.is_setup_flow}")
        if self.original_interaction_message_id:
            try:
                original_message = await interaction.channel.fetch_message(self.original_interaction_message_id)
                view = discord.ui.View.from_message(original_message)
                for item in view.children: item.disabled = True
                await original_message.edit(view=view)
            except (discord.errors.NotFound, AttributeError): pass
        ai_instance = await self.cog.get_or_create_ai_instance(user_id, is_setup_flow=self.is_setup_flow)
        if not ai_instance or not ai_instance.profile:
            await interaction.followup.send("錯誤：AI 核心或設定檔案未初始化。", ephemeral=True)
            return
        profile_attr = f"{self.profile_type}_profile"
        try:
            profile_to_update = getattr(ai_instance.profile, profile_attr)
            profile_to_update.name = self.name.value
            profile_to_update.gender = self.gender.value
            profile_to_update.description = self.description.value
            profile_to_update.appearance = self.appearance.value
            await ai_instance.update_and_persist_profile({profile_attr: profile_to_update.model_dump()})
            if not self.is_setup_flow:
                await interaction.followup.send(f"✅ **{profile_to_update.name}** 的角色設定已成功更新！", ephemeral=True)
            elif self.profile_type == 'user': 
                view = ContinueToAiSetupView(cog=self.cog)
                await interaction.followup.send("✅ 您的角色已設定！\n請點擊下方按鈕，為您的 AI 戀人進行設定。", view=view, ephemeral=True)
            elif self.profile_type == 'ai':
                view = ContinueToCanonSetupView(cog=self.cog)
                await interaction.followup.send("✅ AI 戀人基礎設定完成！\n\n**下一步 (可選):**\n請點擊下方按鈕提供您的「世界聖經」，或直接點擊「完成設定」以開始冒險。", view=view, ephemeral=True)
        except Exception as e:
            logger.error(f"[{user_id}] 處理角色設定 Modal 提交時出錯: {e}", exc_info=True)
            await interaction.followup.send("錯誤：在處理您的設定時遇到問題。", ephemeral=True)
            if self.is_setup_flow: self.cog.setup_locks.discard(user_id)
    # 函式：處理 Modal 提交事件
# 類別：設定角色檔案的 Modal

# 類別：設定世界觀的 Modal
class WorldSettingsModal(discord.ui.Modal):
    # 函式：初始化 WorldSettingsModal
    def __init__(self, cog: "BotCog", current_world: str, is_setup_flow: bool = False, original_interaction_message_id: int = None):
        super().__init__(title="步驟 1/3: 世界觀設定", timeout=600.0)
        self.cog = cog
        self.is_setup_flow = is_setup_flow
        self.original_interaction_message_id = original_interaction_message_id
        self.world_settings = discord.ui.TextInput(label="世界觀核心原則", style=discord.TextStyle.paragraph, max_length=4000, default=current_world, placeholder="請描述這個世界的基本規則...")
        self.add_item(self.world_settings)
    # 函式：初始化 WorldSettingsModal
        
    # 函式：處理 Modal 提交事件
    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) WorldSettingsModal submitted. is_setup_flow: {self.is_setup_flow}")
        if self.original_interaction_message_id:
            try:
                original_message = await interaction.channel.fetch_message(self.original_interaction_message_id)
                view = discord.ui.View.from_message(original_message)
                for item in view.children: item.disabled = True
                await original_message.edit(view=view)
            except (discord.errors.NotFound, AttributeError): pass
        ai_instance = await self.cog.get_or_create_ai_instance(user_id, is_setup_flow=self.is_setup_flow)
        if not ai_instance:
            await interaction.followup.send("錯誤：無法初始化 AI 核心。", ephemeral=True)
            if self.is_setup_flow: self.cog.setup_locks.discard(user_id)
            return
        await ai_instance.update_and_persist_profile({'world_settings': self.world_settings.value})
        if self.is_setup_flow:
            view = ContinueToUserSetupView(cog=self.cog)
            await interaction.followup.send("✅ 世界觀已設定！\n請點擊下方按鈕，開始設定您的個人角色。", view=view, ephemeral=True)
        else:
            await interaction.followup.send("✅ 世界觀設定已成功更新！", ephemeral=True)
    # 函式：處理 Modal 提交事件
# 類別：設定世界觀的 Modal

# 類別：設定回覆風格的 Modal
class ResponseStyleModal(discord.ui.Modal, title="自訂 AI 回覆風格"):
    response_style = discord.ui.TextInput(label="回覆風格指令", style=discord.TextStyle.paragraph, placeholder="在此處定義 AI 的敘事和對話風格...", required=True, max_length=4000)
    
    # 函式：初始化 ResponseStyleModal
    def __init__(self, cog: "BotCog", current_style: str):
        super().__init__()
        self.cog = cog
        self.response_style.default = current_style
    # 函式：初始化 ResponseStyleModal
        
    # 函式：處理 Modal 提交事件
    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user_id = str(interaction.user.id)
        ai_instance = await self.cog.get_or_create_ai_instance(user_id)
        if not ai_instance:
            await interaction.followup.send("錯誤：找不到您的使用者資料。", ephemeral=True)
            return
        if await ai_instance.update_and_persist_profile({'response_style_prompt': self.response_style.value}):
            await interaction.followup.send("✅ AI 回覆風格已成功更新！", ephemeral=True)
        else:
            await interaction.followup.send("錯誤：更新 AI 回覆風格失敗。", ephemeral=True)
    # 函式：處理 Modal 提交事件
# 類別：設定回覆風格的 Modal

# 類別：強制重啟 /start 流程的視圖
class ForceRestartView(discord.ui.View):
    # 函式：初始化 ForceRestartView
    def __init__(self, *, cog: "BotCog"):
        super().__init__(timeout=180.0)
        self.cog = cog
        self.original_interaction_user_id = None
    # 函式：初始化 ForceRestartView
        
    # 函式：檢查互動是否來自原始使用者
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.original_interaction_user_id:
            await interaction.response.send_message("你無法操作不屬於你的指令。", ephemeral=True)
            return False
        return True
    # 函式：檢查互動是否來自原始使用者
        
    # 函式：處理「強制終止並重新開始」按鈕點擊事件
    @discord.ui.button(label="強制終止並重新開始", style=discord.ButtonStyle.danger)
    async def force_restart(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        for item in self.children: item.disabled = True
        await interaction.edit_original_response(content="正在強制終止舊流程並為您重置所有資料，請稍候...", view=self)
        await self.cog.start_reset_flow(interaction)
        self.stop()
    # 函式：處理「強制終止並重新開始」按鈕點擊事件
        
    # 函式：處理「取消」按鈕點擊事件
    @discord.ui.button(label="取消本次操作", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="操作已取消。", view=None)
        self.stop()
    # 函式：處理「取消」按鈕點擊事件
# 類別：強制重啟 /start 流程的視圖

# 類別：確認 /start 重置的視圖 (v53.0 - 臨時視圖修正)
# 更新紀錄:
# v53.0 (2025-11-22): [災難性BUG修復] 徹底重構了此視圖的實現方式。移除了按鈕的 custom_id 並恢復了 timeout，將其從一個錯誤的「持久化視圖」改為正確的「臨時狀態視圖」。此修改解決了因全局註冊導致 interaction_check 永遠失敗，從而使按鈕無響應的根本問題。
# v52.0 (2025-11-22): [架構調整] 引入此視圖以提供更安全的重置流程。
# v50.0 (2025-11-14): [完整性修復] 提供了此檔案的完整版本。
class ConfirmStartView(discord.ui.View):
    # 函式：初始化 ConfirmStartView
    def __init__(self, *, cog: "BotCog"):
        # [v53.0 核心修正] 臨時視圖必須有超時
        super().__init__(timeout=180.0)
        self.cog = cog
        self.original_interaction_user_id = None
    # 初始化 ConfirmStartView 函式結束
        
    # 函式：檢查互動是否來自原始使用者
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.original_interaction_user_id:
            await interaction.response.send_message("你無法操作不屬於你的指令。", ephemeral=True)
            return False
        return True
    # 檢查互動是否來自原始使用者 函式結束
        
    # 函式：處理「確認重置」按鈕點擊事件
    # [v53.0 核心修正] 移除了 custom_id
    @discord.ui.button(label="【確認重置並開始】", style=discord.ButtonStyle.danger)
    async def confirm_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.cog.setup_locks.add(str(interaction.user.id))
        # 在回應前先禁用按鈕，提供即時反饋
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(content="正在為您重置所有資料，請稍候...", view=self)
        # 將耗時操作作為背景任務執行，避免互動超時
        asyncio.create_task(self.cog.start_reset_flow(interaction))
        self.stop()
    # 處理「確認重置」按鈕點擊事件 函式結束
        
    # 函式：處理「取消」按鈕點擊事件
    # [v53.0 核心修正] 移除了 custom_id
    @discord.ui.button(label="取消", style=discord.ButtonStyle.secondary)
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="操作已取消。", view=None)
        self.stop()
    # 處理「取消」按鈕點擊事件 函式結束
        
    # 函式：處理視圖超時事件
    async def on_timeout(self):
        # 確保超時後按鈕也會被禁用
        for item in self.children:
            item.disabled = True
        # 這裡可以選擇編輯原始訊息，告知使用者操作已超時
        # try:
        #     await self.message.edit(content="操作已超時，請重新發起指令。", view=self)
        # except discord.HTTPException:
        #     pass
    # 處理視圖超時事件 函式結束
# 確認 /start 重置的視圖 類別結束

# 類別：/settings 指令的選擇視圖
class SettingsChoiceView(discord.ui.View):
    # 函式：初始化 SettingsChoiceView
    def __init__(self, cog: "BotCog"):
        super().__init__(timeout=180)
        self.cog = cog
    # 函式：初始化 SettingsChoiceView
        
    # 函式：處理「使用者角色設定」按鈕點擊事件
    @discord.ui.button(label="👤 使用者角色設定", style=discord.ButtonStyle.primary, emoji="👤")
    async def user_settings_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        ai_instance = await self.cog.get_or_create_ai_instance(str(interaction.user.id))
        profile_data = ai_instance.profile.user_profile.model_dump() if ai_instance and ai_instance.profile else {}
        modal = CharacterSettingsModal(self.cog, title="👤 使用者角色設定", profile_data=profile_data, profile_type='user', is_setup_flow=False)
        await interaction.response.send_modal(modal)
    # 函式：處理「使用者角色設定」按鈕點擊事件
        
    # 函式：處理「AI 戀人設定」按鈕點擊事件
    @discord.ui.button(label="❤️ AI 戀人設定", style=discord.ButtonStyle.success, emoji="❤️")
    async def ai_settings_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        ai_instance = await self.cog.get_or_create_ai_instance(str(interaction.user.id))
        profile_data = ai_instance.profile.ai_profile.model_dump() if ai_instance and ai_instance.profile else {}
        modal = CharacterSettingsModal(self.cog, title="❤️ AI 戀人設定", profile_data=profile_data, profile_type='ai', is_setup_flow=False)
        await interaction.response.send_modal(modal)
    # 函式：處理「AI 戀人設定」按鈕點擊事件
        
    # 函式：處理「世界觀設定」按鈕點擊事件
    @discord.ui.button(label="🌍 世界觀設定", style=discord.ButtonStyle.secondary, emoji="🌍")
    async def world_settings_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        ai_instance = await self.cog.get_or_create_ai_instance(str(interaction.user.id))
        world_settings = ai_instance.profile.world_settings if ai_instance and ai_instance.profile else ""
        modal = WorldSettingsModal(self.cog, current_world=world_settings, is_setup_flow=False)
        await interaction.response.send_modal(modal)
    # 函式：處理「世界觀設定」按鈕點擊事件
# 類別：/settings 指令的選擇視圖

# 類別：確認編輯角色檔案的視圖
class ConfirmEditView(discord.ui.View):
    # 函式：初始化 ConfirmEditView
    def __init__(self, *, cog: "BotCog", target_type: Literal['user', 'ai', 'npc'], target_key: str, new_description: str):
        super().__init__(timeout=300.0)
        self.cog = cog
        self.target_type = target_type
        self.target_key = target_key
        self.new_description = new_description
    # 函式：初始化 ConfirmEditView
        
    # 函式：處理「確認儲存」按鈕點擊事件
    @discord.ui.button(label="✅ 確認儲存", style=discord.ButtonStyle.success)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user_id = str(interaction.user.id)
        display_name = self.target_key.split(' > ')[-1]
        ai_instance = await self.cog.get_or_create_ai_instance(user_id)
        if not ai_instance or not ai_instance.profile:
            await interaction.followup.send("錯誤：無法獲取 AI 實例。", ephemeral=True)
            return
        try:
            if self.target_type in ['user', 'ai']:
                profile_attr = 'user_profile' if self.target_type == 'user' else 'ai_profile'
                profile_obj = getattr(ai_instance.profile, profile_attr)
                profile_obj.description = self.new_description
                await ai_instance.update_and_persist_profile({profile_attr: profile_obj.model_dump()})
            elif self.target_type == 'npc':
                lore = await lore_book.get_lore(user_id, 'npc_profile', self.target_key)
                if not lore:
                    await interaction.followup.send(f"錯誤：找不到名為 {display_name} 的 NPC。", ephemeral=True)
                    return
                lore.content['description'] = self.new_description
                await lore_book.add_or_update_lore(user_id, 'npc_profile', self.target_key, lore.content)
                await ai_instance.initialize()
            await interaction.followup.send(f"✅ 角色 **{display_name}** 的檔案已成功更新！", ephemeral=True)
            await interaction.edit_original_response(content=f"角色 **{display_name}** 的檔案已更新。", view=None, embed=None)
        except Exception as e:
            logger.error(f"儲存角色 {display_name} 的新描述時出錯: {e}", exc_info=True)
            await interaction.followup.send("儲存更新時發生嚴重錯誤。", ephemeral=True)
        self.stop()
    # 函式：處理「確認儲存」按鈕點擊事件
        
    # 函式：處理「取消」按鈕點擊事件
    @discord.ui.button(label="❌ 取消", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="操作已取消。", view=None, embed=None)
        self.stop()
    # 函式：處理「取消」按鈕點擊事件
# 類別：確認編輯角色檔案的視圖

# 類別：編輯角色檔案的 Modal
class ProfileEditModal(discord.ui.Modal):
    edit_instruction = discord.ui.TextInput(label="修改指令", style=discord.TextStyle.paragraph, placeholder="請用自然語言描述您想如何修改這個角色...", required=True, max_length=1000)
    
    # 函式：初始化 ProfileEditModal
    def __init__(self, *, cog: "BotCog", target_type: Literal['user', 'ai', 'npc'], target_key: str, display_name: str, original_description: str):
        super().__init__(title=f"編輯角色：{display_name}")
        self.cog = cog
        self.target_type = target_type
        self.target_key = target_key
        self.display_name = display_name
        self.original_description = original_description
    # 函式：初始化 ProfileEditModal
        
    # 函式：處理 Modal 提交事件
    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user_id = str(interaction.user.id)
        try:
            ai_instance = await self.cog.get_or_create_ai_instance(user_id)
            if not ai_instance:
                await interaction.followup.send("錯誤：無法初始化 AI 核心。", ephemeral=True)
                return
            rewriting_chain = ai_instance.get_profile_rewriting_chain()
            new_description = await ai_instance.ainvoke_with_rotation(rewriting_chain, {"original_description": self.original_description, "edit_instruction": self.edit_instruction.value})
            if not new_description:
                await interaction.followup.send("錯誤：AI 未能根據您的指令生成新的描述。", ephemeral=True)
                return
            embed = Embed(title=f"✍️ 角色檔案更新預覽：{self.display_name}", color=discord.Color.orange())
            original_desc_preview = (self.original_description[:450] + '...') if len(self.original_description) > 450 else self.original_description
            new_desc_preview = (new_description[:450] + '...') if len(new_description) > 450 else new_description
            embed.add_field(name="📜 修改前", value=f"```{original_desc_preview}```", inline=False)
            embed.add_field(name="✨ 修改後", value=f"```{new_desc_preview}```", inline=False)
            embed.set_footer(text="請確認修改後的內容，然後點擊下方按鈕儲存。")
            view = ConfirmEditView(cog=self.cog, target_type=self.target_type, target_key=self.target_key, new_description=new_description)
            await interaction.followup.send(embed=embed, view=view, ephemeral=True)
        except Exception as e:
            logger.error(f"[{user_id}] 在編輯角色 '{self.display_name}' 時發生錯誤: {e}", exc_info=True)
            await interaction.followup.send(f"生成角色預覽時發生嚴重錯誤: {e}", ephemeral=True)
    # 函式：處理 Modal 提交事件
# 類別：編輯角色檔案的 Modal

# 函式：建立角色檔案的 Embed
def _create_profile_embed(profile: CharacterProfile, title_prefix: str) -> Embed:
    embed = Embed(title=f"{title_prefix}：{profile.name}", color=discord.Color.blue())
    base_info = [f"**性別:** {profile.gender or '未設定'}", f"**年齡:** {profile.age or '未知'}", f"**種族:** {profile.race or '未知'}"]
    embed.add_field(name="基礎資訊", value="\n".join(base_info), inline=False)
    if profile.description: embed.add_field(name="📜 核心描述", value=f"```{profile.description[:1000]}```", inline=False)
    if profile.appearance: embed.add_field(name="🎨 外觀總覽", value=f"```{profile.appearance[:1000]}```", inline=False)
    if profile.appearance_details: embed.add_field(name="✨ 外觀細節", value="\n".join([f"- {k}: {v}" for k, v in profile.appearance_details.items()]), inline=True)
    if profile.equipment: embed.add_field(name="⚔️ 當前裝備", value="、".join(profile.equipment), inline=True)
    if profile.skills: embed.add_field(name="🌟 掌握技能", value="、".join(profile.skills), inline=True)
    return embed
# 函式：建立角色檔案的 Embed

# 類別：確認並編輯角色檔案的視圖
class ConfirmAndEditView(discord.ui.View):
    # 函式：初始化 ConfirmAndEditView
    def __init__(self, *, cog: "BotCog", target_type: Literal['user', 'ai', 'npc'], target_key: str, display_name: str, original_description: str):
        super().__init__(timeout=300.0)
        self.cog = cog
        self.target_type = target_type
        self.target_key = target_key
        self.display_name = display_name
        self.original_description = original_description
    # 函式：初始化 ConfirmAndEditView
        
    # 函式：處理「點此開始編輯」按鈕點擊事件
    @discord.ui.button(label="✍️ 點此開始編輯", style=discord.ButtonStyle.success)
    async def edit(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = ProfileEditModal(cog=self.cog, target_type=self.target_type, target_key=self.target_key, display_name=self.display_name, original_description=self.original_description)
        await interaction.response.send_modal(modal)
        self.stop()
        await interaction.message.edit(view=self)
    # 函式：處理「點此開始編輯」按鈕點擊事件
        
    # 函式：處理視圖超時事件
    async def on_timeout(self):
        for item in self.children: item.disabled = True
    # 函式：處理視圖超時事件
# 類別：確認並編輯角色檔案的視圖

# 類別：編輯 NPC 的下拉選單
class NpcEditSelect(discord.ui.Select):
    # 函式：初始化 NpcEditSelect
    def __init__(self, cog: "BotCog", all_npcs: List[Lore]):
        self.cog = cog
        self.all_npcs = {npc.key: npc for npc in all_npcs}
        options = []
        for lore in all_npcs:
            content = lore.content
            name = content.get('name', '未知名稱')
            description_part = (content.get('description', '未知')[:50] + '...') if content.get('description') else '未知'
            options.append(discord.SelectOption(label=name[:100], description=description_part[:100], value=lore.key[:100]))
        super().__init__(placeholder="選擇一位您想編輯的 NPC...", min_values=1, max_values=1, options=options)
    # 函式：初始化 NpcEditSelect
        
    # 函式：處理下拉選單選擇事件
    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        selected_key = self.values[0]
        lore = self.all_npcs.get(selected_key)
        if not lore:
            await interaction.followup.send("錯誤：找不到所選的NPC資料。", ephemeral=True)
            return
        profile = CharacterProfile.model_validate(lore.content)
        embed = _create_profile_embed(profile, "👥 NPC 檔案")
        view = ConfirmAndEditView(cog=self.cog, target_type='npc', target_key=selected_key, display_name=profile.name, original_description=profile.description or "")
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)
        self.disabled = True
        await interaction.edit_original_response(view=self.view)
    # 函式：處理下拉選單選擇事件
# 類別：編輯 NPC 的下拉選單

# 類別：/edit_profile 指令的根視圖
class EditProfileRootView(discord.ui.View):
    # 函式：初始化 EditProfileRootView
    def __init__(self, cog: "BotCog", original_user_id: int):
        super().__init__(timeout=180)
        self.cog = cog
        self.original_user_id = original_user_id
    # 函式：初始化 EditProfileRootView
        
    # 函式：檢查互動是否來自原始使用者
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.original_user_id:
            await interaction.response.send_message("你無法操作不屬於你的指令。", ephemeral=True)
            return False
        return True
    # 函式：檢查互動是否來自原始使用者
        
    # 函式：發送角色檔案以供編輯的輔助函式
    async def _send_profile_for_editing(self, interaction: discord.Interaction, target_type: Literal['user', 'ai']):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user_id = str(interaction.user.id)
        ai_instance = await self.cog.get_or_create_ai_instance(user_id)
        if not ai_instance or not ai_instance.profile:
            await interaction.followup.send("錯誤：找不到您的使用者資料。", ephemeral=True)
            return
        profile = ai_instance.profile.user_profile if target_type == 'user' else ai_instance.profile.ai_profile
        title_prefix = "👤 您的角色檔案" if target_type == 'user' else "❤️ AI 戀人檔案"
        embed = _create_profile_embed(profile, title_prefix)
        view = ConfirmAndEditView(cog=self.cog, target_type=target_type, target_key=profile.name, display_name=profile.name, original_description=profile.description or "")
        await interaction.followup.send("這是您選擇角色的當前檔案，請預覽後點擊按鈕進行修改：", embed=embed, view=view, ephemeral=True)
    # 函式：發送角色檔案以供編輯的輔助函式
        
    # 函式：處理「編輯我的檔案」按鈕點擊事件
    @discord.ui.button(label="👤 編輯我的檔案", style=discord.ButtonStyle.primary)
    async def edit_user(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._send_profile_for_editing(interaction, 'user')
    # 函式：處理「編輯我的檔案」按鈕點擊事件
        
    # 函式：處理「編輯 AI 戀人檔案」按鈕點擊事件
    @discord.ui.button(label="❤️ 編輯 AI 戀人檔案", style=discord.ButtonStyle.success)
    async def edit_ai(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._send_profile_for_editing(interaction, 'ai')
    # 函式：處理「編輯 AI 戀人檔案」按鈕點擊事件
        
    # 函式：處理「編輯 NPC 檔案」按鈕點擊事件
    @discord.ui.button(label="👥 編輯 NPC 檔案", style=discord.ButtonStyle.secondary)
    async def edit_npc(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user_id = str(interaction.user.id)
        all_npcs = await lore_book.get_lores_by_category_and_filter(user_id, 'npc_profile')
        if not all_npcs:
            await interaction.followup.send("您的世界中還沒有任何 NPC 可供編輯。", ephemeral=True)
            return
        view = discord.ui.View(timeout=180)
        view.add_item(NpcEditSelect(self.cog, all_npcs))
        await interaction.followup.send("請從下方選單中選擇您要編輯的 NPC：", view=view, ephemeral=True)
    # 函式：處理「編輯 NPC 檔案」按鈕點擊事件
# 類別：/edit_profile 指令的根視圖

# 類別：版本控制 - 創建新 Tag 的 Modal
class CreateTagModal(discord.ui.Modal, title="創建新版本 (Tag)"):
    version = discord.ui.TextInput(label="版本號", placeholder="v1.2.1", required=True)
    description = discord.ui.TextInput(label="版本描述 (可選)", style=discord.TextStyle.paragraph, placeholder="簡短描述此版本的變更", required=False)
    
    # 函式：初始化 CreateTagModal
    def __init__(self, view: "VersionControlView"):
        super().__init__()
        self.view = view
    # 函式：初始化 CreateTagModal
        
    # 函式：處理 Modal 提交事件
    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        success, message = await self.view.cog._git_create_tag(self.version.value, self.description.value)
        if success:
            await interaction.followup.send(f"✅ **版本創建成功！**\nTag: `{self.version.value}`。", ephemeral=True)
            await self.view.update_message(interaction)
        else:
            await interaction.followup.send(f"❌ **版本創建失敗！**\n```\n{message}\n```", ephemeral=True)
    # 函式：處理 Modal 提交事件
# 類別：版本控制 - 創建新 Tag 的 Modal

# 類別：版本控制 - 回退版本的下拉選單
class RollbackSelect(discord.ui.Select):
    # 函式：初始化 RollbackSelect
    def __init__(self, tags: List[str]):
        options = [discord.SelectOption(label=tag, value=tag) for tag in tags] or [discord.SelectOption(label="沒有可用的版本", value="disabled")]
        super().__init__(placeholder="選擇要回退到的版本...", options=options, disabled=not tags)
    # 函式：初始化 RollbackSelect
        
    # 函式：處理下拉選單選擇事件
    async def callback(self, interaction: discord.Interaction):
        await self.view.show_rollback_confirmation(interaction, self.values[0])
    # 函式：處理下拉選單選擇事件
# 類別：版本控制 - 回退版本的下拉選單

# 類別：版本控制主視圖
class VersionControlView(discord.ui.View):
    # 函式：初始化 VersionControlView
    def __init__(self, cog: "BotCog", original_user_id: int):
        super().__init__(timeout=300)
        self.cog = cog
        self.original_user_id = original_user_id
        self.selected_rollback_version = None
    # 函式：初始化 VersionControlView
        
    # 函式：檢查互動是否來自原始使用者
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.original_user_id:
            await interaction.response.send_message("你無法操作此面板。", ephemeral=True)
            return False
        return True
    # 函式：檢查互動是否來自原始使用者
        
    # 函式：更新面板訊息
    async def update_message(self, interaction: discord.Interaction, show_select: bool = False):
        self.clear_items()
        self.add_item(self.refresh_button)
        self.add_item(self.create_tag_button)
        self.add_item(self.rollback_button)
        if show_select:
            success, tags_or_error = await self.cog._git_get_remote_tags()
            if success: self.add_item(RollbackSelect(tags_or_error))
            else:
                await interaction.edit_original_response(content=f"❌ 獲取版本列表失敗:\n```\n{tags_or_error}\n```", embed=None, view=self)
                return
        embed = await self._build_embed()
        await interaction.edit_original_response(content=None, embed=embed, view=self)
    # 函式：更新面板訊息
        
    # 函式：建立版本資訊 Embed
    async def _build_embed(self) -> discord.Embed:
        success, version_or_error = await self.cog._git_get_current_version()
        if success:
            embed = discord.Embed(title="⚙️ 版本控制面板", description="伺服器當前運行的程式碼版本。", color=discord.Color.blue())
            embed.add_field(name="🏷️ 當前版本", value=f"```\n{version_or_error}\n```", inline=False)
        else:
            embed = discord.Embed(title="⚙️ 版本控制面板", description="❌ 無法獲取當前版本資訊。", color=discord.Color.red())
            embed.add_field(name="錯誤詳情", value=f"```\n{version_or_error}\n```", inline=False)
        embed.set_footer(text="請使用下方按鈕進行操作。")
        return embed
    # 函式：建立版本資訊 Embed
        
    # 函式：處理「刷新」按鈕點擊事件
    @discord.ui.button(label="🔄 刷新", style=discord.ButtonStyle.success, custom_id="vc_refresh")
    async def refresh_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        await self.update_message(interaction)
    # 函式：處理「刷新」按鈕點擊事件
        
    # 函式：處理「創建新版本」按鈕點擊事件
    @discord.ui.button(label="➕ 創建新版本", style=discord.ButtonStyle.primary, custom_id="vc_create_tag")
    async def create_tag_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(CreateTagModal(self))
    # 函式：處理「創建新版本」按鈕點擊事件
        
    # 函式：處理「回退版本」按鈕點擊事件
    @discord.ui.button(label="⏪ 回退版本", style=discord.ButtonStyle.secondary, custom_id="vc_rollback")
    async def rollback_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        await self.update_message(interaction, show_select=True)
    # 函式：處理「回退版本」按鈕點擊事件
        
    # 函式：顯示回退確認介面
    async def show_rollback_confirmation(self, interaction: discord.Interaction, version: str):
        self.selected_rollback_version = version
        self.clear_items()
        confirm_button = discord.ui.Button(label=f"【確認回退到 {version}】", style=discord.ButtonStyle.danger, custom_id="vc_confirm_rollback")
        cancel_button = discord.ui.Button(label="取消", style=discord.ButtonStyle.secondary, custom_id="vc_cancel_rollback")
        async def confirm_callback(interaction: discord.Interaction):
            await interaction.response.defer(ephemeral=True, thinking=True)
            await interaction.edit_original_response(content=f"⏳ **正在執行回滾到 `{self.selected_rollback_version}`...**", embed=None, view=None)
            success, message = await self.cog._git_rollback_version(self.selected_rollback_version)
            if success: await interaction.followup.send("✅ **回滾指令已發送！** 伺服器正在重啟。", ephemeral=True)
            else:
                await interaction.followup.send(f"❌ **回滾失敗！**\n```\n{message}\n```", ephemeral=True)
                await self.update_message(interaction)
        async def cancel_callback(interaction: discord.Interaction):
            await interaction.response.defer()
            await self.update_message(interaction)
        confirm_button.callback = confirm_callback
        cancel_button.callback = cancel_callback
        self.add_item(confirm_button)
        self.add_item(cancel_button)
        embed = await self._build_embed()
        embed.color = discord.Color.red()
        embed.add_field(name="⚠️ 最終確認", value=f"您確定要將伺服器程式碼回退到 **`{version}`** 嗎？", inline=False)
        await interaction.edit_original_response(embed=embed, view=self)
    # 函式：顯示回退確認介面
# 類別：版本控制主視圖

# 類別：機器人核心功能集 (Cog)
class BotCog(commands.Cog):
    # 函式：初始化 BotCog
    def __init__(self, bot: "AILoverBot", git_lock: asyncio.Lock):
        self.bot = bot
        self.ai_instances: dict[str, AILover] = {}
        self.setup_locks: set[str] = set()
        self.git_lock = git_lock
    # 函式：初始化 BotCog

    # 函式：Cog 卸載時執行的清理
    def cog_unload(self):
        self.connection_watcher.cancel()
    # 函式：Cog 卸載時執行的清理

# 函式：獲取或創建使用者的 AI 實例 (v52.0 - 按需加載記憶)
# 更新紀錄:
# v52.0 (2025-11-22): [重大架構升級] 根據「按需加載」原則，在此函式中增加了對 ai_instance._rehydrate_scene_histories() 的調用。此修改確保了短期記憶只在用戶開始一個新會話、首次創建AI實例時從資料庫恢復一次，避免了在 /start 流程中錯誤地恢復舊記憶。
# v50.0 (2025-11-14): [完整性修復] 提供了此檔案的完整版本。
# v48.0 (2025-10-19): [重大架構重構] 徹底移除了對 LangGraph 的所有依賴。
    async def get_or_create_ai_instance(self, user_id: str, is_setup_flow: bool = False) -> AILover | None:
        if user_id in self.ai_instances:
            return self.ai_instances[user_id]
        
        logger.info(f"使用者 {user_id} 沒有活躍的 AI 實例，嘗試創建...")
        ai_instance = AILover(user_id)
        
        if await ai_instance.initialize():
            logger.info(f"為使用者 {user_id} 成功創建並初始化 AI 實例。")
            
            # [v52.0 核心修正] 在實例成功初始化後，立即為其恢復短期記憶
            await ai_instance._rehydrate_scene_histories()

            self.ai_instances[user_id] = ai_instance
            return ai_instance
        elif is_setup_flow:
            logger.info(f"[{user_id}] 處於設定流程中，即使資料庫無記錄，也創建一個臨時的記憶體實例。")
            ai_instance.profile = UserProfile(user_id=user_id, user_profile=CharacterProfile(name=""), ai_profile=CharacterProfile(name=""))
            try:
                await ai_instance._configure_pre_requisites()
            except Exception as e:
                logger.error(f"[{user_id}] 為臨時實例配置前置資源時失敗: {e}", exc_info=True)
            self.ai_instances[user_id] = ai_instance
            return ai_instance
        else:
            logger.warning(f"為使用者 {user_id} 初始化 AI 實例失敗。")
            return None
# 獲取或創建使用者的 AI 實例 函式結束

    # 函式：安全地異步執行 Git 命令並返回結果
    async def _run_git_command(self, command: List[str]) -> Tuple[bool, str]:
        async with self.git_lock:
            try:
                process = await asyncio.to_thread(
                    subprocess.run, 
                    command, 
                    capture_output=True, 
                    text=True, 
                    encoding='utf-8', 
                    check=True, 
                    cwd=PROJ_DIR
                )
                return True, process.stdout.strip()
            except subprocess.CalledProcessError as e:
                error_message = e.stderr.strip() or e.stdout.strip()
                logger.error(f"Git指令 '{' '.join(command)}' 執行失敗: {error_message}")
                return False, error_message
            except Exception as e: 
                logger.error(f"執行 Git 指令時發生未知錯誤: {e}", exc_info=True)
                return False, str(e)
    # 函式：安全地異步執行 Git 命令並返回結果

    # 函式：獲取當前的 Git 版本描述
    async def _git_get_current_version(self) -> Tuple[bool, str]:
        return await self._run_git_command(["git", "describe", "--tags", "--always"])
    # 函式：獲取當前的 Git 版本描述

    # 函式：獲取所有遠程 Git 標籤 (版本) 列表
    async def _git_get_remote_tags(self) -> Tuple[bool, List[str]]:
        await self._run_git_command(["git", "fetch", "--tags", "--force"])
        success, msg = await self._run_git_command(["git", "tag", "-l", "--sort=-v:refname"])
        return (True, msg.splitlines()) if success else (False, [msg])
    # 函式：獲取所有遠程 Git 標籤 (版本) 列表

    # 函式：創建並推送一個新的 Git 標籤 (版本)
    async def _git_create_tag(self, version: str, description: str) -> Tuple[bool, str]:
        success, msg = await self._run_git_command(["git", "status", "--porcelain"])
        if success and msg: return False, "錯誤：工作區尚有未提交的變更。"
        success, msg = await self._run_git_command(["git", "tag", "-a", version, "-m", description])
        if not success: return False, f"創建Tag失敗: {msg}"
        success, msg = await self._run_git_command(["git", "push", "origin", version])
        if not success:
            await self._run_git_command(["git", "tag", "-d", version])
            return False, f"推送Tag失敗: {msg}"
        return True, f"成功創建並推送Tag {version}"
    # 函式：創建並推送一個新的 Git 標籤 (版本)

    # 函式：回退到指定的 Git 標籤 (版本) 並觸發重啟
    async def _git_rollback_version(self, version: str) -> Tuple[bool, str]:
        logger.info(f"管理員觸發版本回退至: {version}")
        success, msg = await self._run_git_command(["git", "checkout", f"tags/{version}"])
        if not success: return False, f"Checkout失敗: {msg}"
        pip_command = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        try:
            await asyncio.to_thread(subprocess.run, pip_command, check=True, capture_output=True)
        except Exception as e: 
            logger.error(f"安裝依賴項時失敗: {e}", exc_info=True)
            return False, f"安裝依賴項失敗: {e}"
        if self.bot.shutdown_event: self.bot.shutdown_event.set()
        return True, "回退指令已發送，伺服器正在重啟。"
    # 函式：回退到指定的 Git 標籤 (版本) 並觸發重啟

    # 函式：Discord 連線健康檢查與狀態更新的背景任務
    @tasks.loop(seconds=240)
    async def connection_watcher(self):
        try:
            await self.bot.wait_until_ready()
            if math.isinf(self.bot.latency): 
                logger.critical("【重大錯誤】與 Discord 的 WebSocket 連線已中斷！")
            else: 
                await self.bot.change_presence(activity=discord.Game(name="與你共度時光"))
        except Exception as e: 
            logger.error(f"【健康檢查】任務中發生未預期的錯誤: {e}", exc_info=True)
    # 函式：Discord 連線健康檢查與狀態更新的背景任務

    # 函式：在 connection_watcher 任務首次運行前執行的設置
    @connection_watcher.before_loop
    async def before_connection_watcher(self):
        await self.bot.wait_until_ready()
        logger.info("【健康檢查 & Keep-Alive】背景任務已啟動。")
    # 函式：在 connection_watcher 任務首次運行前執行的設置

# 函式：監聽並處理所有符合條件的訊息 (v58.0 - 雙重保險LORE提取)
# 更新紀錄:
# v58.0 (2025-11-21): [重大架構升級] 移除了依賴主模型摘要的LORE提取流程，改為總是無條件地創建一個獨立的 `_background_lore_extraction` 背景任務。此修改實現了「雙重保險」機制，確保即使主模型判斷失誤，專門的提取鏈也能捕獲並創建被遺漏的LORE。
# v55.0 (2025-11-16): [功能整合] 整合了「重新生成」功能。
# v54.2 (2025-11-15): [災難性BUG修復] 修正了日誌記錄中對 user_id 的錯誤引用。
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot: return
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.bot.user in message.mentions
        if not is_dm and not is_mentioned: return
        ctx = await self.bot.get_context(message)
        if ctx.valid: return
        user_id = str(message.author.id)
        user_input = message.content.replace(f'<@{self.bot.user.id}>', '').strip()
        if is_mentioned and not user_input:
            await message.channel.send(f"你好，{message.author.mention}！需要我做什麼嗎？")
            return
    
        ai_instance = await self.get_or_create_ai_instance(user_id)
        if not ai_instance:
            await message.channel.send("歡迎！請使用 `/start` 指令來開始或重置您的 AI 戀人。")
            return
            
        ai_instance.last_user_input = user_input

        async with message.channel.typing():
            try:
                logger.info(f"[{user_id}] 啟動「生成即摘要」對話流程...")
                input_data = { "user_input": user_input }
                final_response, summary_data = await ai_instance.preprocess_and_generate(input_data)
                
                if final_response and final_response.strip():
                    view = RegenerateView(cog=self)
                    for i in range(0, len(final_response), 2000):
                        current_view = view if i + 2000 >= len(final_response) else None
                        await message.channel.send(final_response[i:i+2000], view=current_view)
                    
                    logger.info(f"[{user_id}] 回應已發送。正在啟動事後處理任務...")
                    
                    # 任務1: 更新長期記憶 (來自摘要)
                    if summary_data.get("memory_summary"):
                        asyncio.create_task(ai_instance.update_memories_from_summary(summary_data))
                    
                    # [v58.0 核心修正] 任務2: 啟動獨立的、作為雙重保險的LORE提取流程
                    asyncio.create_task(ai_instance._background_lore_extraction(user_input, final_response))

                else:
                    logger.error(f"為使用者 {user_id} 的生成流程返回了空的或無效的回應。")
                    await message.channel.send("（抱歉，我好像突然斷線了...）")

            except Exception as e:
                logger.error(f"處理使用者 {user_id} 的「生成即摘要」流程時發生異常: {e}", exc_info=True)
                await message.channel.send(f"處理您的訊息時發生了一個嚴重的內部錯誤: `{type(e).__name__}`")
# 監聽並處理所有符合條件的訊息 函式結束


    
    
    # 函式：完成設定流程 (v51.0 - 持久化開場白)
    # 更新紀錄:
    # v51.2 (2025-09-25): [災難性BUG修復] 徹底重構創世流程，將 LORE 解析的調用邏輯從 ai_core.py 移至此處，並確保所有步驟都通過 await 嚴格同步執行，從根本上解決了開場白在聖經解析完成前生成的問題。
    # v51.1 (2025-09-25): [災難性BUG修復] 修正了 LORE 解析的同步調用問題。
    async def finalize_setup(self, interaction: discord.Interaction, canon_text: Optional[str] = None):
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) finalize_setup 總指揮流程啟動。Canon provided: {bool(canon_text)}")
        
        # 初始回應，告知使用者流程已開始，防止互動超時
        # 使用 followup 是因為初始的 interaction 可能已經被 defer 或回應過
        try:
            await interaction.followup.send("🚀 **正在為您執行最終創世...**\n這是一個耗時過程，可能需要數分鐘，請耐心等候最終的開場白。", ephemeral=True)
        except discord.errors.HTTPException:
            # 如果 followup 失敗，嘗試用原始互動回應
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message("🚀 **正在為您執行最終創世...**\n這是一個耗時過程，可能需要數分鐘，請耐心等候最終的開場白。", ephemeral=True)
            except discord.errors.HTTPException as e:
                logger.error(f"[{user_id}] 無法發送初始等待訊息: {e}")

        try:
            ai_instance = await self.get_or_create_ai_instance(user_id, is_setup_flow=True)
            if not ai_instance or not ai_instance.profile:
                logger.error(f"[{user_id}] 在 finalize_setup 中獲取 AI 核心失敗。")
                await interaction.user.send("❌ 錯誤：無法從資料庫加載您的基礎設定以進行創世。")
                self.setup_locks.discard(user_id)
                return

            # --- 步驟 1: 世界聖經處理 (如果提供) ---
            if canon_text:
                logger.info(f"[{user_id}] [/start 流程 1/4] 正在處理世界聖經...")
                await ai_instance.add_canon_to_vector_store(canon_text)
                logger.info(f"[{user_id}] [/start] 聖經文本已存入 RAG 資料庫。")
                
                logger.info(f"[{user_id}] [/start] 正在進行 LORE 智能解析 (此步驟將被嚴格等待)...")
                # 【核心修正】直接在此處 AWAIT LORE 解析函式
                await ai_instance.parse_and_create_lore_from_canon(canon_text=canon_text, is_setup_flow=True)
                logger.info(f"[{user_id}] [/start] LORE 智能解析【已同步完成】。")
            else:
                 logger.info(f"[{user_id}] [/start 流程 1/4] 跳過世界聖經處理。")
            
            # --- 步驟 2: 補完角色檔案 ---
            logger.info(f"[{user_id}] [/start 流程 2/4] 正在補完角色檔案...")
            await ai_instance.complete_character_profiles()
            
            # --- 步驟 3: 生成世界創世資訊 ---
            logger.info(f"[{user_id}] [/start 流程 3/4] 正在生成世界創世資訊...")
            await ai_instance.generate_world_genesis(canon_text=canon_text)
            
            # --- 步驟 4: 生成開場白 ---
            logger.info(f"[{user_id}] [/start 流程 4/4] 正在生成開場白...")
            opening_scene = await ai_instance.generate_opening_scene(canon_text=canon_text)
            logger.info(f"[{user_id}] [/start 流程 4/4] 開場白生成完畢。")

            # --- 最終步驟: 發送開場白並清理 ---
            scene_key = ai_instance._get_scene_key()
            await ai_instance._add_message_to_scene_history(scene_key, AIMessage(content=opening_scene))
            logger.info(f"[{user_id}] 開場白已成功存入場景 '{scene_key}' 的歷史記錄並持久化。")

            dm_channel = await interaction.user.create_dm()
            
            logger.info(f"[{user_id}] /start 流程：正在向使用者私訊發送開場白...")
            for i in range(0, len(opening_scene), 2000):
                await dm_channel.send(opening_scene[i:i+2000])
            logger.info(f"[{user_id}] /start 流程：開場白發送完畢。設定流程成功結束。")

        except Exception as e:
            logger.error(f"[{user_id}] 在手動編排的創世流程中發生嚴重錯誤: {e}", exc_info=True)
            try:
                await interaction.user.send(f"❌ **創世失敗**：在執行最終設定時發生了未預期的嚴重錯誤: `{e}`")
            except discord.errors.HTTPException as send_e:
                 logger.error(f"[{user_id}] 無法向使用者發送最終的錯誤訊息: {send_e}")
        finally:
            # 確保在流程結束時，無論成功或失敗，都釋放鎖
            self.setup_locks.discard(user_id)
            logger.info(f"[{user_id}] /start 流程鎖已釋放。")
# 完成設定流程 函式結束

    # 指令：[管理員] 瀏覽 LORE 詳細資料 (分頁)
    @app_commands.command(name="admin_browse_lores", description="[管理員] 分頁瀏覽指定使用者的 LORE 資料庫。")
    @app_commands.check(is_admin)
    @app_commands.describe(target_user="要瀏覽其 LORE 的目標使用者。", category="要瀏覽的 LORE 類別。")
    @app_commands.autocomplete(target_user=user_autocomplete)
    @app_commands.choices(category=LORE_CATEGORIES)
    async def admin_browse_lores(self, interaction: discord.Interaction, target_user: str, category: str):
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        all_lores = await lore_book.get_lores_by_category_and_filter(target_user, category)
        
        if not all_lores:
            await interaction.followup.send(f"❌ 在類別 `{category}` 中找不到使用者 `{target_user}` 的任何 LORE 條目。", ephemeral=True)
            return

        view = LorePaginatorView(lores=all_lores, user_id=target_user, category=category)
        embed = await view._create_embed()
        view.prev_page.disabled = True # 初始禁用上一頁
        if view.total_pages == 0:
             view.next_page.disabled = True # 如果只有一頁，也禁用下一頁

        await interaction.followup.send(embed=embed, view=view, ephemeral=True)
    # 指令：[管理員] 瀏覽 LORE 詳細資料 (分頁)
    

    # 函式：在背景處理世界聖經文本
    # 更新紀錄:
    # v3.1 (2025-09-23): [災難性BUG修復] 修正了 except 塊中，因嘗試訪問 `self.user_id`（一個不存在的屬性）而導致的 AttributeError。現在，它會正確地使用從 interaction 中獲取的局部變數 `user_id` 來記錄錯誤日誌。
    # v3.0 (2025-09-23): [架構簡化] 恢復了原始的成功訊息。
    async def _background_process_canon(self, interaction: discord.Interaction, content_text: str, is_setup_flow: bool):
        user_id = str(interaction.user.id)
        user = self.bot.get_user(interaction.user.id) or await self.bot.fetch_user(interaction.user.id)
        try:
            ai_instance = await self.get_or_create_ai_instance(user_id, is_setup_flow=is_setup_flow)
            if not ai_instance:
                await user.send("❌ **處理失敗！** 無法初始化您的 AI 核心，請嘗試重新 `/start`。")
                return
            if len(content_text) > 5000:
                await user.send("⏳ **請注意：**\n您提供的世界聖經內容較多，處理可能需要 **幾分鐘** 的時間，請耐心等候最終的「智能解析完成」訊息。")
            
            chunk_count = await ai_instance.add_canon_to_vector_store(content_text)
            
            if is_setup_flow:
                await interaction.followup.send("✅ 世界聖經已提交！正在為您啟動最終創世...", ephemeral=True)
                asyncio.create_task(self.finalize_setup(interaction, content_text))
                return

            await user.send(f"✅ **世界聖經已向量化！**\n內容已被分解為 **{chunk_count}** 個知識片段。\n\n🧠 AI 正在進行終極智能解析，將其轉化為結構化的 LORE 數據庫...")
            
            await ai_instance.parse_and_create_lore_from_canon(content_text)
            
            await user.send("✅ **智能解析完成！**\n您的世界聖經已成功轉化為 AI 的核心知識。您現在可以使用 `/admin_check_lore` (需管理員權限) 或其他方式來驗證 LORE 條目。")
        except Exception as e:
            # [v3.1 核心修正] 使用局部變數 user_id 而不是 self.user_id
            logger.error(f"[{user_id}] 背景處理世界聖經時發生錯誤: {e}", exc_info=True)
            await user.send(f"❌ **處理失敗！**\n發生了嚴重錯誤: `{type(e).__name__}`\n請檢查後台日誌以獲取詳細資訊。")
    # 函式：在背景處理世界聖經文本



    
    
# 函式：開始 /start 指令的重置流程 (v52.2 - 導入修正)
# 更新紀錄:
# v52.2 (2025-11-22): [災難性BUG修復] 修正了因缺少對 SceneHistoryData 模型的導入而導致的 NameError。
# v52.1 (2025-11-22): [災難性BUG修復] 修正了獲取使用者 ID 的方式，將錯誤的 `interaction.user_id` 改為正確的 `interaction.user.id`。
# v52.0 (2025-11-22): [重大架構升級] 在刪除用戶數據的流程中，增加了對 ai_instance._clear_scene_histories() 的顯式調用。
    async def start_reset_flow(self, interaction: discord.Interaction):
        user_id = str(interaction.user.id)
        try:
            logger.info(f"[{user_id}] 後台重置任務開始...")
            
            # 獲取一個臨時實例以執行清除操作
            ai_instance = await self.get_or_create_ai_instance(user_id, is_setup_flow=True)
            if not ai_instance:
                logger.warning(f"[{user_id}] 在重置流程中無法創建AI實例，將嘗試直接刪除資料庫數據。")
            
            # 關閉並移除記憶體中的實例
            if user_id in self.ai_instances:
                await self.ai_instances.pop(user_id).shutdown()
                gc.collect()
                await asyncio.sleep(1.5)

            # 在刪除其他數據之前，先清除短期記憶
            if ai_instance:
                await ai_instance._clear_scene_histories()

            # 刪除資料庫中的所有其他數據
            async with AsyncSessionLocal() as session:
                # 再次確保短期記憶被刪除（雙重保險）
                await session.execute(delete(SceneHistoryData).where(SceneHistoryData.user_id == user_id))
                await session.execute(delete(MemoryData).where(MemoryData.user_id == user_id))
                await session.execute(delete(Lore).where(Lore.user_id == user_id))
                await session.execute(delete(UserData).where(UserData.user_id == user_id))
                await session.commit()
            
            # 刪除向量數據庫文件
            vector_store_path = Path(f"./data/vector_stores/{user_id}")
            if vector_store_path.exists():
                await asyncio.to_thread(shutil.rmtree, vector_store_path)
            
            view = StartSetupView(cog=self)
            # 使用 followup 來回應已經被 defer/edit 過的互動
            await interaction.followup.send(
                content="✅ 重置完成！請點擊下方按鈕開始全新的設定流程。", 
                view=view, 
                ephemeral=True
            )
        except Exception as e:
            logger.error(f"[{user_id}] 後台重置任務失敗: {e}", exc_info=True)
            # 確保即使發生錯誤也能通知使用者
            if not interaction.response.is_done():
                try:
                    # 如果初始互動尚未回應，用它來發送錯誤
                    await interaction.response.send_message(f"執行重置時發生未知的嚴重錯誤: {e}", ephemeral=True)
                except discord.errors.InteractionResponded:
                    # 如果在我們檢查和發送之間，有其他東西回應了互動
                    await interaction.followup.send(f"執行重置時發生未知的嚴重錯誤: {e}", ephemeral=True)
            else:
                await interaction.followup.send(f"執行重置時發生未知的嚴重錯誤: {e}", ephemeral=True)
        finally:
            self.setup_locks.discard(user_id)
# 開始 /start 指令的重置流程 函式結束


    

# 指令：開始全新的冒險（重置所有資料） (v53.0 - 視圖生命週期修正)
# 更新紀錄:
# v53.0 (2025-11-22): [災難性BUG修復] 在發送 ConfirmStartView 後，增加了 `await view.wait()`。此修改會阻塞指令函式的退出，直到視圖被交互（按鈕點擊）或超時，從而確保了 View 物件在其生命週期內始終存活於記憶體中，徹底解決了按鈕點擊後無響應的問題。
# v52.0 (2025-11-22): [重大架構升級] 整合了顯式清除短期記憶的邏輯。
# v50.0 (2025-11-14): [完整性修復] 提供了此檔案的完整版本。
    @app_commands.command(name="start", description="開始全新的冒險（這將重置您所有的現有資料）")
    async def start(self, interaction: discord.Interaction):
        user_id = str(interaction.user.id)
        if not isinstance(interaction.channel, discord.DMChannel):
            await interaction.response.send_message("此指令只能在私訊頻道中使用。", ephemeral=True)
            return
        if user_id in self.setup_locks:
            view = ForceRestartView(cog=self)
            view.original_interaction_user_id = interaction.user.id
            await interaction.response.send_message("偵測到您有尚未完成的設定流程。您想要？", view=view, ephemeral=True)
            # 增加 view.wait() 以確保此視圖也能正常工作
            await view.wait()
            return
            
        view = ConfirmStartView(cog=self)
        view.original_interaction_user_id = interaction.user.id
        await interaction.response.send_message(
            "⚠️ **警告** ⚠️\n您確定要開始一段全新的冒險嗎？\n這將會**永久刪除**您當前所有的角色、世界、記憶和進度。", 
            view=view, 
            ephemeral=True
        )
        
        # [v53.0 核心修正] 等待視圖交互完成或超時
        # 這可以防止 view 物件在 start 函式結束後被垃圾回收，從而確保按鈕能夠被響應。
        await view.wait()
# 開始全新的冒險（重置所有資料） 指令結束

    # 指令：進入設定中心
    @app_commands.command(name="settings", description="進入設定中心，管理你的角色、AI戀人與世界觀")
    async def settings_command(self, interaction: discord.Interaction):
        if not isinstance(interaction.channel, discord.DMChannel):
            await interaction.response.send_message("此指令只能在私訊頻道中使用。", ephemeral=True)
            return
        ai_instance = await self.get_or_create_ai_instance(str(interaction.user.id))
        if ai_instance and ai_instance.profile:
            await interaction.response.send_message("請選擇您想修改的設定項目：", view=SettingsChoiceView(self), ephemeral=True)
        else:
            await interaction.response.send_message("請先使用 `/start` 指令進行初始設定。", ephemeral=True)
    # 指令：進入設定中心

    # 指令：客製化 AI 的回覆風格
    # 更新紀錄:
    # v1.1 (2025-09-24): [功能更新] 根據使用者要求，將預設的回應風格更新為“非常具體詳細描述，豐富對話互動”。
    @app_commands.command(name="response_style", description="客製化 AI 的回覆風格")
    async def response_style(self, interaction: discord.Interaction):
        if not isinstance(interaction.channel, discord.DMChannel):
            await interaction.response.send_message("此指令只能在私訊頻道中使用。", ephemeral=True)
            return
        ai_instance = await self.get_or_create_ai_instance(str(interaction.user.id))
        if not ai_instance or not ai_instance.profile:
            await interaction.response.send_message("請先使用 `/start` 指令進行初始設定。", ephemeral=True)
            return
        
        # [核心修正] 更新此處的預設值
        current_style = ai_instance.profile.response_style_prompt or "非常具體詳細描述，豐富對話互動"
        await interaction.response.send_modal(ResponseStyleModal(self, current_style))
    # 指令：客製化 AI 的回覆風格

    # 指令：編輯角色檔案
    @app_commands.command(name="edit_profile", description="編輯您或任何角色的個人檔案。")
    async def edit_profile(self, interaction: discord.Interaction):
        await interaction.response.send_message("請選擇您想編輯的角色檔案：", view=EditProfileRootView(self, interaction.user.id), ephemeral=True)
    # 指令：編輯角色檔案
        
    # 指令：通過貼上文字來設定世界聖經
    @app_commands.command(name="set_canon_text", description="通過貼上文字來設定您的世界聖經")
    async def set_canon_text(self, interaction: discord.Interaction):
        await interaction.response.send_modal(WorldCanonPasteModal(self, is_setup_flow=False))
    # 指令：通過貼上文字來設定世界聖經





    
# 指令：通過上傳檔案來設定世界聖經 (v54.0 - 超時修正)
# 更新紀錄:
# v54.1 (2025-09-24): [災難性BUG修復] 增加了對 setup_locks 的檢查，使指令能夠智能判斷當前是否處於 /start 創世流程中，從而正確觸發後續的創世步驟，解決了流程中斷的問題。
# v54.0 (2025-11-22): [災難性BUG修復] 徹底重構了此函式的回應流程以解決超時問題。
# v52.0 (2025-11-22): [架構調整] 創建此指令。
    @app_commands.command(name="set_canon_file", description="通過上傳 .txt 檔案來設定您的世界聖經")
    @app_commands.describe(file="請上傳一個 .txt 格式的檔案，最大 5MB。")
    async def set_canon_file(self, interaction: discord.Interaction, file: discord.Attachment):
        await interaction.response.defer(ephemeral=True, thinking=True)

        if not file.filename.lower().endswith('.txt'):
            await interaction.followup.send("❌ 檔案格式錯誤！請上傳一個 .txt 檔案。", ephemeral=True)
            return
        
        if file.size > 5 * 1024 * 1024: # 5MB
            await interaction.followup.send("❌ 檔案過大！請上傳小於 5MB 的檔案。", ephemeral=True)
            return

        try:
            content_bytes = await file.read()
            
            try:
                content_text = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    content_text = content_bytes.decode('gbk')
                except UnicodeDecodeError:
                    await interaction.followup.send("❌ 檔案編碼錯誤！請確保您的 .txt 檔案是 UTF-8 或 GBK 編碼。", ephemeral=True)
                    return

            # [v54.1 核心修正] 智能判斷當前是否處於 /start 設置流程中
            user_id = str(interaction.user.id)
            is_currently_in_setup = user_id in self.setup_locks
            
            if is_currently_in_setup:
                 # 如果在創世流程中，禁用按鈕並提示後續流程會自動開始
                if interaction.channel and interaction.message:
                     try:
                        original_message = await interaction.channel.fetch_message(interaction.message.id)
                        view = discord.ui.View.from_message(original_message)
                        for item in view.children: item.disabled = True
                        await original_message.edit(view=view)
                     except (discord.errors.NotFound, AttributeError): pass
            
            await interaction.followup.send("✅ 檔案已接收！正在後台為您進行向量化和智能解析，這可能需要幾分鐘時間，請稍候...", ephemeral=True)
            
            # 將動態判斷的 is_setup_flow 值傳遞給背景任務
            asyncio.create_task(self._background_process_canon(interaction, content_text, is_setup_flow=is_currently_in_setup))
        except Exception as e:
            logger.error(f"處理上傳的世界聖經檔案時發生錯誤: {e}", exc_info=True)
            await interaction.followup.send(f"讀取或處理檔案時發生嚴重錯誤: `{type(e).__name__}`", ephemeral=True)
# 通過上傳檔案來設定世界聖經 指令結束



    

    # 指令：[管理員] 設定好感度
    @app_commands.command(name="admin_set_affinity", description="[管理員] 設定指定使用者的好感度")
    @app_commands.check(is_admin)
    @app_commands.autocomplete(target_user=user_autocomplete)
    async def admin_set_affinity(self, interaction: discord.Interaction, target_user: str, value: app_commands.Range[int, -1000, 1000]):
        async with AsyncSessionLocal() as session:
            user_data = await session.get(UserData, target_user)
            if user_data:
                user_data.affinity = value
                await session.commit()
                if target_user in self.ai_instances and self.ai_instances[target_user].profile: self.ai_instances[target_user].profile.affinity = value
                await interaction.response.send_message(f"已將使用者 {user_data.username} 的好感度設定為 {value}。", ephemeral=True)
            else: await interaction.response.send_message(f"錯誤：找不到使用者 {target_user}。", ephemeral=True)
    # 指令：[管理員] 設定好感度

    # 指令：[管理員] 重置使用者資料
    @app_commands.command(name="admin_reset", description="[管理員] 清除指定使用者的所有資料")
    @app_commands.check(is_admin)
    @app_commands.autocomplete(target_user=user_autocomplete)
    async def admin_reset(self, interaction: discord.Interaction, target_user: str):
        await interaction.response.defer(ephemeral=True, thinking=True)
        if target_user in self.ai_instances: await self.ai_instances.pop(target_user).shutdown()
        async with AsyncSessionLocal() as session:
            await session.execute(delete(MemoryData).where(MemoryData.user_id == target_user))
            await session.execute(delete(Lore).where(Lore.user_id == target_user))
            await session.execute(delete(UserData).where(UserData.user_id == target_user))
            await session.commit()
        try:
            vector_store_path = Path(f"./data/vector_stores/{target_user}")
            if vector_store_path.exists(): await asyncio.to_thread(shutil.rmtree, vector_store_path)
        except Exception as e: logger.error(f"管理員重置使用者 {target_user} 時刪除向量目錄失敗: {e}", exc_info=True)
        await interaction.followup.send(f"已成功重置使用者 {target_user} 的所有資料。", ephemeral=True)
    # 指令：[管理員] 重置使用者資料

    # 指令：[管理員] 強制更新程式碼
    @app_commands.command(name="admin_force_update", description="[管理員] 強制從 GitHub 同步最新程式碼並重啟機器人。")
    @app_commands.check(is_admin)
    async def admin_force_update(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        await interaction.followup.send("✅ **指令已接收！**\n正在背景中為您執行強制同步與重啟...", ephemeral=True)
        asyncio.create_task(self._perform_update_and_restart(interaction))
    # 指令：[管理員] 強制更新程式碼
    
    # 函式：執行更新與重啟的背景任務
    async def _perform_update_and_restart(self, interaction: discord.Interaction):
        try:
            await asyncio.sleep(1)
            success, msg = await self._run_git_command(["git", "reset", "--hard", "origin/main"])
            if success:
                if settings.ADMIN_USER_ID:
                    try:
                        admin_user = self.bot.get_user(int(settings.ADMIN_USER_ID)) or await self.bot.fetch_user(int(settings.ADMIN_USER_ID))
                        await admin_user.send("✅ **系統更新成功！** 機器人即將重啟。")
                    except Exception as e: logger.error(f"發送更新成功通知給管理員時發生未知錯誤: {e}", exc_info=True)
                print("🔄 [Admin Update] Git 同步成功，觸發程式退出以進行重啟...")
                if self.bot.shutdown_event: self.bot.shutdown_event.set()
            else:
                await interaction.followup.send(f"🔥 **同步失敗！**\n```\n{msg}\n```", ephemeral=True)
        except Exception as e: 
            logger.error(f"背景任務：執行強制更新時發生未預期錯誤: {e}", exc_info=True)
            if interaction:
                try: await interaction.followup.send(f"🔥 **更新時發生錯誤！**\n`{type(e).__name__}: {e}`", ephemeral=True)
                except discord.errors.NotFound: pass
    # 函式：執行更新與重啟的背景任務

    # 指令：[管理員] 切換直連 LLM 模式
    @app_commands.command(name="admin_direct_mode", description="[管理員] 為指定使用者開啟或關閉直連 LLM 測試模式。")
    @app_commands.check(is_admin)
    @app_commands.autocomplete(target_user=user_autocomplete)
    @app_commands.describe(target_user="要修改模式的目標使用者。", mode="選擇要開啟還是關閉直連模式。")
    async def admin_direct_mode(self, interaction: discord.Interaction, target_user: str, mode: Literal['on', 'off']):
        await interaction.response.defer(ephemeral=True, thinking=True)
        ai_instance = await self.get_or_create_ai_instance(target_user)
        if not ai_instance or not ai_instance.profile:
            await interaction.followup.send(f"❌ 錯誤：找不到使用者 {target_user} 的資料，或其資料未初始化。", ephemeral=True)
            return
        try:
            new_state = True if mode == 'on' else False
            ai_instance.profile.game_state.direct_mode_enabled = new_state
            if await ai_instance.update_and_persist_profile({'game_state': ai_instance.profile.game_state.model_dump()}):
                discord_user = self.bot.get_user(int(target_user)) or await self.bot.fetch_user(int(target_user))
                status_text = "🟢 開啟" if new_state else "🔴 關閉"
                await interaction.followup.send(f"✅ 成功！已為使用者 **{discord_user.name}** (`{target_user}`) 將直連 LLM 模式設定為 **{status_text}**。", ephemeral=True)
            else:
                await interaction.followup.send(f"❌ 錯誤：更新使用者 {target_user} 的設定檔失敗。", ephemeral=True)
        except Exception as e:
            logger.error(f"為使用者 {target_user} 切換直連模式時發生錯誤: {e}", exc_info=True)
            await interaction.followup.send(f"❌ 處理您的請求時發生了未預期的錯誤: {type(e).__name__}", ephemeral=True)
    # 指令：[管理員] 切換直連 LLM 模式

    # 指令：[管理員] 查詢使用者狀態
    @app_commands.command(name="admin_check_status", description="[管理員] 查詢指定使用者的當前狀態")
    @app_commands.check(is_admin)
    @app_commands.autocomplete(target_user=user_autocomplete)
    async def admin_check_status(self, interaction: discord.Interaction, target_user: str):
        discord_user = self.bot.get_user(int(target_user))
        async with AsyncSessionLocal() as session:
            user_data = await session.get(UserData, target_user)
            if user_data:
                game_state = GameState.model_validate(user_data.game_state or {})
                embed = Embed(title=f"📊 使用者狀態查詢: {user_data.username}", color=discord.Color.blue())
                if discord_user: embed.set_thumbnail(url=discord_user.display_avatar.url)
                embed.add_field(name="❤️ AI 好感度", value=f"**{user_data.affinity}**", inline=True)
                embed.add_field(name="💰 金錢", value=str(game_state.money), inline=True)
                embed.add_field(name="📍 當前地點", value=' > '.join(game_state.location_path), inline=False)
                await interaction.response.send_message(embed=embed, ephemeral=True)
            else: await interaction.response.send_message(f"錯誤：找不到使用者 {target_user}。", ephemeral=True)
    # 指令：[管理員] 查詢使用者狀態
            
    # 指令：[管理員] 查詢 Lore 詳細資料
    @app_commands.command(name="admin_check_lore", description="[管理員] 查詢指定使用者的 Lore 詳細資料")
    @app_commands.check(is_admin)
    @app_commands.describe(target_user="...", category="...", key="...")
    @app_commands.autocomplete(target_user=user_autocomplete, key=lore_key_autocomplete)
    @app_commands.choices(category=LORE_CATEGORIES)
    async def admin_check_lore(self, interaction: discord.Interaction, target_user: str, category: str, key: str):
        lore_entry = await lore_book.get_lore(target_user, category, key)
        if lore_entry:
            content_str = json.dumps(lore_entry.content, ensure_ascii=False, indent=2)
            embed = Embed(title=f"📜 Lore 查詢: {key.split(' > ')[-1]}", color=discord.Color.green())
            embed.add_field(name="詳細資料", value=f"```json\n{content_str[:1000]}\n```", inline=False)
            await interaction.response.send_message(embed=embed, ephemeral=True)
        else: await interaction.response.send_message(f"錯誤：找不到 Lore。", ephemeral=True)
    # 指令：[管理員] 查詢 Lore 詳細資料
        
    # 指令：[管理員] 推送日誌
    @app_commands.command(name="admin_push_log", description="[管理員] 強制將最新的100條LOG推送到GitHub倉庫。")
    @app_commands.check(is_admin)
    async def admin_push_log(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.push_log_to_github_repo(interaction)
    # 指令：[管理員] 推送日誌
        
    # 函式：將日誌推送到 GitHub 倉庫
    async def push_log_to_github_repo(self, interaction: Optional[discord.Interaction] = None):
        try:
            log_file_path = PROJ_DIR / "data" / "logs" / "app.log"
            if not log_file_path.is_file():
                if interaction: await interaction.followup.send("❌ **推送失敗**：找不到日誌檔案。", ephemeral=True)
                return
            with open(log_file_path, 'r', encoding='utf-8') as f: latest_lines = f.readlines()[-100:]
            upload_log_path = PROJ_DIR / "latest_log.txt"
            with open(upload_log_path, 'w', encoding='utf-8') as f: f.write(f"### AI Lover Log - {datetime.datetime.now().isoformat()} ###\n\n" + "".join(latest_lines))
            
            await self._run_git_command(["git", "add", str(upload_log_path)])
            commit_message = f"docs: Update latest_log.txt at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            await self._run_git_command_unlocked(["git", "commit", "-m", commit_message])
            await self._run_git_command(["git", "push", "origin", "main"])

            if interaction: await interaction.followup.send(f"✅ **LOG 推送成功！**", ephemeral=True)
        except Exception as e:
            if interaction: await interaction.followup.send(f"❌ **推送失敗**：`{e}`", ephemeral=True)
    # 函式：將日誌推送到 GitHub 倉庫

    # 函式：無鎖執行 Git 命令的輔助函式
    async def _run_git_command_unlocked(self, command: list):
         async with self.git_lock:
            await asyncio.to_thread(subprocess.run, command, check=False, cwd=PROJ_DIR, capture_output=True)
    # 函式：無鎖執行 Git 命令的輔助函式

    # 指令：[管理員] 版本控制
    @app_commands.command(name="admin_version_control", description="[管理員] 打開圖形化版本控制面板。")
    @app_commands.check(is_admin)
    async def admin_version_control(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        view = VersionControlView(cog=self, original_user_id=interaction.user.id)
        embed = await view._build_embed()
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)
    # 指令：[管理員] 版本控制
        
    # 函式：全域應用程式指令錯誤處理器
    @commands.Cog.listener()
    async def on_app_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.CheckFailure):
            await interaction.response.send_message("你沒有權限使用此指令。", ephemeral=True)
        else:
            logger.error(f"一個應用程式指令發生錯誤: {error}", exc_info=True)
            if not interaction.response.is_done():
                await interaction.response.send_message(f"發生未知錯誤。", ephemeral=True)
    # 函式：全域應用程式指令錯誤處理器
# 類別：機器人核心功能集 (Cog)

# 類別：AI 戀人機器人主體
class AILoverBot(commands.Bot):
    # 函式：初始化 AILoverBot
    def __init__(self, shutdown_event: asyncio.Event, git_lock: asyncio.Lock):
        super().__init__(command_prefix='/', intents=intents, activity=discord.Game(name="與你共度時光"))
        self.shutdown_event = shutdown_event
        self.git_lock = git_lock
        self.is_ready_once = False
    # 函式：初始化 AILoverBot
    
# 函式：Discord 機器人設置鉤子 (v52.0 - 移除錯誤的持久化視圖)
# 更新紀錄:
# v52.0 (2025-11-22): [災難性BUG修復] 移除了對 ConfirmStartView 的全局註冊。ConfirmStartView 是一個有狀態的臨時視圖，不應被持久化，錯誤的註冊導致了其 interaction_check 永遠失敗。
# v51.3 (2025-11-17): [功能擴展] 實現了伺服器特定指令同步。
# v51.2 (2025-11-17): [健壯性強化] 為指令同步 (tree.sync) 增加了詳細的日誌記錄和 try...except 錯誤處理。
    async def setup_hook(self):
        cog = BotCog(self, self.git_lock)
        await self.add_cog(cog)

        cog.connection_watcher.start()
        
        # [v52.0 核心修正] 只註冊真正無狀態的持久化視圖
        self.add_view(StartSetupView(cog=cog))
        self.add_view(ContinueToUserSetupView(cog=cog))
        self.add_view(ContinueToAiSetupView(cog=cog))
        self.add_view(ContinueToCanonSetupView(cog=cog))
        self.add_view(RegenerateView(cog=cog))
        logger.info("所有持久化 UI 視圖已成功註冊。")

        try:
            if settings.TEST_GUILD_ID:
                guild = discord.Object(id=int(settings.TEST_GUILD_ID))
                self.tree.copy_global_to(guild=guild)
                logger.info(f"正在嘗試將應用程式指令同步到指定的測試伺服器 (ID: {settings.TEST_GUILD_ID})...")
                await self.tree.sync(guild=guild)
                logger.info(f"✅ 應用程式指令已成功同步到測試伺服器！")
            else:
                logger.info("正在嘗試將應用程式指令 (Slash Commands) 全域同步到 Discord...")
                await self.tree.sync()
                logger.info("✅ 應用程式指令全域同步成功！(注意：全域指令更新可能需要長達一小時才能在所有伺服器生效)")
        except Exception as e:
            logger.error(f"🔥 應用程式指令同步失敗: {e}", exc_info=True)
        
        logger.info("Discord Bot is ready!")
# Discord 機器人設置鉤子 函式結束
    
    # 函式：機器人準備就緒時的事件處理器
    async def on_ready(self):
        logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
        if not self.is_ready_once:
            self.is_ready_once = True
            if settings.ADMIN_USER_ID:
                try:
                    admin_user = self.get_user(int(settings.ADMIN_USER_ID)) or await self.bot.fetch_user(int(settings.ADMIN_USER_ID))
                    await admin_user.send(f"✅ **系統啟動成功！**")
                    logger.info(f"已成功發送啟動成功通知給管理員。")
                except Exception as e:
                    logger.error(f"發送啟動成功通知給管理員時發生未知錯誤: {e}", exc_info=True)
    # 函式：機器人準備就緒時的事件處理器
# 類別：AI 戀人機器人主體
