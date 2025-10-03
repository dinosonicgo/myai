# src/discord_bot.py 的中文註釋(v62.0 - 新增管理員調試工具)
# 更新紀錄:
# v62.0 (2025-10-02): [功能擴展] 新增了兩個核心的管理員調試指令：1. `/admin_rag_peek`，允許管理員直接查看 RAG 系統對任意查詢返回的原始文檔，以診斷檢索問題。2. `/admin_direct_chat`，實現了一個繞過所有複雜圖形邏輯的 RAG 直通 LLM 管道，用於在最純淨的環境下測試核心 Prompt 和 RAG 上下文的協同效果。
# v61.0 (2025-10-02): [災難性BUG修復] 根據「串行化」原則，再次徹底重構了創世流程 `_perform_full_setup_flow`。
# v60.0 (2025-10-02): [災難性BUG修復] 再次重構並最終確定了創世流程的執行時序。

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
import time
from pathlib import Path

from .logger import logger
from .ai_core import AILover, GENERATION_MODEL_PRIORITY # [v62.0 核心修正] 導入常量
from . import lore_book
from .lore_book import Lore
from .database import AsyncSessionLocal, UserData, MemoryData, init_db, SceneHistoryData
from .schemas import CharacterProfile, LocationInfo, WorldGenesisResult
from .models import UserProfile, GameState
from src.config import settings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.generativeai.types import BlockedPromptException

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
    target_user_id = str(interaction.namespace.target_user) if hasattr(interaction.namespace, 'target_user') else str(interaction.user.id)
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
        if user_id in self.cog.active_setups:
            await interaction.response.send_message("⏳ 您已經有一個創世流程正在後台執行，請耐心等候。", ephemeral=True)
            return
        
        modal = WorldCanonPasteModal(self.cog, original_interaction_message_id=interaction.message.id)
        await interaction.response.send_modal(modal)
    # 函式：處理「貼上世界聖經」按鈕點擊事件

    # 處理「上傳世界聖經」按鈕點擊事件
    @discord.ui.button(label="📄 上傳世界聖經 (.txt)", style=discord.ButtonStyle.success, custom_id="persistent_upload_canon")
    async def upload_canon(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        if user_id in self.cog.active_setups:
            await interaction.response.send_message("⏳ 您已經有一個創世流程正在後台執行，請耐心等候。", ephemeral=True)
            return

        self.cog.active_setups.add(user_id)
        logger.info(f"[{user_id}] [創世流程] 檔案上傳開始，已設定 active_setups 狀態鎖。")

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
            
            if attachment.size > 5 * 1024 * 1024: # 5MB
                await interaction.followup.send("❌ 檔案過大！請重新開始 `/start` 流程。", ephemeral=True)
                self.cog.active_setups.discard(user_id)
                return

            await interaction.followup.send("✅ 檔案已接收！創世流程已在後台啟動，完成後您將收到開場白。這可能需要數分鐘，請耐心等候。", ephemeral=True)
            
            content_bytes = await attachment.read()
            content_text = content_bytes.decode('utf-8', errors='ignore')
            
            asyncio.create_task(self.cog._perform_full_setup_flow(user=interaction.user, canon_text=content_text))
            
        except asyncio.TimeoutError:
            await interaction.followup.send("⏳ 操作已超時。請重新開始 `/start` 流程。", ephemeral=True)
            self.cog.active_setups.discard(user_id)
        except Exception as e:
            logger.error(f"[{user_id}] 在等待檔案上傳時發生錯誤: {e}", exc_info=True)
            await interaction.followup.send(f"處理您的檔案時發生錯誤: `{e}`。請重新開始 `/start` 流程。", ephemeral=True)
            self.cog.active_setups.discard(user_id)
        finally:
            self.stop()
    # 處理「上傳世界聖經」按鈕點擊事件

    # 函式：處理「完成設定」按鈕點擊事件
    @discord.ui.button(label="✅ 完成設定並開始冒險（跳過聖經)", style=discord.ButtonStyle.primary, custom_id="persistent_finalize_setup")
    async def finalize(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        if user_id in self.cog.active_setups:
            await interaction.response.send_message("⏳ 您已經有一個創世流程正在後台執行，請耐心等候。", ephemeral=True)
            return
            
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(view=self)
        
        await interaction.followup.send("✅ 基礎設定完成！創世流程已在後台啟動，完成後您將收到開場白。這可能需要幾分鐘，請耐心等候。", ephemeral=True)
        
        self.cog.active_setups.add(user_id)
        asyncio.create_task(self.cog._perform_full_setup_flow(user=interaction.user, canon_text=None))
        self.stop()
    # 函式：處理「完成設定」按鈕點擊事件
# 類別：繼續到世界聖經設定的視圖

# 類別：重新生成或撤銷回覆的視圖
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
                    history.messages.pop() 
                    history.messages.pop()
                    logger.info(f"[{user_id}] [重新生成] 已從場景 '{scene_key}' 的短期記憶中撤銷上一回合。")

            await ai_instance._delete_last_memory()
            await interaction.message.delete()

            logger.info(f"[{user_id}] [重新生成] 正在使用上次輸入重新生成回應...")
            input_data = {"user_input": ai_instance.last_user_input}
            
            final_response = await ai_instance.preprocess_and_generate(input_data)

            if final_response and final_response.strip():
                view = self
                for i in range(0, len(final_response), 2000):
                    current_view = view if i + 2000 >= len(final_response) else None
                    await interaction.channel.send(final_response[i:i+2000], view=current_view)
                
                logger.info(f"[{user_id}] [重新生成] 新回應已發送，正在啟動統一的「事後分析」任務...")
                if ai_instance.last_context_snapshot:
                    asyncio.create_task(ai_instance._background_lore_extraction(ai_instance.last_context_snapshot))
                else:
                    logger.error(f"[{user_id}] 災難性錯誤：重新生成後未能創建上下文快照，事後分析無法啟動！")
            else:
                await interaction.followup.send("（抱歉，我重新思考了一下，但腦海還是一片空白...）", ephemeral=True)

        except Exception as e:
            logger.error(f"[{user_id}] [重新生成] 流程執行時發生異常: {e}", exc_info=True)
            await interaction.followup.send(f"重新生成時發生了一個嚴重的內部錯誤: `{type(e).__name__}`", ephemeral=True)
    # 函式：處理「重新生成」按鈕點擊事件
    
    # 函式：處理「撤銷」按鈕點擊事件
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

            history.messages.pop()
            last_user_message = history.messages.pop()
            logger.info(f"[{user_id}] [撤銷] 已成功從場景 '{scene_key}' 的短期記憶中撤銷上一回合。")

            await ai_instance._delete_last_memory()

            await interaction.message.delete()
            
            if not isinstance(interaction.channel, discord.DMChannel):
                try:
                    async for msg in interaction.channel.history(limit=10):
                        if msg.author.id == interaction.user.id and msg.content == last_user_message.content:
                            await msg.delete()
                            logger.info(f"[{user_id}] [撤銷] 已成功刪除使用者在伺服器頻道中的上一條指令訊息。")
                            break
                except (discord.errors.Forbidden, discord.errors.NotFound) as e:
                    logger.warning(f"[{user_id}] [撤銷] 刪除使用者訊息時發生非致命錯誤（可能權限不足）: {e}")
            else:
                logger.info(f"[{user_id}] [撤銷] 處於DM頻道，跳過刪除使用者訊息的步驟。")
            
            ai_instance.last_user_input = None

            await interaction.followup.send("✅ 上一回合已成功深度撤銷（包含長期記憶）。", ephemeral=True)

        except Exception as e:
            logger.error(f"[{user_id}] [撤銷] 流程執行時發生異常: {e}", exc_info=True)
            await interaction.followup.send(f"撤銷時發生了一個嚴重的內部錯誤: `{type(e).__name__}`", ephemeral=True)
    # 函式：處理「撤銷」按鈕點擊事件
# 類別：重新生成或撤銷回覆的視圖

# 類別：貼上世界聖經的 Modal
class WorldCanonPasteModal(discord.ui.Modal, title="貼上您的世界聖經文本"):
    canon_text = discord.ui.TextInput(label="請將您的世界觀/角色背景故事貼於此處", style=discord.TextStyle.paragraph, placeholder="在此貼上您的 .txt 檔案內容或直接編寫...", required=True, max_length=4000)
    
    # 函式：初始化 WorldCanonPasteModal
    def __init__(self, cog: "BotCog", original_interaction_message_id: int = None):
        super().__init__(timeout=600.0)
        self.cog = cog
        self.original_interaction_message_id = original_interaction_message_id
    # 函式：初始化 WorldCanonPasteModal
    
    # 函式：處理 Modal 提交事件
    async def on_submit(self, interaction: discord.Interaction):
        user_id = str(interaction.user.id)
        if user_id in self.cog.active_setups:
            await interaction.response.send_message("⏳ 您已經有一個創世流程正在後台執行，請耐心等候。", ephemeral=True)
            return

        if self.original_interaction_message_id:
            try:
                original_message = await interaction.channel.fetch_message(self.original_interaction_message_id)
                view = discord.ui.View.from_message(original_message)
                for item in view.children: item.disabled = True
                await original_message.edit(view=view)
            except (discord.errors.NotFound, AttributeError): pass
        
        await interaction.response.send_message("✅ 文字已接收！創世流程已在後台啟動，完成後您將收到開場白。這可能需要數分鐘，請耐心等候。", ephemeral=True)
        
        self.cog.active_setups.add(user_id)
        asyncio.create_task(self.cog._perform_full_setup_flow(user=interaction.user, canon_text=self.canon_text.value))
    # 函式：處理 Modal 提交事件
# 類別：貼上世界聖經的 Modal

# 類別：LORE 瀏覽器分頁視圖
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
# 類別：LORE 瀏覽器分頁視圖

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
        
        self.gender = discord.ui.TextInput(
            label="性別 (必填)", 
            default=profile_data.get('gender', ''), 
            placeholder="請輸入 男 / 女 / 其他"
        )
        
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
            if self.is_setup_flow: self.cog.active_setups.discard(user_id)
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
            if self.is_setup_flow: self.cog.active_setups.discard(user_id)
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

# 類別：確認 /start 重置的視圖
class ConfirmStartView(discord.ui.View):
    # 函式：初始化 ConfirmStartView
    def __init__(self, *, cog: "BotCog"):
        super().__init__(timeout=180.0)
        self.cog = cog
        self.original_interaction_user_id = None
    # 函式：初始化 ConfirmStartView
        
    # 函式：檢查互動是否來自原始使用者
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.original_interaction_user_id:
            await interaction.response.send_message("你無法操作不屬於你的指令。", ephemeral=True)
            return False
        return True
    # 函式：檢查互動是否來自原始使用者
        
    # 函式：處理「確認重置」按鈕點擊事件
    @discord.ui.button(label="【確認重置並開始】", style=discord.ButtonStyle.danger)
    async def confirm_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(content="⏳ 正在為您重置所有資料，此過程可能需要一點時間，請稍候...", view=self)
        
        await self.cog.start_reset_flow(interaction)
        self.stop()
    # 函式：處理「確認重置」按鈕點擊事件
        
    # 函式：處理「取消」按鈕點擊事件
    @discord.ui.button(label="取消", style=discord.ButtonStyle.secondary)
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="操作已取消。", view=None)
        self.stop()
    # 函式：處理「取消」按鈕點擊事件
        
    # 函式：處理視圖超時事件
    async def on_timeout(self):
        for item in self.children:
            item.disabled = True
        try:
            if self.message:
                await self.message.edit(content="操作已超時，請重新發起指令。", view=self)
        except discord.HTTPException:
            pass
    # 函式：處理視圖超時事件
# 類別：確認 /start 重置的視圖

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
            rewriting_chain = ai_instance.get_profile_rewriting_prompt()
            
            # Manually format the prompt
            full_prompt = rewriting_chain.format(
                original_description=self.original_description,
                edit_instruction=self.edit_instruction.value
            )
            
            new_description = await ai_instance.ainvoke_with_rotation(full_prompt)
            
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
        view = ConfirmAndEditView(cog=self.cog, target_type='npc', target_key=selected_key, display_name=profile.name, original_description=profile.description or "")

        content_str = json.dumps(lore.content, ensure_ascii=False, indent=2)

        if len(content_str) > 1000:
            try:
                temp_dir = PROJ_DIR / "temp"
                temp_dir.mkdir(exist_ok=True)
                file_path = temp_dir / f"npc_{interaction.user.id}_{int(time.time())}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content_str)
                
                file_name = f"{profile.name}.json"
                await interaction.followup.send(
                    f"這是 NPC **{profile.name}** 的當前檔案（由於內容過長，已作為檔案附件發送）。請預覽後點擊按鈕進行修改：",
                    file=discord.File(file_path, filename=file_name),
                    view=view,
                    ephemeral=True
                )
                os.remove(file_path)
            except Exception as e:
                logger.error(f"[{interaction.user.id}] 創建或發送 NPC 檔案時出錯: {e}", exc_info=True)
                await interaction.followup.send("錯誤：創建 NPC 檔案時發生問題。", ephemeral=True)
        else:
            embed = _create_profile_embed(profile, "👥 NPC 檔案")
            await interaction.followup.send("這是您選擇角色的當前檔案，請預覽後點擊按鈕進行修改：", embed=embed, view=view, ephemeral=True)

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
        
        content_str = json.dumps(profile.model_dump(), ensure_ascii=False, indent=2)
        view = ConfirmAndEditView(cog=self.cog, target_type=target_type, target_key=profile.name, display_name=profile.name, original_description=profile.description or "")

        if len(content_str) > 1000:
            try:
                temp_dir = PROJ_DIR / "temp"
                temp_dir.mkdir(exist_ok=True)
                file_path = temp_dir / f"profile_{interaction.user.id}_{int(time.time())}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content_str)
                
                file_name = f"{profile.name}.json"
                await interaction.followup.send(
                    f"這是您選擇角色的當前檔案（由於內容過長，已作為檔案附件發送）。請預覽後點擊按鈕進行修改：",
                    file=discord.File(file_path, filename=file_name),
                    view=view,
                    ephemeral=True
                )
                os.remove(file_path)
            except Exception as e:
                logger.error(f"[{user_id}] 創建或發送個人檔案時出錯: {e}", exc_info=True)
                await interaction.followup.send("錯誤：創建個人檔案時發生問題。", ephemeral=True)
        else:
            embed = _create_profile_embed(profile, title_prefix)
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
    def __init__(self, bot: "AILoverBot", git_lock: asyncio.Lock, is_ollama_available: bool):
        self.bot = bot
        self.ai_instances: dict[str, AILover] = {}
        self.active_setups: set[str] = set()
        self.git_lock = git_lock
        self.is_ollama_available = is_ollama_available
    # 函式：初始化 BotCog

    # 函式：Cog 卸載時執行的清理
    def cog_unload(self):
        self.connection_watcher.cancel()
    # 函式：Cog 卸載時執行的清理

# 函式：執行完整的後台創世流程 (v63.0 - 廢棄創世地點生成)
# 更新紀錄:
# v63.0 (2025-10-03): [架構簡化] 根據「RAG驅動的智能開場導演」策略，徹底移除了流程中對 `generate_world_genesis` 的調用。初始地點的選擇職責現在已完全轉移給了 `generate_opening_scene` 函式內部的 RAG 查詢，使得開場流程更智能、更精簡，並從根源上避免了隨機地點與世界觀的衝突。
# v62.0 (2025-10-02): [根本性重構] 根據“純向量RAG優先”的最終策略，徹底簡化了創世流程。
# v61.0 (2025-10-02): [災難性BUG修復] 根據「串行化」原則，再次徹底重構了創世流程。
    async def _perform_full_setup_flow(self, user: discord.User, canon_text: Optional[str] = None):
        """(v63.0) 一個獨立的、以純向量RAG為核心的簡化版後台創世流程。"""
        user_id = str(user.id)
        try:
            logger.info(f"[{user_id}] [創世流程 v63.0] 純向量RAG簡化版流程已啟動。")
            
            ai_instance = await self.get_or_create_ai_instance(user_id, is_setup_flow=True)
            if not ai_instance or not ai_instance.profile:
                await user.send("❌ 錯誤：無法初始化您的 AI 核心以進行創世。")
                return

            # --- 步驟 1: 補完角色檔案 ---
            logger.info(f"[{user_id}] [後台創世 1/4] 正在補完角色檔案...")
            await ai_instance.complete_character_profiles()
            
            # --- 步驟 2: 準備用於RAG构建的文档 ---
            docs_for_rag = []
            if canon_text and canon_text.strip():
                logger.info(f"[{user_id}] [後台創世 2/4] 正在將世界聖經原文分割成文檔...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
                docs_for_rag = text_splitter.create_documents([canon_text], metadatas=[{"source": "canon"} for _ in [canon_text]])
                logger.info(f"[{user_id}] [後台創世 2/4] 世界聖經已分割為 {len(docs_for_rag)} 個文檔。")
            else:
                logger.info(f"[{user_id}] [後台創世 2/4] 未提供世界聖經，將創建一個空的 RAG 索引。")

            # --- 步驟 3: RAG 索引纯向量创始构建 ---
            logger.info(f"[{user_id}] [後台創世 3/4] 正在觸發 RAG 索引純向量創始構建...")
            await ai_instance._load_or_build_rag_retriever(docs_to_build=docs_for_rag if docs_for_rag else [])
            logger.info(f"[{user_id}] [後台創世 3/4] 純向量 RAG 索引構建完成。")
            
            # --- 步驟 4: 生成開場白 (RAG驅動) ---
            logger.info(f"[{user_id}] [後台創世 4/4] 正在生成 RAG 驅動的智能開場白...")
            # [v63.0 核心修正] 移除對 generate_world_genesis 的調用
            opening_scene = await ai_instance.generate_opening_scene(canon_text=canon_text)
            logger.info(f"[{user_id}] [後台創世 4/4] 開場白生成完畢。")

            # --- 最終步驟: 發送開場白並清理 ---
            # 開場白本身可能暗示了地點，我們需要更新遊戲狀態
            # 為了簡化，我們暫時將初始地點設定為一個通用值，後續對話會修正它
            gs = ai_instance.profile.game_state
            gs.location_path = ["故事的開端"]
            await ai_instance.update_and_persist_profile({'game_state': gs.model_dump()})

            scene_key = ai_instance._get_scene_key()
            await ai_instance._add_message_to_scene_history(scene_key, AIMessage(content=opening_scene))
            
            logger.info(f"[{user_id}] [後台創世] 正在向使用者私訊發送最終開場白...")
            for i in range(0, len(opening_scene), 2000):
                await user.send(opening_scene[i:i+2000])
            logger.info(f"[{user_id}] [後台創世] 開場白發送完畢。")

        except Exception as e:
            logger.error(f"[{user_id}] 後台創世流程發生嚴重錯誤: {e}", exc_info=True)
            try:
                await user.send(f"❌ **創世失敗**：在後台執行時發生了未預期的嚴重錯誤: `{e}`")
            except discord.errors.HTTPException as send_e:
                 logger.error(f"[{user_id}] 無法向使用者發送最終的錯誤訊息: {send_e}")
        finally:
            self.active_setups.discard(user_id)
            logger.info(f"[{user_id}] 後台創世流程結束，狀態鎖已釋放。")
# 函式：執行完整的後台創世流程 (v63.0 - 廢棄創世地點生成)



    

    # 函式：獲取或創建使用者的 AI 實例
    async def get_or_create_ai_instance(self, user_id: str, is_setup_flow: bool = False) -> AILover | None:
        if user_id in self.ai_instances:
            return self.ai_instances[user_id]
        
        logger.info(f"使用者 {user_id} 沒有活躍的 AI 實例，嘗試創建...")
        ai_instance = AILover(user_id=user_id, is_ollama_available=self.is_ollama_available)
        
        if await ai_instance.initialize():
            logger.info(f"為使用者 {user_id} 成功創建並初始化 AI 實例。")
            await ai_instance._configure_pre_requisites()
            
            if not is_setup_flow:
                await ai_instance._load_or_build_rag_retriever()

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
            await ai_instance.shutdown()
            del ai_instance
            gc.collect()
            return None
    # 函式：獲取或創建使用者的 AI 實例

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





    
# 函式：監聽並處理所有符合條件的訊息 (v63.0 - 對接Graph架構)
    # 更新紀錄:
    # v63.0 (2025-10-03): [重大架構重構] 徹底重寫了此函式的核心邏輯。它現在不再調用舊的 `preprocess_and_generate`，而是負責構建初始的 `ConversationGraphState`，並直接調用 `main_graph.ainvoke()` 來驅動全新的、基於圖的工作流，實現了對話處理邏輯的現代化改造。
    # v62.0 (2025-10-02): [功能擴展] 新增了兩個核心的管理員調試指令。
    # v61.0 (2025-10-02): [災難性BUG修復] 根據「串行化」原則，再次徹底重構了創世流程 `_perform_full_setup_flow`。
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot: return
        
        user_id = str(message.author.id)

        if user_id in self.active_setups:
            logger.info(f"[{user_id}] (on_message) 偵測到用戶處於活躍的創世流程中，已忽略常規訊息 '{message.content[:50]}...' 以防止競爭。")
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.bot.user in message.mentions
        if not is_dm and not is_mentioned: return
        
        ctx = await self.bot.get_context(message)
        if ctx.valid: return
        
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
                logger.info(f"[{user_id}] 啟動 Graph-based 對話流程...")

                # 確保 main_graph 已被編譯
                if not self.bot.main_graph:
                    logger.error(f"[{user_id}] 災難性錯誤: main_graph 未被編譯，無法處理訊息。")
                    await message.channel.send("抱歉，系統內部圖形引擎未準備就緒，請聯繫管理員。")
                    return

                # 構建初始狀態
                initial_state = {
                    "user_id": user_id,
                    "ai_core": ai_instance,
                    "messages": [HumanMessage(content=user_input)]
                }

                # 調用 Graph
                final_state = await self.bot.main_graph.ainvoke(initial_state)
                final_response = final_state.get("final_output")
                
                if final_response and final_response.strip():
                    view = RegenerateView(cog=self)
                    for i in range(0, len(final_response), 2000):
                        current_view = view if i + 2000 >= len(final_response) else None
                        await message.channel.send(final_response[i:i+2000], view=current_view)
                    
                    logger.info(f"[{user_id}] Graph 流程執行完畢，回應已發送。")

                else:
                    logger.error(f"為使用者 {user_id} 的 Graph 流程返回了空的或無效的回應。")
                    await message.channel.send("（抱歉，我好像突然斷線了...）")

            except Exception as e:
                logger.error(f"處理使用者 {user_id} 的 Graph 流程時發生異常: {e}", exc_info=True)
                await message.channel.send(f"處理您的訊息時發生了一個嚴重的內部錯誤: `{type(e).__name__}`")
    # 函式：監聽並處理所有符合條件的訊息




    

# 函式：在背景处理世界圣经文本 (v2.0 - 纯向量RAG简化)
# 更新纪录:
# v2.0 (2025-10-02): [根本性重构] 根据“纯向量RAG优先”策略，彻底重写了此函式。它不再执行复杂的LORE解析，而是触发一个与 `/admin_pure_rag_rebuild` 类似的、破坏性的重建流程：1. 彻底清除用户的所有LORE和RAG数据。2. 仅使用新上传的文本文件，构建一个纯净的向量RAG索引。这确保了世界圣经始终是RAG知识库的唯一、最新的“真理之源”。
# v1.0 (原版)
    async def _background_process_canon(self, interaction: discord.Interaction, content_text: str, is_setup_flow: bool):
        """(v2.0) 彻底清除旧数据，并仅使用提供的文本内容重建一个纯向量RAG索引。"""
        user_id = str(interaction.user.id)
        user = self.bot.get_user(interaction.user.id) or await self.bot.fetch_user(interaction.user.id)
        
        # 如果是在 /start 流程中被调用，则转交总流程处理，避免重复操作
        if is_setup_flow:
            logger.info(f"[{user_id}] [Process Canon] 检测到处于 setup_flow，转交至 _perform_full_setup_flow 统一处理。")
            asyncio.create_task(self._perform_full_setup_flow(user=user, canon_text=content_text))
            return

        try:
            await user.send("⏳ **收到新的世界圣经！** 正在为您执行彻底的知识库重建流程，这将清除所有旧的 LORE 和记忆...")

            # --- 步骤 1: 彻底清理 ---
            logger.warning(f"[{user_id}] [Process Canon] 正在为使用者彻底清除所有现有数据...")
            if user_id in self.ai_instances:
                await self.ai_instances[user_id].shutdown()
                del self.ai_instances[user_id]
                gc.collect()
            
            async with AsyncSessionLocal() as session:
                await session.execute(delete(MemoryData).where(MemoryData.user_id == user_id))
                await session.execute(delete(Lore).where(Lore.user_id == user_id))
                await session.commit()
            
            vector_store_path = Path(f"./data/vector_stores/{user_id}")
            if vector_store_path.exists():
                await self._robust_rmtree(vector_store_path)
            
            logger.info(f"[{user_id}] [Process Canon] 数据清理完成。")

            # --- 步骤 2: 纯粹的 RAG 构建 ---
            ai_instance = await self.get_or_create_ai_instance(user_id)
            if not ai_instance:
                await user.send("❌ **重建失败！** 在清理后无法重新初始化 AI 實例。")
                return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            docs_for_rag = text_splitter.create_documents([content_text], metadatas=[{"source": "canon"} for _ in [content_text]])
            
            logger.info(f"[{user_id}] [Process Canon] 圣经已分割为 {len(docs_for_rag)} 个文档，正在构件纯向量索引...")
            
            await ai_instance._load_or_build_rag_retriever(docs_to_build=docs_for_rag)

            await user.send(f"✅ **知识库重建成功！**\n您的世界已完全基于新的圣经文本构建了一个包含 **{len(docs_for_rag)}** 个知识片段的纯向量 RAG 知识库。")

        except Exception as e:
            logger.error(f"[{user_id}] 背景处理世界圣经时发生错误: {e}", exc_info=True)
            await user.send(f"❌ **处理失败！**\n发生了严重错误: `{type(e).__name__}`\n请检查后台日志以获取详细资讯。")
# 函式：在背景处理世界圣经文本 (v2.0 - 纯向量RAG简化)


    

    # 函式：健壯的異步目錄刪除
    async def _robust_rmtree(self, path: Path, retries: int = 10, delay: float = 1.0):
        """
        一個極度健壯的異步 shutil.rmtree 版本，帶有更長的延遲、更多的重試次數和強制垃圾回收，以處理頑固的文件鎖定問題。
        """
        logger.info(f"[{path.name}] (Robust Delete) 正在啟動對目錄 {path} 的健壯刪除流程...")
        for i in range(retries):
            try:
                gc.collect()
                await asyncio.to_thread(shutil.rmtree, path)
                logger.info(f"[{path.name}] (Robust Delete) ✅ 在第 {i+1} 次嘗試中成功刪除目錄: {path}")
                return
            except PermissionError as e:
                logger.warning(f"[{path.name}] (Robust Delete) 刪除時遇到權限錯誤 (文件鎖)，準備在 {delay} 秒後重試 ({i+1}/{retries})...")
                await asyncio.sleep(delay)
            except FileNotFoundError:
                logger.info(f"[{path.name}] (Robust Delete) 目錄已不存在，無需刪除: {path}")
                return
            except Exception as e:
                logger.error(f"[{path.name}] (Robust Delete) 刪除時發生未預期的錯誤: {e}", exc_info=True)
                raise RuntimeError(f"刪除目錄 {path} 時發生不可恢復的錯誤: {e}") from e
        
        raise RuntimeError(f"在 {retries} 次嘗試後，仍然無法刪除目錄: {path}。請手動檢查文件鎖定。")
    # 函式：健壯的異步目錄刪除
    
    # 函式：開始 /start 指令的重置流程
    async def start_reset_flow(self, interaction: discord.Interaction):
        user_id = str(interaction.user.id)
        try:
            logger.info(f"[{user_id}] 後台重置任務開始...")
            
            if user_id in self.ai_instances:
                logger.info(f"[{user_id}] 檢測到活躍的 AI 實例，正在執行徹底的 shutdown 流程...")
                await self.ai_instances[user_id].shutdown()
                del self.ai_instances[user_id]
                gc.collect()
                logger.info(f"[{user_id}] AI 實例已銷毀，進入 2 秒的靜默期以等待 OS 釋放文件鎖...")
                await asyncio.sleep(2.0)
            else:
                logger.info(f"[{user_id}] 未檢測到活躍的 AI 實例，直接進行清理。")

            async with AsyncSessionLocal() as session:
                logger.info(f"[{user_id}] 正在清除所有資料庫記錄...")
                await session.execute(delete(SceneHistoryData).where(SceneHistoryData.user_id == user_id))
                await session.execute(delete(MemoryData).where(MemoryData.user_id == user_id))
                await session.execute(delete(Lore).where(Lore.user_id == user_id))
                await session.execute(delete(UserData).where(UserData.user_id == user_id))
                await session.commit()
                logger.info(f"[{user_id}] 所有資料庫記錄已成功清除。")
            
            vector_store_path = Path(f"./data/vector_stores/{user_id}")
            if vector_store_path.exists():
                await self._robust_rmtree(vector_store_path)
            
            view = StartSetupView(cog=self)
            await interaction.followup.send(
                content="✅ 重置完成！請點擊下方按鈕開始全新的設定流程。", 
                view=view, 
                ephemeral=True
            )
        except Exception as e:
            logger.error(f"[{user_id}] 後台重置任務失敗: {e}", exc_info=True)
            if not interaction.response.is_done():
                try:
                    await interaction.response.send_message(f"執行重置時發生未知的嚴重錯誤: {e}", ephemeral=True)
                except discord.errors.InteractionResponded:
                    await interaction.followup.send(f"執行重置時發生未知的嚴重錯誤: {e}", ephemeral=True)
            else:
                await interaction.followup.send(f"執行重置時發生未知的嚴重錯誤: {e}", ephemeral=True)
        finally:
            self.active_setups.discard(user_id)
    # 函式：開始 /start 指令的重置流程
    
    # 指令：開始全新的冒險（重置所有資料）
    @app_commands.command(name="start", description="開始全新的冒險（這將重置您所有的現有資料）")
    async def start(self, interaction: discord.Interaction):
        if not isinstance(interaction.channel, discord.DMChannel):
            await interaction.response.send_message("此指令只能在私訊頻道中使用。", ephemeral=True)
            return
        
        if str(interaction.user.id) in self.active_setups:
            await interaction.response.send_message("⏳ 您已經有一個創世流程正在後台執行，無法重複開始。請耐心等候或聯繫管理員。", ephemeral=True)
            return
            
        view = ConfirmStartView(cog=self)
        view.original_interaction_user_id = interaction.user.id
        await interaction.response.send_message(
            "⚠️ **警告** ⚠️\n您確定要開始一段全新的冒險嗎？\n這將會**永久刪除**您當前所有的角色、世界、記憶和進度。", 
            view=view, 
            ephemeral=True
        )
        view.message = await interaction.original_response()
    # 指令：開始全新的冒險（重置所有資料）

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
    @app_commands.command(name="response_style", description="客製化 AI 的回覆風格")
    async def response_style(self, interaction: discord.Interaction):
        if not isinstance(interaction.channel, discord.DMChannel):
            await interaction.response.send_message("此指令只能在私訊頻道中使用。", ephemeral=True)
            return
        ai_instance = await self.get_or_create_ai_instance(str(interaction.user.id))
        if not ai_instance or not ai_instance.profile:
            await interaction.response.send_message("請先使用 `/start` 指令進行初始設定。", ephemeral=True)
            return
        
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
        modal = WorldCanonPasteModal(self.cog, original_interaction_message_id=None)
        await interaction.response.send_modal(modal)
    # 指令：通過貼上文字來設定世界聖經

    # 指令：通過上傳檔案來設定世界聖經
    @app_commands.command(name="set_canon_file", description="通過上傳 .txt 檔案來設定您的世界聖經")
    @app_commands.describe(file="請上傳一個 .txt 格式的檔案，最大 5MB。")
    async def set_canon_file(self, interaction: discord.Interaction, file: discord.Attachment):
        await interaction.response.defer(ephemeral=True, thinking=True)

        if not file.filename.lower().endswith('.txt'):
            await interaction.followup.send("❌ 檔案格式錯誤！請上傳一個 .txt 檔案。", ephemeral=True)
            return
        
        if file.size > 5 * 1024 * 1024:
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

            user_id = str(interaction.user.id)
            is_currently_in_setup = user_id in self.active_setups
            
            if is_currently_in_setup:
                if interaction.channel and interaction.message:
                     try:
                        original_message = await interaction.channel.fetch_message(interaction.message.id)
                        view = discord.ui.View.from_message(original_message)
                        for item in view.children: item.disabled = True
                        await original_message.edit(view=view)
                     except (discord.errors.NotFound, AttributeError): pass
            
            await interaction.followup.send("✅ 檔案已接收！正在後台為您進行向量化和智能解析，這可能需要幾分鐘時間，請稍候...", ephemeral=True)
            
            asyncio.create_task(self._background_process_canon(interaction, content_text, is_setup_flow=is_currently_in_setup))
        except Exception as e:
            logger.error(f"處理上傳的世界聖經檔案時發生錯誤: {e}", exc_info=True)
            await interaction.followup.send(f"讀取或處理檔案時發生嚴重錯誤: `{type(e).__name__}`", ephemeral=True)
    # 指令：通過上傳檔案來設定世界聖經

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
            if vector_store_path.exists(): await self._robust_rmtree(vector_store_path)
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
    @app_commands.describe(target_user="要查詢的使用者", category="LORE 的類別", key="LORE 的主鍵")
    @app_commands.autocomplete(target_user=user_autocomplete, key=lore_key_autocomplete)
    @app_commands.choices(category=LORE_CATEGORIES)
    async def admin_check_lore(self, interaction: discord.Interaction, target_user: str, category: str, key: str):
        await interaction.response.defer(ephemeral=True)
        lore_entry = await lore_book.get_lore(target_user, category, key)
        if lore_entry:
            content_str = json.dumps(lore_entry.content, ensure_ascii=False, indent=2)
            
            if len(content_str) > 1000:
                try:
                    temp_dir = PROJ_DIR / "temp"
                    temp_dir.mkdir(exist_ok=True)
                    
                    file_path = temp_dir / f"lore_{interaction.user.id}_{int(time.time())}.json"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content_str)
                    
                    file_name = f"{key.replace(' > ', '_').replace('/', '_')}.json"

                    await interaction.followup.send(
                        f"📜 **Lore 查詢結果 for `{key}`**\n（由於內容過長，已作為檔案附件發送）", 
                        file=discord.File(file_path, filename=file_name),
                        ephemeral=True
                    )
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"[{interaction.user.id}] 創建或發送LORE檔案時出錯: {e}", exc_info=True)
                    await interaction.followup.send("錯誤：創建LORE檔案時發生問題。", ephemeral=True)
            else:
                embed = Embed(title=f"📜 Lore 查詢: {key.split(' > ')[-1]}", color=discord.Color.green())
                embed.add_field(name="詳細資料", value=f"```json\n{content_str}\n```", inline=False)
                embed.set_footer(text=f"User: {target_user} | Category: {category}")
                await interaction.followup.send(embed=embed, ephemeral=True)
        else: 
            await interaction.followup.send(f"錯誤：在類別 `{category}` 中找不到 key 為 `{key}` 的 Lore。", ephemeral=True)
    # 指令：[管理員] 查詢 Lore 詳細資料
        
    # 指令：[管理員] 推送日誌
    @app_commands.command(name="admin_push_log", description="[管理員] 強制將最新的日誌推送到GitHub倉庫。")
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
            with open(log_file_path, 'r', encoding='utf-8') as f: latest_lines = f.readlines()[-300:]
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



    # 檔案：discord_bot.py (在 BotCog 類別內)

# 函式：[管理員] 純向量RAG重建 (v1.1 - 移除自動測試)
# 更新紀錄:
# v1.1 (2025-10-02): [功能簡化] 根據使用者要求，移除了指令末尾的自動測試查詢和結果預覽功能。現在，此指令的職責被簡化為純粹的「清理與重建」，完成後會提示管理員使用 `/admin_rag_peek` 等其他工具自行進行驗證，使其更符合單一職責原則。
# v1.0 (2025-10-02): [全新創建] 創建此終極 RAG 調試工具。
    @app_commands.command(name="admin_pure_rag_rebuild", description="[管理員] 上傳TXT，徹底重建並僅使用純向量RAG。")
    @app_commands.check(is_admin)
    @app_commands.describe(file="您的世界聖經 .txt 檔案。")
    async def admin_pure_rag_rebuild(self, interaction: discord.Interaction, file: discord.Attachment):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user_id = str(interaction.user.id)

        if not file.filename.lower().endswith('.txt'):
            await interaction.followup.send("❌ 檔案格式錯誤！請上傳一個 .txt 檔案。", ephemeral=True)
            return

        try:
            logger.info(f"[{user_id}] [Admin Command] 啟動純向量 RAG 重建流程...")
            
            # --- 步驟 1: 徹底清理 ---
            logger.warning(f"[{user_id}] [Pure RAG Rebuild] 正在為使用者徹底清除所有現有數據...")
            if user_id in self.ai_instances:
                await self.ai_instances[user_id].shutdown()
                del self.ai_instances[user_id]
                gc.collect()
            
            async with AsyncSessionLocal() as session:
                await session.execute(delete(MemoryData).where(MemoryData.user_id == user_id))
                await session.execute(delete(Lore).where(Lore.user_id == user_id))
                await session.commit()
            
            vector_store_path = Path(f"./data/vector_stores/{user_id}")
            if vector_store_path.exists():
                await self._robust_rmtree(vector_store_path)
            
            logger.info(f"[{user_id}] [Pure RAG Rebuild] 數據清理完成。")

            # --- 步驟 2: 純粹的 RAG 構建 ---
            ai_instance = await self.get_or_create_ai_instance(user_id)
            if not ai_instance:
                await interaction.followup.send("❌ 錯誤：在清理後無法重新初始化 AI 實例。", ephemeral=True)
                return

            content_bytes = await file.read()
            content_text = content_bytes.decode('utf-8', errors='ignore')
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            docs_for_rag = text_splitter.create_documents([content_text], metadatas=[{"source": "canon"} for _ in [content_text]])
            
            logger.info(f"[{user_id}] [Pure RAG Rebuild] 聖經已分割為 {len(docs_for_rag)} 個文檔，正在構建純向量索引...")
            
            await ai_instance._load_or_build_rag_retriever(docs_to_build=docs_for_rag)

            logger.info(f"[{user_id}] [Pure RAG Rebuild] 純向量 RAG 索引構建完成。")

            # --- 步驟 3: [v1.1 核心修正] 發送成功通知 ---
            embed = discord.Embed(
                title="✅ 純向量 RAG 重建完成",
                description=f"已為您清除了所有舊的 LORE 和 RAG 數據，並僅使用 `{file.filename}` 的內容重新構建了一個**純向量 RAG 索引**。",
                color=discord.Color.green()
            )
            embed.add_field(name="索引中文檔總數", value=f"`{len(docs_for_rag)}` 個", inline=False)
            embed.add_field(name="下一步", value="您現在可以使用 `/admin_rag_peek` 指令來手動測試新索引的檢索效果。", inline=False)
            
            await interaction.followup.send(embed=embed, ephemeral=True)

        except Exception as e:
            logger.error(f"[{user_id}] 執行 admin_pure_rag_rebuild 時發生錯誤: {e}", exc_info=True)
            await interaction.followup.send(f"❌ 執行時發生嚴重錯誤: `{type(e).__name__}`\n請檢查後台日誌。", ephemeral=True)
# 函式：[管理員] 純向量RAG重建 (v1.1 - 移除自動測試)

    # [v62.0 新增] 管理員指令：窺探 RAG 檢索結果
    @app_commands.command(name="admin_rag_peek", description="[管理員] 輸入查詢，直接查看 RAG 返回的原始文檔內容。")
    @app_commands.check(is_admin)
    @app_commands.describe(query="您想用來查詢 RAG 的文本內容。")
    async def admin_rag_peek(self, interaction: discord.Interaction, query: str):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user_id = str(interaction.user.id)
        
        try:
            ai_instance = await self.get_or_create_ai_instance(user_id)
            if not ai_instance or not ai_instance.retriever:
                await interaction.followup.send("❌ 錯誤：AI 實例或 RAG 檢索器未初始化。", ephemeral=True)
                return

            logger.info(f"[{user_id}] [Admin Command] 執行 RAG Peek，查詢: '{query}'")
            
            retrieved_docs = await ai_instance.retriever.ainvoke(query)

            if not retrieved_docs:
                await interaction.followup.send("ℹ️ RAG 系統未返回任何文檔。", ephemeral=True)
                return

            output_parts = []
            output_parts.append(f"--- RAG Peek 原始檢索結果 ---\n")
            output_parts.append(f"查詢原文: {query}\n")
            output_parts.append(f"檢索到文檔數量: {len(retrieved_docs)}\n")
            output_parts.append("="*40 + "\n\n")

            for i, doc in enumerate(retrieved_docs):
                output_parts.append(f"--- 文檔 #{i+1} ---\n")
                output_parts.append(f"【元數據】:\n{json.dumps(doc.metadata, indent=2, ensure_ascii=False)}\n\n")
                output_parts.append(f"【文檔內容】:\n{doc.page_content}\n")
                output_parts.append("="*40 + "\n\n")
            
            output_text = "".join(output_parts)
            temp_dir = PROJ_DIR / "temp"
            temp_dir.mkdir(exist_ok=True)
            file_path = temp_dir / f"rag_peek_{user_id}_{int(time.time())}.txt"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(output_text)
            
            await interaction.followup.send(
                f"✅ RAG 系統為您的查詢返回了 **{len(retrieved_docs)}** 條原始文檔。詳情請見附件：",
                file=discord.File(file_path, filename=f"rag_peek_results.txt"),
                ephemeral=True
            )
            
            os.remove(file_path)

        except Exception as e:
            logger.error(f"[{user_id}] 執行 admin_rag_peek 時發生錯誤: {e}", exc_info=True)
            await interaction.followup.send(f"❌ 執行時發生嚴重錯誤: `{type(e).__name__}`\n請檢查後台日誌。", ephemeral=True)

# 函式：[管理員] RAG 直通 LLM 對話 (v1.2 - 外部重試鏈)
# 更新紀錄:
# v1.2 (2025-10-02): [災難性BUG修復] 根據 BlockedPromptException 日誌，徹底重構了此指令的錯誤處理和重試邏輯。新版本從依賴 `ainvoke_with_rotation` 的內部重試，改為實現一個顯式的「外部重試鏈」。當第一次調用失敗時，它會捕獲異常，手動執行「最高指令注入重試」；如果再次失敗，則會再次捕獲異常，並手動調用終極的「委婉化重構」備援。此修改確保了重試流程的可控性和有效性，最大限度地提升了抗審查成功率。
# v1.1 (2025-10-02): [災難性BUG修復] 修正了變數名 NameError。
# v1.0 (2025-10-02): [全新創建] 創建此終極調試指令。
    @app_commands.command(name="admin_direct_chat", description="[管理員] RAG直通LLM，用於測試最原始的回應。")
    @app_commands.check(is_admin)
    @app_commands.describe(prompt="您想直接發送給 LLM 的對話內容。")
    async def admin_direct_chat(self, interaction: discord.Interaction, prompt: str):
        await interaction.response.defer(ephemeral=False, thinking=True) # 回應設為公開
        user_id = str(interaction.user.id)
        
        raw_response = None
        
        try:
            ai_instance = await self.get_or_create_ai_instance(user_id)
            if not ai_instance or not ai_instance.profile:
                await interaction.followup.send("❌ 錯誤：AI 實例或 Profile 未初始化。")
                return

            logger.info(f"[{user_id}] [Admin Command] 執行 RAG 直通 LLM，Prompt: '{prompt[:100]}...'")

            rag_context_dict = await ai_instance.retrieve_and_summarize_memories(prompt)
            rag_context = rag_context_dict.get("summary", "（RAG 未返回任何摘要信息。）")
            rag_rules = rag_context_dict.get("rules", "（RAG 未返回任何規則信息。）")
            
            full_rag_context = f"--- RAG 檢索到的規則 ---\n{rag_rules}\n\n--- RAG 檢索到的背景摘要 ---\n{rag_context}"

            final_prompt_template = """{core_protocol}

# === 情報簡報 ===
{rag_context}
# === 情報結束 ===

# === 對話開始 ===
{username}: {user_prompt}
{ai_name}:"""
            
            final_prompt = final_prompt_template.format(
                core_protocol=ai_instance.core_protocol_prompt,
                rag_context=full_rag_context,
                username=ai_instance.profile.user_profile.name,
                user_prompt=prompt,
                ai_name=ai_instance.profile.ai_profile.name
            )
            
            temp_dir = PROJ_DIR / "temp"
            temp_dir.mkdir(exist_ok=True)
            prompt_file_path = temp_dir / f"direct_chat_prompt_{user_id}_{int(time.time())}.txt"
            with open(prompt_file_path, 'w', encoding='utf-8') as f:
                f.write(final_prompt)
            
            await interaction.followup.send(
                f"✅ RAG 上下文已獲取，最終 Prompt 已構建。正在調用 `{GENERATION_MODEL_PRIORITY[0]}` 模型 (第 1 次嘗試)...",
                file=discord.File(prompt_file_path, filename="last_direct_prompt.txt")
            )
            os.remove(prompt_file_path)

            # --- [v1.2 核心重構] 外部重試鏈 ---
            try:
                # 第一次嘗試：標準調用
                logger.info(f"[{user_id}] [Direct Chat] 執行第 1 次嘗試 (標準模式)...")
                raw_response = await ai_instance.ainvoke_with_rotation(
                    final_prompt,
                    output_schema=None,
                    retry_strategy='none', # 關鍵：失敗時立即拋出異常
                    use_degradation=True,
                    models_to_try_override=[GENERATION_MODEL_PRIORITY[0]]
                )
            except BlockedPromptException as e1:
                logger.warning(f"[{user_id}] [Direct Chat] 第 1 次嘗試被審查 ({e1})。啟動第 2 次嘗試 (最高指令注入)...")
                
                try:
                    # 第二次嘗試：最高指令注入
                    forceful_prompt = final_prompt + f"\n\n{ai_instance.core_protocol_prompt}"
                    raw_response = await ai_instance.ainvoke_with_rotation(
                        forceful_prompt,
                        output_schema=None,
                        retry_strategy='none', # 關鍵：失敗時再次拋出
                        use_degradation=True,
                        models_to_try_override=[GENERATION_MODEL_PRIORITY[0]]
                    )
                except BlockedPromptException as e2:
                    logger.warning(f"[{user_id}] [Direct Chat] 第 2 次嘗試被審查 ({e2})。啟動最終備援 (委婉化重構)...")
                    # 最終備援：委婉化重構
                    raw_response = await ai_instance._euphemize_and_retry(final_prompt, None, e2)

            # --- 處理最終結果 ---
            if raw_response and raw_response.strip():
                decoded_response = ai_instance._decode_lore_content(raw_response.strip(), ai_instance.DECODING_MAP)
                for i in range(0, len(decoded_response), 2000):
                    await interaction.channel.send(decoded_response[i:i+2000])
            else:
                await interaction.channel.send("❌ LLM 在所有備援策略後，最終返回了空回應或無法生成有效內容。")

        except Exception as e:
            logger.error(f"[{user_id}] 執行 admin_direct_chat 時發生嚴重錯誤: {e}", exc_info=True)
            # 檢查 interaction 是否還能回應
            if interaction.is_is_done():
                await interaction.channel.send(f"❌ 執行時發生嚴重錯誤: `{type(e).__name__}`\n請檢查後台日誌。")
            else:
                 await interaction.followup.send(f"❌ 執行時發生嚴重錯誤: `{type(e).__name__}`\n請檢查後台日誌。")
# 指令：[管理員] RAG 直通 LLM 對話 (v1.2 - 外部重試鏈)

    
        
    # 函式：全域應用程式指令錯誤處理器
    @commands.Cog.listener()
    async def on_app_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.CheckFailure):
            await interaction.response.send_message("你沒有權限使用此指令。", ephemeral=True)
        else:
            logger.error(f"一個應用程式指令發生錯誤: {error}", exc_info=True)
            if not interaction.response.is_done():
                try:
                    await interaction.response.send_message(f"發生未知錯誤: {error}", ephemeral=True)
                except discord.errors.InteractionResponded:
                    await interaction.followup.send(f"發生未知錯誤: {error}", ephemeral=True)
            else:
                await interaction.followup.send(f"發生未知錯誤: {error}", ephemeral=True)
    # 函式：全域應用程式指令錯誤處理器
# 類別：機器人核心功能集 (Cog)

# 類別：AI 戀人機器人主體 (v1.1 - 新增Graph屬性)
# 更新紀錄:
# v1.1 (2025-10-03): [架構擴展] 新增了 self.main_graph 和 self.setup_graph 兩個實例屬性，並將其初始化為 None。這些屬性將在 setup_hook 中被賦值為編譯好的 LangGraph 實例，以供整個應用程式調用。
# v1.0 (初始版本)
class AILoverBot(commands.Bot):
    # 函式：初始化 AILoverBot
    def __init__(self, shutdown_event: asyncio.Event, git_lock: asyncio.Lock, is_ollama_available: bool):
        super().__init__(command_prefix='/', intents=intents, activity=discord.Game(name="與你共度時光"))
        self.shutdown_event = shutdown_event
        self.git_lock = git_lock
        self.is_ready_once = False
        self.is_ollama_available = is_ollama_available
        # [v1.1 新增] 初始化 Graph 屬性
        self.main_graph = None
        self.setup_graph = None
    # 函式：初始化 AILoverBot

    
# 函式：Discord 機器人設置鉤子 (v1.1 - 編譯Graph)
    # 更新紀錄:
    # v1.1 (2025-10-03): [重大架構升級] 在此函式中導入了 `create_main_response_graph` 和 `create_setup_graph`。在 setup_hook 階段，它會立即編譯這兩個核心工作流圖，並將結果賦值給 `self.main_graph` 和 `self.setup_graph`，確保在機器人響應任何請求之前，所有的核心邏輯都已準備就緒。
    # v1.0 (初始版本)
    async def setup_hook(self):
        # [v1.1 新增] 導入並編譯 Graph
        from src.graph import create_main_response_graph, create_setup_graph
        logger.info("正在編譯主對話圖 (main_graph)...")
        self.main_graph = create_main_response_graph()
        logger.info("✅ 主對話圖編譯成功。")
        logger.info("正在編譯創世流程圖 (setup_graph)...")
        self.setup_graph = create_setup_graph()
        logger.info("✅ 創世流程圖編譯成功。")

        cog = BotCog(self, self.git_lock, self.is_ollama_available)
        await self.add_cog(cog)

        cog.connection_watcher.start()
        
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
    # 函式：Discord 機器人設置鉤子
    
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










