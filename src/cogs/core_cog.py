# src/cogs/core_cog.py 的中文註釋(v1.0 - 結構分離)
# 更新紀錄:
# v1.0 (2025-10-04): [災難性BUG修復-終極方案] 創建此 Cog 檔案，將所有指令、UI元件 (Views/Modals) 和事件監聽器從主 bot 檔案中分離出來。此結構性重構旨在徹底解決因模組初始化悖論導致的 NameError 和 AttributeError。

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
import datetime
import time

# 使用相對導入來引用 src 包內的其他模組
from ..logger import logger
from ..ai_core import AILover, GENERATION_MODEL_PRIORITY
from .. import lore_book
from ..lore_book import Lore
from ..database import AsyncSessionLocal, UserData, MemoryData, init_db, SceneHistoryData
from ..schemas import CharacterProfile, LocationInfo, WorldGenesisResult
from ..models import UserProfile, GameState
from ..config import settings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.generativeai.types import BlockedPromptException

# 由於 AILoverBot 在另一個文件中，我們需要進行類型提示的特殊處理
if sys.version_info >= (3, 9):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from ..discord_bot import AILoverBot
else:
    from ..discord_bot import AILoverBot


PROJ_DIR = Path(__file__).resolve().parent.parent.parent

# --- 所有輔助函式、View/Modal 類別現在都定義在這裡 ---

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

            logger.info(f"[{user_id}] [重新生成] 正在使用上次輸入 '{ai_instance.last_user_input[:50]}...' 重新生成回應...")
            
            # 直接調用 RAG 直通函式
            final_response = await ai_instance.direct_rag_generate(ai_instance.last_user_input)

            if final_response and final_response.strip():
                view = self
                for i in range(0, len(final_response), 2000):
                    current_view = view if i + 2000 >= len(final_response) else None
                    await interaction.channel.send(final_response[i:i+2000], view=current_view)
                
                logger.info(f"[{user_id}] [重新生成] 新回應已發送。")
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

# --- Cog 類別定義 ---

class BotCog(commands.Cog, name="BotCog"):
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

    # 函式：執行完整的後台創世流程 (v65.0 - 原生創世流程)
    async def _perform_full_setup_flow(self, user: discord.User, canon_text: Optional[str] = None):
        """(v65.0) 一個由原生 Python `await` 驅動的、獨立的後台創世流程。"""
        user_id = str(user.id)
        try:
            logger.info(f"[{user_id}] [創世流程 v65.0] 原生 Python 驅動的流程已啟動。")
            
            ai_instance = await self.get_or_create_ai_instance(user_id, is_setup_flow=True)
            if not ai_instance or not ai_instance.profile:
                await user.send("❌ 錯誤：無法初始化您的 AI 核心以進行創世。")
                return

            docs_for_rag = []
            if canon_text and canon_text.strip():
                logger.info(f"[{user_id}] [後台創世] 正在將世界聖經原文分割成文檔...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
                docs_for_rag = text_splitter.create_documents([canon_text], metadatas=[{"source": "canon"} for _ in [canon_text]])
            
            logger.info(f"[{user_id}] [後台創世] 正在觸發 RAG 索引創始構建...")
            await ai_instance._load_or_build_rag_retriever(force_rebuild=True, docs_to_build=docs_for_rag if docs_for_rag else None)
            logger.info(f"[{user_id}] [後台創世] RAG 索引構建完成，準備執行原生創世步驟...")

            logger.info(f"[{user_id}] [後台創世-原生] 步驟 1/2: 正在補完角色檔案...")
            await ai_instance.complete_character_profiles()
            logger.info(f"[{user_id}] [後台創世-原生] 角色檔案補完成功。")

            logger.info(f"[{user_id}] [後台創世-原生] 步驟 2/2: 正在生成開場白...")
            opening_scene = await ai_instance.generate_opening_scene(canon_text=canon_text)
            logger.info(f"[{user_id}] [後台創世-原生] 開場白生成成功。")

            if not opening_scene:
                 raise Exception("原生創世流程未能成功生成開場白。")

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
    # 執行完整的後台創世流程 函式結束

    # 函式：獲取或創建使用者的 AI 實例
    async def get_or_create_ai_instance(self, user_id: str, is_setup_flow: bool = False) -> Optional[AILover]:
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
                logger.info(f"[{user_id}] 啟動 RAG 直通對話流程...")
                final_response = await ai_instance.direct_rag_generate(user_input)
                
                if final_response and final_response.strip():
                    view = RegenerateView(cog=self)
                    for i in range(0, len(final_response), 2000):
                        current_view = view if i + 2000 >= len(final_response) else None
                        await message.channel.send(final_response[i:i+2000], view=current_view)
                    
                    logger.info(f"[{user_id}] RAG 直通流程執行完畢，回應已發送。事後學習任務已在背景啟動。")

                else:
                    logger.error(f"為使用者 {user_id} 的 RAG 直通流程返回了空的或無效的回應。")
                    await message.channel.send("（抱歉，我好像突然斷線了...）")

            except Exception as e:
                logger.error(f"處理使用者 {user_id} 的 RAG 直通流程時發生異常: {e}", exc_info=True)
                await message.channel.send(f"處理您的訊息時發生了一個嚴重的內部錯誤: `{type(e).__name__}`")
    
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
        modal = WorldCanonPasteModal(self, original_interaction_message_id=None)
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
                        # 這裡的 interaction.message 可能為 None，需要檢查
                        if interaction.message:
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

    # 指令：[管理員] 版本控制
    @app_commands.command(name="admin_version_control", description="[管理員] 打開圖形化版本控制面板。")
    @app_commands.check(is_admin)
    async def admin_version_control(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        view = VersionControlView(cog=self, original_user_id=interaction.user.id)
        embed = await view._build_embed()
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)
    # 指令：[管理員] 版本控制

    # 函式：[管理員] 純向量RAG重建 (v1.1 - 移除自動測試)
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

            ai_instance = await self.get_or_create_ai_instance(user_id)
            if not ai_instance:
                await interaction.followup.send("❌ 錯誤：在清理後無法重新初始化 AI 實例。", ephemeral=True)
                return

            content_bytes = await file.read()
            content_text = content_bytes.decode('utf-8', errors='ignore')
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            docs_for_rag = text_splitter.create_documents([content_text], metadatas=[{"source": "canon"} for _ in [content_text]])
            
            await ai_instance._load_or_build_rag_retriever(docs_to_build=docs_for_rag)

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

    # 管理員指令：窺探 RAG 檢索結果
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

    # 指令：[管理員] RAG 直通 LLM 對話
    @app_commands.command(name="admin_direct_chat", description="[管理員] RAG直通LLM，用於測試最原始的回應。")
    @app_commands.check(is_admin)
    @app_commands.describe(prompt="您想直接發送給 LLM 的對話內容。")
    async def admin_direct_chat(self, interaction: discord.Interaction, prompt: str):
        await interaction.response.defer(ephemeral=False, thinking=True)
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

            try:
                raw_response = await ai_instance.ainvoke_with_rotation(
                    final_prompt,
                    output_schema=None,
                    retry_strategy='none',
                    use_degradation=True,
                    models_to_try_override=[GENERATION_MODEL_PRIORITY[0]]
                )
            except BlockedPromptException as e1:
                logger.warning(f"[{user_id}] [Direct Chat] 第 1 次嘗試被審查 ({e1})。啟動第 2 次嘗試 (最高指令注入)...")
                
                try:
                    forceful_prompt = final_prompt + f"\n\n{ai_instance.core_protocol_prompt}"
                    raw_response = await ai_instance.ainvoke_with_rotation(
                        forceful_prompt,
                        output_schema=None,
                        retry_strategy='none',
                        use_degradation=True,
                        models_to_try_override=[GENERATION_MODEL_PRIORITY[0]]
                    )
                except BlockedPromptException as e2:
                    logger.warning(f"[{user_id}] [Direct Chat] 第 2 次嘗試被審查 ({e2})。啟動最終備援 (委婉化重構)...")
                    raw_response = await ai_instance._euphemize_and_retry(final_prompt, None, e2)

            if raw_response and raw_response.strip():
                decoded_response = ai_instance._decode_lore_content(raw_response.strip(), ai_instance.DECODING_MAP)
                for i in range(0, len(decoded_response), 2000):
                    await interaction.channel.send(decoded_response[i:i+2000])
            else:
                await interaction.channel.send("❌ LLM 在所有備援策略後，最終返回了空回應或無法生成有效內容。")

        except Exception as e:
            logger.error(f"[{user_id}] 執行 admin_direct_chat 時發生嚴重錯誤: {e}", exc_info=True)
            if interaction.is_is_done():
                await interaction.channel.send(f"❌ 執行時發生嚴重錯誤: `{type(e).__name__}`\n請檢查後台日誌。")
            else:
                 await interaction.followup.send(f"❌ 執行時發生嚴重錯誤: `{type(e).__name__}`\n請檢查後台日誌。")
    # 指令：[管理員] RAG 直通 LLM 對話

  
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

# --- Cog 設置函式 ---

async def setup(bot: "AILoverBot"):
    """Cog 的標準入口點函式"""
    # 創建 Cog 實例時，從 bot 實例獲取所需的依賴
    cog_instance = BotCog(bot, bot.git_lock, bot.is_ollama_available)
    await bot.add_cog(cog_instance)
    
    # 在 Cog 加載後，註冊持久化視圖
    bot.add_view(StartSetupView(cog=cog_instance))
    bot.add_view(ContinueToUserSetupView(cog=cog_instance))
    bot.add_view(ContinueToAiSetupView(cog=cog_instance))
    bot.add_view(ContinueToCanonSetupView(cog=cog_instance))
    bot.add_view(RegenerateView(cog=cog_instance))
    
    logger.info("✅ 核心 Cog (core_cog) 已加載，並且所有持久化視圖已成功註冊。")
