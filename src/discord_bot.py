# src/discord_bot.py 的中文註釋(v46.0 - 持久化視圖重構)
# 更新紀錄:
# v46.0 (2025-10-02): [災難性BUG修復] 為了從根本上解決因後端重啟導致的 UI 狀態丟失（僵屍UI）問題，徹底重構了整個 /start 設置流程。所有用於流程推進的 View（如 StartSetupView, ContinueToUserSetupView 等）現在都實現了 discord.py 的“持久化視圖”：它們的 timeout 被設為 None，並且所有關鍵按鈕都擁有了固定的 custom_id。機器人現在會在啟動時通過 setup_hook 重新註冊這些視圖，確保即使在重啟後，舊消息上的按鈕點擊依然能夠被正確地響應和處理。
# v45.0 (2025-10-01): [災難性BUG修復] 採用“链式 Modals”策略重构了 /start 流程。
# v42.0 (2025-09-04): [災難性BUG修復] 彻底重构了 on_message 事件。

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
from .database import AsyncSessionLocal, UserData, MemoryData, init_db
from .schemas import CharacterProfile, LocationInfo, WorldGenesisResult
from .models import UserProfile, GameState
from src.config import settings
from .graph import create_main_response_graph, create_setup_graph
from .graph_state import ConversationGraphState, SetupGraphState
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.output_parsers import StrOutputParser

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

# --- [v46.0 新增] 持久化視圖 (Persistent Views) ---

# --- [v46.2 新增] /start 流程持久化視圖與 Modals ---
# 更新紀錄:
# v46.2 (2025-10-12): [災難性BUG修復] 彻底重构了 /start 流程的 UI 交互逻辑，以修复因错误使用 interaction API 导致的流程中断问题。严格遵循“一次互动，一次初始响应”的原则。现在，所有按钮点击的唯一响应就是 `response.send_modal()`，而禁用旧视图按钮的操作被移到了下一个 Modal 的 on_submit 方法中执行，确保了流程的绝对连贯性。

# 類別：开始设定视图 (v46.2 - 持久化与 API 调用修复)
class StartSetupView(discord.ui.View):
    def __init__(self, *, cog: "BotCog"):
        super().__init__(timeout=None)
        self.cog = cog

    @discord.ui.button(label="🚀 開始設定", style=discord.ButtonStyle.success, custom_id="persistent_start_setup_button")
    async def start_setup_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) Persistent 'StartSetupView' button clicked.")
        
        # [v46.2 核心修正] 一次互动只能有一次初始响应。我们唯一的响应就是弹出 Modal。
        world_modal = WorldSettingsModal(
            self.cog, 
            current_world="這是一個魔法與科技交織的幻想世界。", 
            is_setup_flow=True,
            original_interaction_message_id=interaction.message.id # 传递原始消息ID
        )
        await interaction.response.send_modal(world_modal)
# 類別：开始设定视图 (v46.2 - 持久化与 API 调用修复)

# 類別：继续使用者设定视图 (v46.2 - 持久化与 API 调用修复)
class ContinueToUserSetupView(discord.ui.View):
    def __init__(self, *, cog: "BotCog"):
        super().__init__(timeout=None)
        self.cog = cog

    @discord.ui.button(label="下一步：設定您的角色", style=discord.ButtonStyle.primary, custom_id="persistent_continue_to_user_setup")
    async def continue_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) Persistent 'ContinueToUserSetupView' button clicked.")

        ai_instance = await self.cog.get_or_create_ai_instance(user_id, is_setup_flow=True)
        profile_data = ai_instance.profile.user_profile.model_dump() if ai_instance and ai_instance.profile else {}
        
        modal = CharacterSettingsModal(
            self.cog, 
            title="步驟 2/3: 您的角色設定", 
            profile_data=profile_data, 
            profile_type='user', 
            is_setup_flow=True,
            original_interaction_message_id=interaction.message.id
        )
        await interaction.response.send_modal(modal)
# 類別：继续使用者设定视图 (v46.2 - 持久化与 API 调用修复)

# 類別：继续 AI 设定视图 (v46.2 - 持久化与 API 调用修复)
class ContinueToAiSetupView(discord.ui.View):
    def __init__(self, *, cog: "BotCog"):
        super().__init__(timeout=None)
        self.cog = cog

    @discord.ui.button(label="最後一步：設定 AI 戀人", style=discord.ButtonStyle.primary, custom_id="persistent_continue_to_ai_setup")
    async def continue_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) Persistent 'ContinueToAiSetupView' button clicked.")
        
        ai_instance = await self.cog.get_or_create_ai_instance(str(interaction.user.id), is_setup_flow=True)
        profile_data = ai_instance.profile.ai_profile.model_dump() if ai_instance and ai_instance.profile else {}

        modal = CharacterSettingsModal(
            self.cog, 
            title="步驟 3/3: AI 戀人設定", 
            profile_data=profile_data, 
            profile_type='ai', 
            is_setup_flow=True,
            original_interaction_message_id=interaction.message.id
        )
        await interaction.response.send_modal(modal)
# 類別：继续 AI 设定视图 (v46.2 - 持久化与 API 调用修复)

# 類別：继续世界圣经设定视图 (v46.2 - 持久化与 API 调用修复)
class ContinueToCanonSetupView(discord.ui.View):
    def __init__(self, *, cog: "BotCog"):
        super().__init__(timeout=None)
        self.cog = cog

    @discord.ui.button(label="📄 貼上世界聖經 (文字)", style=discord.ButtonStyle.success, custom_id="persistent_paste_canon")
    async def paste_canon(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) Persistent 'ContinueToCanonSetupView' paste button clicked.")
        
        modal = WorldCanonPasteModal(self.cog, is_setup_flow=True, original_interaction_message_id=interaction.message.id)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="✅ 完成設定並開始冒險 (跳過聖經)", style=discord.ButtonStyle.primary, custom_id="persistent_finalize_setup")
    async def finalize(self, interaction: discord.Interaction, button: discord.ui.Button):
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) Persistent 'ContinueToCanonSetupView' finalize button clicked.")
        
        # 禁用按钮并更新消息
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(content="✅ 基礎設定完成！正在為您啟動創世...", view=self)

        # 在后台启动创世
        asyncio.create_task(self.cog.finalize_setup(interaction, canon_text=None))
# 類別：继续世界圣经设定视图 (v46.2 - 持久化与 API 调用修复)

# --- End of Persistent Views ---

# 類別：世界圣经贴上文字弹出视窗 (v46.2 - 流程串联)
class WorldCanonPasteModal(discord.ui.Modal, title="貼上您的世界聖經文本"):
    canon_text = discord.ui.TextInput(label="請將您的世界觀/角色背景故事貼於此處", style=discord.TextStyle.paragraph, placeholder="在此貼上您的 .txt 檔案內容或直接編寫...", required=True, max_length=4000)

    def __init__(self, cog: "BotCog", is_setup_flow: bool = False, original_interaction_message_id: int = None):
        super().__init__(timeout=600.0)
        self.cog = cog
        self.is_setup_flow = is_setup_flow
        self.original_interaction_message_id = original_interaction_message_id

    async def on_submit(self, interaction: discord.Interaction):
        # 禁用旧视图的按钮
        if self.original_interaction_message_id:
            try:
                original_message = await interaction.channel.fetch_message(self.original_interaction_message_id)
                view = discord.ui.View.from_message(original_message)
                for item in view.children:
                    item.disabled = True
                await original_message.edit(view=view)
            except (discord.errors.NotFound, AttributeError):
                pass # 如果找不到原始消息或视图，则忽略

        await interaction.response.send_message("✅ 指令已接收！正在後台為您處理世界聖經...", ephemeral=True)
        asyncio.create_task(
            self.cog._background_process_canon(
                interaction=interaction,
                content_text=self.canon_text.value,
                is_setup_flow=self.is_setup_flow
            )
        )
# 類別：世界圣经贴上文字弹出视窗 (v46.2 - 流程串联)

# 類別：角色设定弹出视窗 (v46.2 - 流程串联)
class CharacterSettingsModal(discord.ui.Modal):
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

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) CharacterSettingsModal submitted for profile_type: '{self.profile_type}', is_setup_flow: {self.is_setup_flow}")
        
        # 禁用旧视图的按钮
        if self.original_interaction_message_id:
            try:
                original_message = await interaction.channel.fetch_message(self.original_interaction_message_id)
                view = discord.ui.View.from_message(original_message)
                for item in view.children:
                    item.disabled = True
                await original_message.edit(view=view)
            except (discord.errors.NotFound, AttributeError):
                pass

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
# 類別：角色设定弹出视窗 (v46.2 - 流程串联)

# 類別：世界观设定弹出视窗 (v46.2 - 流程串联)
class WorldSettingsModal(discord.ui.Modal):
    def __init__(self, cog: "BotCog", current_world: str, is_setup_flow: bool = False, original_interaction_message_id: int = None):
        super().__init__(title="步驟 1/3: 世界觀設定", timeout=600.0)
        self.cog = cog
        self.is_setup_flow = is_setup_flow
        self.original_interaction_message_id = original_interaction_message_id
        self.world_settings = discord.ui.TextInput(label="世界觀核心原則", style=discord.TextStyle.paragraph, max_length=4000, default=current_world, placeholder="請描述這個世界的基本規則...")
        self.add_item(self.world_settings)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) WorldSettingsModal submitted. is_setup_flow: {self.is_setup_flow}")
        
        # 禁用旧视图的按钮
        if self.original_interaction_message_id:
            try:
                original_message = await interaction.channel.fetch_message(self.original_interaction_message_id)
                view = discord.ui.View.from_message(original_message)
                for item in view.children:
                    item.disabled = True
                await original_message.edit(view=view)
            except (discord.errors.NotFound, AttributeError):
                pass
        
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
# 類別：世界观设定弹出视窗 (v46.2 - 流程串联)




class ResponseStyleModal(discord.ui.Modal, title="自訂 AI 回覆風格"):
    response_style = discord.ui.TextInput(label="回覆風格指令", style=discord.TextStyle.paragraph, placeholder="在此處定義 AI 的敘事和對話風格...", required=True, max_length=4000)
    def __init__(self, cog: "BotCog", current_style: str):
        super().__init__()
        self.cog = cog
        self.response_style.default = current_style
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

class ForceRestartView(discord.ui.View):
    def __init__(self, *, cog: "BotCog"):
        super().__init__(timeout=180.0)
        self.cog = cog
        self.original_interaction_user_id = None
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.original_interaction_user_id:
            await interaction.response.send_message("你無法操作不屬於你的指令。", ephemeral=True)
            return False
        return True
    @discord.ui.button(label="強制終止並重新開始", style=discord.ButtonStyle.danger)
    async def force_restart(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        for item in self.children: item.disabled = True
        await interaction.edit_original_response(content="正在強制終止舊流程並為您重置所有資料，請稍候...", view=self)
        await self.cog.start_reset_flow(interaction)
        self.stop()
    @discord.ui.button(label="取消本次操作", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="操作已取消。", view=None)
        self.stop()

class ConfirmStartView(discord.ui.View):
    def __init__(self, *, cog: "BotCog"):
        super().__init__(timeout=180.0)
        self.cog = cog
        self.original_interaction_user_id = None
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.original_interaction_user_id:
            await interaction.response.send_message("你無法操作不屬於你的指令。", ephemeral=True)
            return False
        return True
    @discord.ui.button(label="【確認重置並開始】", style=discord.ButtonStyle.danger, custom_id="confirm_start")
    async def confirm_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.cog.setup_locks.add(str(interaction.user.id))
        await interaction.response.defer(ephemeral=True)
        for item in self.children: item.disabled = True
        await interaction.edit_original_response(content="正在為您重置所有資料，請稍候...", view=self)
        await self.cog.start_reset_flow(interaction)
        self.stop()
    @discord.ui.button(label="取消", style=discord.ButtonStyle.secondary, custom_id="cancel_start")
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="操作已取消。", view=None)
        self.stop()
    async def on_timeout(self):
        for item in self.children: item.disabled = True

class SettingsChoiceView(discord.ui.View):
    def __init__(self, cog: "BotCog"):
        super().__init__(timeout=180)
        self.cog = cog
    @discord.ui.button(label="👤 使用者角色設定", style=discord.ButtonStyle.primary, emoji="👤")
    async def user_settings_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        ai_instance = await self.cog.get_or_create_ai_instance(str(interaction.user.id))
        profile_data = ai_instance.profile.user_profile.model_dump() if ai_instance and ai_instance.profile else {}
        modal = CharacterSettingsModal(self.cog, title="👤 使用者角色設定", profile_data=profile_data, profile_type='user', is_setup_flow=False)
        await interaction.response.send_modal(modal)
    @discord.ui.button(label="❤️ AI 戀人設定", style=discord.ButtonStyle.success, emoji="❤️")
    async def ai_settings_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        ai_instance = await self.cog.get_or_create_ai_instance(str(interaction.user.id))
        profile_data = ai_instance.profile.ai_profile.model_dump() if ai_instance and ai_instance.profile else {}
        modal = CharacterSettingsModal(self.cog, title="❤️ AI 戀人設定", profile_data=profile_data, profile_type='ai', is_setup_flow=False)
        await interaction.response.send_modal(modal)
    @discord.ui.button(label="🌍 世界觀設定", style=discord.ButtonStyle.secondary, emoji="🌍")
    async def world_settings_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        ai_instance = await self.cog.get_or_create_ai_instance(str(interaction.user.id))
        world_settings = ai_instance.profile.world_settings if ai_instance and ai_instance.profile else ""
        modal = WorldSettingsModal(self.cog, current_world=world_settings, is_setup_flow=False)
        await interaction.response.send_modal(modal)

class ConfirmEditView(discord.ui.View):
    def __init__(self, *, cog: "BotCog", target_type: Literal['user', 'ai', 'npc'], target_key: str, new_description: str):
        super().__init__(timeout=300.0)
        self.cog = cog
        self.target_type = target_type
        self.target_key = target_key
        self.new_description = new_description
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
    @discord.ui.button(label="❌ 取消", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="操作已取消。", view=None, embed=None)
        self.stop()

class ProfileEditModal(discord.ui.Modal):
    edit_instruction = discord.ui.TextInput(label="修改指令", style=discord.TextStyle.paragraph, placeholder="請用自然語言描述您想如何修改這個角色...", required=True, max_length=1000)
    def __init__(self, *, cog: "BotCog", target_type: Literal['user', 'ai', 'npc'], target_key: str, display_name: str, original_description: str):
        super().__init__(title=f"編輯角色：{display_name}")
        self.cog = cog
        self.target_type = target_type
        self.target_key = target_key
        self.display_name = display_name
        self.original_description = original_description
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

class ConfirmAndEditView(discord.ui.View):
    def __init__(self, *, cog: "BotCog", target_type: Literal['user', 'ai', 'npc'], target_key: str, display_name: str, original_description: str):
        super().__init__(timeout=300.0)
        self.cog = cog
        self.target_type = target_type
        self.target_key = target_key
        self.display_name = display_name
        self.original_description = original_description
    @discord.ui.button(label="✍️ 點此開始編輯", style=discord.ButtonStyle.success)
    async def edit(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = ProfileEditModal(cog=self.cog, target_type=self.target_type, target_key=self.target_key, display_name=self.display_name, original_description=self.original_description)
        await interaction.response.send_modal(modal)
        self.stop()
        await interaction.message.edit(view=self)
    async def on_timeout(self):
        for item in self.children: item.disabled = True

class NpcEditSelect(discord.ui.Select):
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

class EditProfileRootView(discord.ui.View):
    def __init__(self, cog: "BotCog", original_user_id: int):
        super().__init__(timeout=180)
        self.cog = cog
        self.original_user_id = original_user_id
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.original_user_id:
            await interaction.response.send_message("你無法操作不屬於你的指令。", ephemeral=True)
            return False
        return True
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
    @discord.ui.button(label="👤 編輯我的檔案", style=discord.ButtonStyle.primary)
    async def edit_user(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._send_profile_for_editing(interaction, 'user')
    @discord.ui.button(label="❤️ 編輯 AI 戀人檔案", style=discord.ButtonStyle.success)
    async def edit_ai(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._send_profile_for_editing(interaction, 'ai')
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

class CreateTagModal(discord.ui.Modal, title="創建新版本 (Tag)"):
    version = discord.ui.TextInput(label="版本號", placeholder="v1.2.1", required=True)
    description = discord.ui.TextInput(label="版本描述 (可選)", style=discord.TextStyle.paragraph, placeholder="簡短描述此版本的變更", required=False)
    def __init__(self, view: "VersionControlView"):
        super().__init__()
        self.view = view
    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        success, message = await self.view.cog._git_create_tag(self.version.value, self.description.value)
        if success:
            await interaction.followup.send(f"✅ **版本創建成功！**\nTag: `{self.version.value}`。", ephemeral=True)
            await self.view.update_message(interaction)
        else:
            await interaction.followup.send(f"❌ **版本創建失敗！**\n```\n{message}\n```", ephemeral=True)

class RollbackSelect(discord.ui.Select):
    def __init__(self, tags: List[str]):
        options = [discord.SelectOption(label=tag, value=tag) for tag in tags] or [discord.SelectOption(label="沒有可用的版本", value="disabled")]
        super().__init__(placeholder="選擇要回退到的版本...", options=options, disabled=not tags)
    async def callback(self, interaction: discord.Interaction):
        await self.view.show_rollback_confirmation(interaction, self.values[0])

class VersionControlView(discord.ui.View):
    def __init__(self, cog: "BotCog", original_user_id: int):
        super().__init__(timeout=300)
        self.cog = cog
        self.original_user_id = original_user_id
        self.selected_rollback_version = None
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.original_user_id:
            await interaction.response.send_message("你無法操作此面板。", ephemeral=True)
            return False
        return True
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
    @discord.ui.button(label="🔄 刷新", style=discord.ButtonStyle.success, custom_id="vc_refresh")
    async def refresh_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        await self.update_message(interaction)
    @discord.ui.button(label="➕ 創建新版本", style=discord.ButtonStyle.primary, custom_id="vc_create_tag")
    async def create_tag_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(CreateTagModal(self))
    @discord.ui.button(label="⏪ 回退版本", style=discord.ButtonStyle.secondary, custom_id="vc_rollback")
    async def rollback_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        await self.update_message(interaction, show_select=True)
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

# 類別：機器人核心功能集 (Cog)
class BotCog(commands.Cog):
    def __init__(self, bot: "AILoverBot"):
        self.bot = bot
        self.ai_instances: dict[str, AILover] = {}
        self.setup_locks: set[str] = set()
        
        self.main_response_graph = create_main_response_graph()
        self.setup_graph = create_setup_graph()
        
        self.connection_watcher.start()

    def cog_unload(self):
        self.connection_watcher.cancel()

    async def get_or_create_ai_instance(self, user_id: str, is_setup_flow: bool = False) -> AILover | None:
        if user_id in self.ai_instances:
            return self.ai_instances[user_id]
        
        logger.info(f"使用者 {user_id} 沒有活躍的 AI 實例，嘗試創建...")
        ai_instance = AILover(user_id)
        
        if await ai_instance.initialize():
            logger.info(f"為使用者 {user_id} 成功創建並初始化 AI 實例。")
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

    # Git 操作輔助函式 (略)
    async def _run_git_command(self, command: List[str]) -> Tuple[bool, str]:
        try:
            process = await asyncio.to_thread(subprocess.run, command, capture_output=True, text=True, encoding='utf-8', check=True, cwd=PROJ_DIR)
            return True, process.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.strip() or e.stdout.strip()
            logger.error(f"Git指令 '{' '.join(command)}' 執行失敗: {error_message}")
            return False, error_message
        except Exception as e: return False, str(e)
    async def _git_get_current_version(self) -> Tuple[bool, str]:
        return await self._run_git_command(["git", "describe", "--tags", "--always"])
    async def _git_get_remote_tags(self) -> Tuple[bool, List[str]]:
        await self._run_git_command(["git", "fetch", "--tags", "--force"])
        success, msg = await self._run_git_command(["git", "tag", "-l", "--sort=-v:refname"])
        return (True, msg.splitlines()) if success else (False, [msg])
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
    async def _git_rollback_version(self, version: str) -> Tuple[bool, str]:
        logger.info(f"管理員觸發版本回退至: {version}")
        success, msg = await self._run_git_command(["git", "checkout", f"tags/{version}"])
        if not success: return False, f"Checkout失敗: {msg}"
        pip_command = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        try:
            await asyncio.to_thread(subprocess.run, pip_command, check=True, capture_output=True)
        except Exception as e: return False, f"安裝依賴項失敗: {e}"
        if self.bot.shutdown_event: self.bot.shutdown_event.set()
        return True, "回退指令已發送，伺服器正在重啟。"

    @tasks.loop(seconds=240)
    async def connection_watcher(self):
        try:
            await self.bot.wait_until_ready()
            if math.isinf(self.bot.latency): logger.critical("【重大錯誤】與 Discord 的 WebSocket 連線已中斷！")
            else: await self.bot.change_presence(activity=discord.Game(name="與你共度時光"))
        except Exception as e: logger.error(f"【健康檢查】任務中發生未預期的錯誤: {e}", exc_info=True)
    @connection_watcher.before_loop
    async def before_connection_watcher(self):
        await self.bot.wait_until_ready()
        logger.info("【健康檢查 & Keep-Alive】背景任務已啟動。")

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
        logger.info(f"[{user_id}] 响应条件满足，启动 LangGraph 對話流程...")
        async with message.channel.typing():
            try:
                chat_history_manager = ai_instance.session_histories.setdefault(user_id, ChatMessageHistory())
                current_messages = chat_history_manager.messages.copy()
                current_messages.append(HumanMessage(content=user_input))
                initial_state = ConversationGraphState(user_id=user_id, ai_core=ai_instance, messages=current_messages)
                final_state = await self.main_response_graph.ainvoke(initial_state)
                response = final_state.get('final_output')
                if response and response.strip() and response != "（...）":
                    for i in range(0, len(response), 2000): await message.channel.send(response[i:i+2000])
                else:
                    logger.error(f"LangGraph 為使用者 {user_id} 返回了空的或無效的回應。")
                    await message.channel.send("（抱歉，我好像突然斷線了...）")
            except Exception as e:
                logger.error(f"處理使用者 {user_id} 的 LangGraph 聊天流程時發生異常: {e}", exc_info=True)
                await message.channel.send(f"處理您的訊息時發生了一個嚴重的內部錯誤: `{type(e).__name__}`")

    # finalize_setup (v46.0 - 適配持久化視圖)
    async def finalize_setup(self, interaction: discord.Interaction, canon_text: Optional[str] = None):
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] (UI Event) finalize_setup 被觸發。Canon provided: {bool(canon_text)}")
        
        ai_instance = await self.get_or_create_ai_instance(user_id, is_setup_flow=True)
        if not ai_instance or not ai_instance.profile:
            logger.error(f"[{user_id}] 在 finalize_setup 中獲取 AI 核心失敗。")
            await interaction.followup.send("❌ 錯誤：無法從資料庫加載您的基礎設定以進行創世。", ephemeral=True)
            self.setup_locks.discard(user_id)
            return

        try:
            await interaction.followup.send("🚀 **正在為您執行最終創世...**\n這可能需要一到兩分鐘，請稍候。", ephemeral=True)
            
            logger.info(f"[{user_id}] /start 流程：正在強制初始化 AI 核心組件...")
            await ai_instance._configure_pre_requisites()
            
            initial_state = SetupGraphState(user_id=user_id, ai_core=ai_instance, canon_text=canon_text)
            logger.info(f"[{user_id}] /start 流程：準備調用 LangGraph 設定圖...")
            final_state = await self.setup_graph.ainvoke(initial_state)
            logger.info(f"[{user_id}] /start 流程：LangGraph 設定圖執行完畢。")
            
            opening_scene = final_state.get('opening_scene')
            
            if not opening_scene:
                 opening_scene = (f"在一片柔和的光芒中，你和 {ai_instance.profile.ai_profile.name} 發現自己身處於一個寧靜的空間裡...")

            dm_channel = await interaction.user.create_dm()
            
            logger.info(f"[{user_id}] /start 流程：正在向使用者私訊發送開場白...")
            for i in range(0, len(opening_scene), 2000):
                await dm_channel.send(opening_scene[i:i+2000])
            logger.info(f"[{user_id}] /start 流程：開場白發送完畢。設定流程成功結束。")

        except Exception as e:
            logger.error(f"[{user_id}] 在 LangGraph 設定流程中發生嚴重錯誤: {e}", exc_info=True)
            try:
                await interaction.followup.send(f"❌ **錯誤**：在執行最終設定時發生了未預期的嚴重錯誤: {e}", ephemeral=True)
            except discord.errors.NotFound:
                await interaction.user.send(f"❌ **錯誤**：在執行最終設定時發生了未預期的嚴重錯誤: {e}")
        finally:
            self.setup_locks.discard(user_id)
    # finalize_setup (v46.0 - 適配持久化視圖)

    async def _background_process_canon(self, interaction: discord.Interaction, content_text: str, is_setup_flow: bool):
        user_id = str(interaction.user.id)
        user = self.bot.get_user(interaction.user.id) or await self.bot.fetch_user(interaction.user.id)
        try:
            ai_instance = await self.get_or_create_ai_instance(user_id, is_setup_flow=is_setup_flow)
            if not ai_instance:
                await user.send("❌ **處理失敗！**")
                return
            if len(content_text) > 5000:
                await user.send("⏳ **請注意：**\n您提供的世界聖經內容較多，處理可能需要 **幾分鐘** 的時間。")
            if not ai_instance.vector_store:
                await ai_instance._configure_pre_requisites()
            chunk_count = await ai_instance.add_canon_to_vector_store(content_text)
            
            if is_setup_flow:
                await interaction.followup.send("✅ 世界聖經已提交！正在為您啟動最終創世...", ephemeral=True)
                asyncio.create_task(self.finalize_setup(interaction, content_text))
                return

            await user.send(f"✅ **世界聖經已向量化！**\n內容已被分解為 **{chunk_count}** 個知識片段。\n\n🧠 AI 正在進行更深層的智能解析...")
            await ai_instance.parse_and_create_lore_from_canon(interaction, content_text, is_setup_flow)
            await user.send("✅ **智能解析完成！**")
        except Exception as e:
            logger.error(f"[{user_id}] 背景處理世界聖經時發生錯誤: {e}", exc_info=True)
            await user.send(f"❌ **處理失敗！**\n發生了嚴重錯誤: `{type(e).__name__}`")
    
    # 函式：開始重置流程 (v47.0 - 徹底異步化)
    async def start_reset_flow(self, interaction: discord.Interaction):
        user_id = str(interaction.user.id)
        try:
            logger.info(f"[{user_id}] 後台重置任務開始...")
            if user_id in self.ai_instances:
                await self.ai_instances.pop(user_id).shutdown()
                gc.collect()
                await asyncio.sleep(1.5)
            async with AsyncSessionLocal() as session:
                await session.execute(delete(MemoryData).where(MemoryData.user_id == user_id))
                await session.execute(delete(Lore).where(Lore.user_id == user_id))
                await session.execute(delete(UserData).where(UserData.user_id == user_id))
                await session.commit()
            
            vector_store_path = Path(f"./data/vector_stores/{user_id}")
            if vector_store_path.exists():
                await asyncio.to_thread(shutil.rmtree, vector_store_path)
            
            view = StartSetupView(cog=self)
            await interaction.followup.send(
                content="✅ 重置完成！請點擊下方按鈕開始全新的設定流程。", 
                view=view, 
                ephemeral=True
            )
        except Exception as e:
            logger.error(f"[{user_id}] 後台重置任務失敗: {e}", exc_info=True)
            await interaction.followup.send(f"執行重置時發生未知的嚴重錯誤: {e}", ephemeral=True)
        finally:
            self.setup_locks.discard(user_id)
    # 函式：開始重置流程 (v47.0 - 徹底異步化)

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
            return
        view = ConfirmStartView(cog=self)
        view.original_interaction_user_id = interaction.user.id
        await interaction.response.send_message("⚠️ **警告** ⚠️\n您確定要開始一段全新的冒險嗎？\n這將會**永久刪除**您當前所有的角色、世界、記憶和進度。", view=view, ephemeral=True)

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

    @app_commands.command(name="response_style", description="客製化 AI 的回覆風格")
    async def response_style(self, interaction: discord.Interaction):
        if not isinstance(interaction.channel, discord.DMChannel):
            await interaction.response.send_message("此指令只能在私訊頻道中使用。", ephemeral=True)
            return
        ai_instance = await self.get_or_create_ai_instance(str(interaction.user.id))
        if not ai_instance or not ai_instance.profile:
            await interaction.response.send_message("請先使用 `/start` 指令進行初始設定。", ephemeral=True)
            return
        current_style = ai_instance.profile.response_style_prompt or "..."
        await interaction.response.send_modal(ResponseStyleModal(self, current_style))

    @app_commands.command(name="edit_profile", description="編輯您或任何角色的個人檔案。")
    async def edit_profile(self, interaction: discord.Interaction):
        await interaction.response.send_message("請選擇您想編輯的角色檔案：", view=EditProfileRootView(self, interaction.user.id), ephemeral=True)
        
    @app_commands.command(name="set_canon_text", description="通過貼上文字來設定您的世界聖經")
    async def set_canon_text(self, interaction: discord.Interaction):
        await interaction.response.send_modal(WorldCanonPasteModal(self, is_setup_flow=False))

    @app_commands.command(name="set_canon_file", description="通過上傳 .txt 檔案來設定您的世界聖經")
    @app_commands.describe(file="請上傳一個 .txt 格式的檔案，最大 5MB。")
    async def set_canon_file(self, interaction: discord.Interaction, file: discord.Attachment):
        if not file.filename.lower().endswith('.txt'):
            await interaction.response.send_message("❌ 檔案格式錯誤！", ephemeral=True)
            return
        try:
            content_text = (await file.read()).decode('utf-8')
            await interaction.response.send_message("✅ 檔案已接收！正在後台處理...", ephemeral=True)
            asyncio.create_task(self._background_process_canon(interaction, content_text, is_setup_flow=False))
        except Exception as e:
            logger.error(f"處理上傳的世界聖經檔案時發生錯誤: {e}", exc_info=True)
            await interaction.response.send_message(f"讀取檔案時發生錯誤。", ephemeral=True)

    # 管理員指令 (略)
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
    @app_commands.command(name="admin_force_update", description="[管理員] 強制從 GitHub 同步最新程式碼並重啟機器人。")
    @app_commands.check(is_admin)
    async def admin_force_update(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        await interaction.followup.send("✅ **指令已接收！**\n正在背景中為您執行強制同步與重啟...", ephemeral=True)
        asyncio.create_task(self._perform_update_and_restart(interaction))
    async def _perform_update_and_restart(self, interaction: discord.Interaction):
        try:
            await asyncio.sleep(1)
            def run_git_sync(): return subprocess.run(["git", "reset", "--hard", "origin/main"], capture_output=True, text=True, encoding='utf-8', check=False)
            process = await asyncio.to_thread(run_git_sync)
            if process.returncode == 0:
                if settings.ADMIN_USER_ID:
                    try:
                        admin_user = self.bot.get_user(int(settings.ADMIN_USER_ID)) or await self.bot.fetch_user(int(settings.ADMIN_USER_ID))
                        await admin_user.send("✅ **系統更新成功！** 機器人即將重啟。")
                    except Exception as e: logger.error(f"發送更新成功通知給管理員時發生未知錯誤: {e}", exc_info=True)
                await asyncio.sleep(3)
                if self.bot.shutdown_event: self.bot.shutdown_event.set()
            else:
                await interaction.followup.send(f"🔥 **同步失敗！**\n```\n{process.stderr.strip()}\n```", ephemeral=True)
        except Exception as e: logger.error(f"背景任務：執行強制更新時發生未預期錯誤: {e}", exc_info=True)
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
    @app_commands.command(name="admin_push_log", description="[管理員] 強制將最新的100條LOG推送到GitHub倉庫。")
    @app_commands.check(is_admin)
    async def admin_push_log(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.push_log_to_github_repo(interaction)
    async def push_log_to_github_repo(self, interaction: Optional[discord.Interaction] = None):
        try:
            log_file_path = PROJ_DIR / "data" / "logs" / "app.log"
            if not log_file_path.is_file():
                if interaction: await interaction.followup.send("❌ **推送失敗**：找不到日誌檔案。", ephemeral=True)
                return
            with open(log_file_path, 'r', encoding='utf-8') as f: latest_lines = f.readlines()[-100:]
            upload_log_path = PROJ_DIR / "latest_log.txt"
            with open(upload_log_path, 'w', encoding='utf-8') as f: f.write(f"### AI Lover Log - {datetime.datetime.now().isoformat()} ###\n\n" + "".join(latest_lines))
            def run_git_commands():
                subprocess.run(["git", "add", str(upload_log_path)], check=True, cwd=PROJ_DIR)
                commit_message = f"docs: Update latest_log.txt at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                subprocess.run(["git", "commit", "-m", commit_message], check=False, cwd=PROJ_DIR)
                subprocess.run(["git", "push", "origin", "main"], check=True, cwd=PROJ_DIR)
            await asyncio.to_thread(run_git_commands)
            if interaction: await interaction.followup.send(f"✅ **LOG 推送成功！**", ephemeral=True)
        except Exception as e:
            if interaction: await interaction.followup.send(f"❌ **推送失敗**：`{e}`", ephemeral=True)
    @app_commands.command(name="admin_version_control", description="[管理員] 打開圖形化版本控制面板。")
    @app_commands.check(is_admin)
    async def admin_version_control(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        view = VersionControlView(cog=self, original_user_id=interaction.user.id)
        embed = await view._build_embed()
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)
    @commands.Cog.listener()
    async def on_app_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.CheckFailure):
            await interaction.response.send_message("你沒有權限使用此指令。", ephemeral=True)
        else:
            logger.error(f"一個應用程式指令發生錯誤: {error}", exc_info=True)
            if not interaction.response.is_done():
                await interaction.response.send_message(f"發生未知錯誤。", ephemeral=True)

# 類別：AI 戀人機器人主體 (v46.0 - 持久化視圖註冊)
class AILoverBot(commands.Bot):
    def __init__(self, shutdown_event: asyncio.Event):
        super().__init__(command_prefix='/', intents=intents, activity=discord.Game(name="與你共度時光"))
        self.shutdown_event = shutdown_event
        self.is_ready_once = False
    
    async def setup_hook(self):
        cog = BotCog(self)
        await self.add_cog(cog)

        # [v46.0 核心修正] 在啟動時註冊所有持久化視圖
        self.add_view(StartSetupView(cog=cog))
        self.add_view(ContinueToUserSetupView(cog=cog))
        self.add_view(ContinueToAiSetupView(cog=cog))
        self.add_view(ContinueToCanonSetupView(cog=cog))
        logger.info("所有持久化 UI 視圖已成功註冊。")
        
        await self.tree.sync()
        logger.info("Discord Bot is ready and commands are synced!")
    
    async def on_ready(self):
        logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
        if not self.is_ready_once:
            self.is_ready_once = True
            if settings.ADMIN_USER_ID:
                try:
                    admin_user = self.get_user(int(settings.ADMIN_USER_ID)) or await self.fetch_user(int(settings.ADMIN_USER_ID))
                    await admin_user.send(f"✅ **系統啟動成功！**")
                    logger.info(f"已成功發送啟動成功通知給管理員。")
                except Exception as e:
                    logger.error(f"發送啟動成功通知給管理員時發生未知錯誤: {e}", exc_info=True)
# 類別：AI 戀人機器人主體 (v46.0 - 持久化視圖註冊)
