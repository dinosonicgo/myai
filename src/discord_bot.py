# discord_bot.py 的中文註釋(v39.0 - 強制更新指令)
# 更新紀錄:
# v39.0 (2025-09-02): [健壯性] 新增了僅限管理員使用的 `/admin_force_update` 指令。此指令允許管理員手動觸發 `git reset --hard origin/main` 並立即重啟應用，以強制同步最新的程式碼，極大地提高了遠程調試和部署更新的效率與可靠性。
# v38.3 (2025-09-02): [BUG修復] 修正了 on_message 函式的縮排錯誤。
# v38.2 (2025-09-02): [災難性BUG修復] 修正了 `ChatMessageHistory` 的導入來源，解決了 LangGraph 流程因 `AttributeError` 而無法啟動的問題。

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
from typing import Optional, Literal, List, Dict, Any
from collections import defaultdict
import os
import sys
import subprocess

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

# 類別：世界聖經上傳彈出視窗
class WorldCanonUploadModal(discord.ui.Modal, title="步驟 4/4: 上傳世界聖經 (可選)"):
    canon_text = discord.ui.TextInput(
        label="請將您的世界觀/角色背景故事貼於此處",
        style=discord.TextStyle.paragraph,
        placeholder="在此貼上您的 .txt 檔案內容... AI 將在創世時參考這些設定。",
        required=True,
        max_length=4000
    )

    def __init__(self, cog: "BotCog"):
        super().__init__(timeout=600.0)
        self.cog = cog

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.cog.finalize_setup(interaction, self.canon_text.value)
# 類別：世界聖經上傳彈出視窗

# 類別：繼續世界聖經上傳視圖
class ContinueToCanonUploadView(discord.ui.View):
    def __init__(self, *, cog: "BotCog", user_id: str):
        super().__init__(timeout=300.0)
        self.cog = cog
        self.user_id = user_id

    @discord.ui.button(label="📝 貼上文本內容", style=discord.ButtonStyle.primary, row=0)
    async def paste_text(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = WorldCanonUploadModal(self.cog)
        await interaction.response.send_modal(modal)
        self.stop()
        await interaction.edit_original_response(view=None)

    @discord.ui.button(label="📄 上傳 .txt 檔案", style=discord.ButtonStyle.primary, row=0)
    async def upload_file(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        for item in self.children:
            item.disabled = True
        await interaction.edit_original_response(view=self)
        finalize_view = FinalizeAfterUploadView(cog=self.cog, user_id=self.user_id)
        guide_message = (
            "好的，請按以下步驟上傳您的檔案：\n\n"
            "1️⃣ **點擊下方的訊息輸入框**。\n"
            "2️⃣ **輸入一個斜線 `/`**，Discord 就會自動彈出一個可用的指令列表。\n"
            "3️⃣ 從列表中找到並**點擊 `📄 /upload_canon` 指令**。\n"
            "4️⃣ 點擊指令附帶的 `file` 參數，就可以選擇您的 `.txt` 檔案並發送了。\n\n"
            "--- \n"
            "上傳成功後，請點擊下方的 **「✅ 我已上傳完畢」** 按鈕繼續。"
        )
        await interaction.followup.send(
            content=guide_message,
            view=finalize_view,
            ephemeral=True
        )
        self.stop()

    @discord.ui.button(label="🚀 跳過並完成設定", style=discord.ButtonStyle.secondary, row=1)
    async def skip_and_finalize(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.cog.finalize_setup(interaction)
        self.stop()
        await interaction.edit_original_response(view=None)

    async def on_timeout(self):
        self.cog.setup_locks.discard(self.user_id)
        for item in self.children:
            item.disabled = True
# 類別：繼續世界聖經上傳視圖

# 類別：上傳後完成設定視圖
class FinalizeAfterUploadView(discord.ui.View):
    def __init__(self, *, cog: "BotCog", user_id: str):
        super().__init__(timeout=600.0)
        self.cog = cog
        self.user_id = user_id

    @discord.ui.button(label="✅ 我已上傳完畢，完成設定", style=discord.ButtonStyle.success)
    async def finalize(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.cog.finalize_setup(interaction)
        self.stop()
        await interaction.edit_original_response(content="正在為您完成最終設定...", view=None)

    async def on_timeout(self):
        self.cog.setup_locks.discard(self.user_id)
        for item in self.children:
            item.disabled = True
# 類別：上傳後完成設定視圖

# 類別：角色設定彈出視窗
class CharacterSettingsModal(discord.ui.Modal):
    def __init__(self, cog: "BotCog", title: str, profile_data: dict, profile_type: str, is_setup_flow: bool = False):
        super().__init__(title=title)
        self.cog = cog
        self.profile_type = profile_type
        self.is_setup_flow = is_setup_flow
        
        self.name = discord.ui.TextInput(
            label="名字 (必填)", default=profile_data.get('name', ''), 
            required=True
        )
        self.gender = discord.ui.TextInput(
            label="性別 (必填)", default=profile_data.get('gender', ''), 
            placeholder="男 / 女 / 其他", required=True
        )
        self.description = discord.ui.TextInput(
            label="性格、背景、種族、年齡等綜合描述", style=discord.TextStyle.paragraph, 
            default=profile_data.get('description', ''), required=True, max_length=1000,
            placeholder="請用自然語言描述角色的核心特徵..."
        )
        self.appearance = discord.ui.TextInput(
            label="外觀描述 (髮型/瞳色/身材等)", style=discord.TextStyle.paragraph, 
            default=profile_data.get('appearance', ''), 
            placeholder="請用自然語言描述角色的外觀，例如：她有一頭瀑布般的綠色長髮，琥珀色的眼睛像貓一樣...", 
            required=False, max_length=1000
        )

        self.add_item(self.name)
        self.add_item(self.gender)
        self.add_item(self.description)
        self.add_item(self.appearance)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user_id = str(interaction.user.id)

        ai_instance = await self.cog.get_or_create_ai_instance(user_id, is_setup_flow=self.is_setup_flow)
        if not ai_instance or not ai_instance.profile:
            await interaction.followup.send("錯誤：AI 核心或設定檔案未初始化。", ephemeral=True)
            return
            
        profile_attr = f"{self.profile_type}_profile"
        
        try:
            updated_profile = getattr(ai_instance.profile, profile_attr)

            updated_profile.name = self.name.value
            updated_profile.gender = self.gender.value
            updated_profile.description = self.description.value
            updated_profile.appearance = self.appearance.value
            
            success = await ai_instance.update_and_persist_profile({
                profile_attr: updated_profile.model_dump()
            })

            if not success:
                raise Exception("AI 核心更新 profile 失敗。")

            if not self.is_setup_flow:
                await interaction.followup.send(f"✅ **{updated_profile.name}** 的角色設定已成功更新！", ephemeral=True)
            elif self.profile_type == 'user': 
                view = ContinueToAiSetupView(cog=self.cog, user_id=user_id)
                await interaction.followup.send("✅ 您的角色已設定！\n請點擊下方按鈕，為您的 AI 戀人進行設定。", view=view, ephemeral=True)
            elif self.profile_type == 'ai':
                view = ContinueToCanonUploadView(cog=self.cog, user_id=user_id)
                await interaction.followup.send(
                    "✅ AI 戀人基礎設定完成！\n\n"
                    "**下一步是可選的，但強烈推薦：**\n"
                    "您可以上傳一份包含您自訂世界觀、角色背景或故事劇情的「世界聖經」(.txt 內容)，AI 將在創世時完全基於您的設定來生成一切！",
                    view=view,
                    ephemeral=True
                )

        except Exception as e:
            logger.error(f"[{user_id}] 處理角色設定時出錯: {e}", exc_info=True)
            await interaction.followup.send("錯誤：在處理您的設定時遇到問題，請稍後再試。", ephemeral=True)
            return
# 類別：角色設定彈出視窗

# 類別：世界觀設定彈出視窗
class WorldSettingsModal(discord.ui.Modal):
    def __init__(self, cog: "BotCog", current_world: str, is_setup_flow: bool = False):
        super().__init__(title="世界觀設定")
        self.cog = cog
        self.is_setup_flow = is_setup_flow
        self.world_settings = discord.ui.TextInput(
            label="世界觀核心原則", 
            style=discord.TextStyle.paragraph, 
            max_length=4000, 
            default=current_world,
            placeholder="請描述這個世界的基本規則、風格、科技或魔法水平等..."
        )
        self.add_item(self.world_settings)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user_id = str(interaction.user.id)
        
        ai_instance = await self.cog.get_or_create_ai_instance(user_id, is_setup_flow=self.is_setup_flow)
        if not ai_instance:
            await interaction.followup.send("錯誤：無法初始化 AI 核心。", ephemeral=True)
            return

        success = await ai_instance.update_and_persist_profile({
            'world_settings': self.world_settings.value
        })
        
        if not success:
            await interaction.followup.send("錯誤：更新世界觀失敗。", ephemeral=True)
            return
        
        if self.is_setup_flow:
            view = ContinueToUserSetupView(cog=self.cog, user_id=user_id)
            await interaction.followup.send("✅ 世界觀已設定！\n請點擊下方按鈕，開始設定您的個人角色。", view=view, ephemeral=True)
        else:
            await interaction.followup.send("✅ 世界觀設定已成功更新！", ephemeral=True)
# 類別：世界觀設定彈出視窗

# 類別：回覆風格設定彈出視窗
class ResponseStyleModal(discord.ui.Modal, title="自訂 AI 回覆風格"):
    response_style = discord.ui.TextInput(
        label="回覆風格指令",
        style=discord.TextStyle.paragraph,
        placeholder="在此處定義 AI 的敘事和對話風格...",
        required=True,
        max_length=4000
    )

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

        success = await ai_instance.update_and_persist_profile({
            'response_style_prompt': self.response_style.value
        })

        if success:
            await interaction.followup.send("✅ AI 回覆風格已成功更新！新的風格將在下次對話時生效。", ephemeral=True)
        else:
            await interaction.followup.send("錯誤：更新 AI 回覆風格失敗。", ephemeral=True)
# 類別：回覆風格設定彈出視窗

# 類別：強制重啟視圖
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
        for item in self.children:
            item.disabled = True
        await interaction.edit_original_response(content="正在強制終止舊流程並為您重置所有資料，請稍候...", view=self)
        await self.cog.start_reset_flow(interaction)
        self.stop()

    @discord.ui.button(label="取消本次操作", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="操作已取消。舊有的設定流程（如果存在）可能會繼續或最終超時。", view=None)
        self.stop()
# 類別：強制重啟視圖

# 類別：確認開始視圖
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
        for item in self.children:
            item.disabled = True
        await interaction.edit_original_response(content="正在為您重置所有資料，請稍候...", view=self)
        await self.cog.start_reset_flow(interaction)
        self.stop()

    @discord.ui.button(label="取消", style=discord.ButtonStyle.secondary, custom_id="cancel_start")
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="操作已取消。", view=None)
        self.stop()

    async def on_timeout(self):
        for item in self.children:
            item.disabled = True
# 類別：確認開始視圖

# 類別：開始設定視圖
class StartSetupView(discord.ui.View):
    def __init__(self, *, cog: "BotCog", user_id: str):
        super().__init__(timeout=300.0)
        self.cog = cog
        self.user_id = user_id

    @discord.ui.button(label="🚀 開始設定", style=discord.ButtonStyle.success)
    async def start_setup_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        world_modal = WorldSettingsModal(self.cog, current_world="這是一個魔法與科技交織的幻想世界。", is_setup_flow=True)
        await interaction.response.send_modal(world_modal)
        self.stop()
        await interaction.edit_original_response(view=None)
        
    async def on_timeout(self):
        self.cog.setup_locks.discard(self.user_id)
        for item in self.children:
            item.disabled = True
# 類別：開始設定視圖

# 類別：繼續使用者設定視圖
class ContinueToUserSetupView(discord.ui.View):
    def __init__(self, *, cog: "BotCog", user_id: str):
        super().__init__(timeout=300.0)
        self.cog = cog
        self.user_id = user_id

    @discord.ui.button(label="下一步：設定您的角色", style=discord.ButtonStyle.primary)
    async def continue_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        ai_instance = await self.cog.get_or_create_ai_instance(str(interaction.user.id), is_setup_flow=True)
        profile_data = ai_instance.profile.user_profile.model_dump() if ai_instance and ai_instance.profile else {}
        modal = CharacterSettingsModal(self.cog, title="步驟 2/3: 您的角色設定", profile_data=profile_data, profile_type='user', is_setup_flow=True)
        await interaction.response.send_modal(modal)
        await interaction.edit_original_response(view=None)

    async def on_timeout(self):
        self.cog.setup_locks.discard(self.user_id)
        for item in self.children:
            item.disabled = True
# 類別：繼續使用者設定視圖

# 類別：繼續 AI 設定視圖
class ContinueToAiSetupView(discord.ui.View):
    def __init__(self, *, cog: "BotCog", user_id: str):
        super().__init__(timeout=300.0)
        self.cog = cog
        self.user_id = user_id

    @discord.ui.button(label="最後一步：設定 AI 戀人", style=discord.ButtonStyle.primary)
    async def continue_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        ai_instance = await self.cog.get_or_create_ai_instance(str(interaction.user.id), is_setup_flow=True)
        profile_data = ai_instance.profile.ai_profile.model_dump() if ai_instance and ai_instance.profile else {}
        modal = CharacterSettingsModal(self.cog, title="步驟 3/3: AI 戀人設定", profile_data=profile_data, profile_type='ai', is_setup_flow=True)
        await interaction.response.send_modal(modal)
        await interaction.edit_original_response(view=None)

    async def on_timeout(self):
        self.cog.setup_locks.discard(self.user_id)
        for item in self.children:
            item.disabled = True
# 類別：繼續 AI 設定視圖

# 類別：設定選項視圖
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
# 類別：設定選項視圖

# 類別：確認編輯視圖
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
                await ai_instance.update_and_persist_profile({
                    profile_attr: profile_obj.model_dump()
                })
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
# 類別：確認編輯視圖

# 類別：角色編輯彈出視窗
class ProfileEditModal(discord.ui.Modal):
    edit_instruction = discord.ui.TextInput(
        label="修改指令",
        style=discord.TextStyle.paragraph,
        placeholder="請用自然語言描述您想如何修改這個角色...",
        required=True,
        max_length=1000,
    )

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
        
        ai_instance = await self.cog.get_or_create_ai_instance(user_id)
        if not ai_instance or not ai_instance.profile:
            await interaction.followup.send("錯誤：無法初始化 AI 核心來處理您的請求。", ephemeral=True)
            return

        try:
            zero_instruction_str = ai_instance.zero_instruction_template.replace(
                "{username}", ai_instance.profile.user_profile.name
            ).replace(
                "{ai_name}", ai_instance.profile.ai_profile.name
            )

            prompt = ai_instance.get_profile_rewriting_prompt()
            rewriter_llm = ai_instance.gm_model | StrOutputParser()
            rewriter_chain = prompt | rewriter_llm

            new_description = await ai_instance.ainvoke_with_rotation(rewriter_chain, {
                "zero_instruction": zero_instruction_str,
                "username": ai_instance.profile.user_profile.name or "使用者",
                "ai_name": ai_instance.profile.ai_profile.name or "AI",
                "original_description": self.original_description,
                "edit_instruction": self.edit_instruction.value
            })
            
            if not new_description or not new_description.strip():
                logger.warning(f"[{user_id}] 角色描述重寫鏈為角色 {self.display_name} 返回了空內容，可能觸發內容審查。")
                await interaction.followup.send("AI 在生成新的角色描述時遇到了困難，返回了空的內容。這可能是由於觸發了內容安全策略，請嘗試調整您的修改指令。", ephemeral=True)
                return

            embed = Embed(title=f"✍️ 角色檔案預覽：{self.display_name}", color=discord.Color.gold())
            embed.add_field(name="【原始描述】", value=f"```{self.original_description[:1000] if self.original_description else '（無）'}```", inline=False)
            embed.add_field(name="【您的指令】", value=f"```{self.edit_instruction.value[:1000]}```", inline=False)
            embed.add_field(name="【AI 生成的新描述】", value=f"```{new_description[:1000]}```", inline=False)
            embed.set_footer(text="請確認新描述是否符合您的期望。")

            view = ConfirmEditView(
                cog=self.cog,
                target_type=self.target_type,
                target_key=self.target_key,
                new_description=new_description
            )
            await interaction.followup.send("AI 已根據您的指令生成了新的角色描述，請預覽並確認：", embed=embed, view=view, ephemeral=True)

        except Exception as e:
            logger.error(f"重寫角色 {self.display_name} 描述時發生未預期錯誤: {e}", exc_info=True)
            await interaction.followup.send("在處理您的編輯指令時，AI 發生了一個嚴重的內部錯誤，請稍後再試。管理員已收到錯誤報告。", ephemeral=True)
# 類別：角色編輯彈出視窗

# 類別：NPC 編輯選擇器
class NpcEditSelect(discord.ui.Select):
    def __init__(self, cog: "BotCog", all_npcs: List[Lore]):
        self.cog = cog
        self.all_npcs = {npc.key: npc for npc in all_npcs}
        
        options = []
        for npc in all_npcs:
            npc_name = npc.content.get('name', '未知名稱')
            location_path = npc.content.get('location_path', [])
            location_display = " > ".join(location_path) if location_path else (npc.content.get('location') or '未知')
            
            label = npc_name[:100]
            description = f"位於: {location_display}"[:100]
            value = npc.key[:100]
            
            options.append(discord.SelectOption(label=label, description=description, value=value))

        super().__init__(placeholder="選擇一位您想編輯的 NPC...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        selected_key = self.values[0]
        lore = self.all_npcs.get(selected_key)
        
        if not lore:
            await interaction.response.send_message("錯誤：找不到所選的NPC資料。", ephemeral=True)
            return
            
        profile_data = lore.content
        display_name = profile_data.get('name', selected_key.split(' > ')[-1])
        original_description = profile_data.get('description', '')

        modal = ProfileEditModal(
            cog=self.cog,
            target_type='npc',
            target_key=selected_key,
            display_name=display_name,
            original_description=original_description
        )
        await interaction.response.send_modal(modal)
        
        self.disabled = True
        await interaction.edit_original_response(view=self.view)
# 類別：NPC 編輯選擇器

# 函式：創建角色檔案 Embed
def _create_profile_embed(profile: CharacterProfile, title_prefix: str) -> Embed:
    embed = Embed(title=f"{title_prefix}：{profile.name}", color=discord.Color.blue())
    
    base_info = [
        f"**性別:** {profile.gender or '未設定'}",
        f"**年齡:** {profile.age or '未知'}",
        f"**種族:** {profile.race or '未知'}"
    ]
    embed.add_field(name="基礎資訊", value="\n".join(base_info), inline=False)

    if profile.description:
        embed.add_field(name="📜 核心描述", value=f"```{profile.description[:1000]}```", inline=False)
    
    if profile.appearance:
        embed.add_field(name="🎨 外觀總覽", value=f"```{profile.appearance[:1000]}```", inline=False)
        
    if profile.appearance_details:
        details_str = "\n".join([f"- {k}: {v}" for k, v in profile.appearance_details.items()])
        embed.add_field(name="✨ 外觀細節", value=details_str, inline=True)

    if profile.equipment:
        embed.add_field(name="⚔️ 當前裝備", value="、".join(profile.equipment), inline=True)
        
    if profile.skills:
        embed.add_field(name="🌟 掌握技能", value="、".join(profile.skills), inline=True)

    return embed
# 函式：創建角色檔案 Embed

# 類別：確認並編輯視圖 (用於 /edit_profile)
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
        modal = ProfileEditModal(
            cog=self.cog,
            target_type=self.target_type,
            target_key=self.target_key,
            display_name=self.display_name,
            original_description=self.original_description
        )
        await interaction.response.send_modal(modal)
        self.stop()
        await interaction.message.edit(view=self)

    async def on_timeout(self):
        for item in self.children:
            item.disabled = True
# 類別：確認並編輯視圖 (用於 /edit_profile)

# 類別：NPC 編輯選擇器
class NpcEditSelect(discord.ui.Select):
    def __init__(self, cog: "BotCog", all_npcs: List[Lore]):
        self.cog = cog
        self.all_npcs = {npc.key: npc for npc in all_npcs}
        
        options = []
        for lore in all_npcs:
            content = lore.content
            name = content.get('name', '未知名稱')
            description_part = (content.get('description', '未知')[:50] + '...') if content.get('description') else '未知'
            
            label = name[:100]
            description = description_part[:100]
            value = lore.key[:100]
            
            options.append(discord.SelectOption(label=label, description=description, value=value))

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
        view = ConfirmAndEditView(
            cog=self.cog,
            target_type='npc',
            target_key=selected_key,
            display_name=profile.name,
            original_description=profile.description
        )
        
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)
        
        self.disabled = True
        await interaction.edit_original_response(view=self.view)
# 類別：NPC 編輯選擇器

# 類別：編輯角色檔案根視圖
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
            
        if target_type == 'user':
            profile = ai_instance.profile.user_profile
            title_prefix = "👤 您的角色檔案"
        else: # 'ai'
            profile = ai_instance.profile.ai_profile
            title_prefix = "❤️ AI 戀人檔案"
        
        embed = _create_profile_embed(profile, title_prefix)
        view = ConfirmAndEditView(
            cog=self.cog,
            target_type=target_type,
            target_key=profile.name,
            display_name=profile.name,
            original_description=profile.description or ""
        )
        
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
# 類別：編輯角色檔案根視圖

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
            ai_instance.profile = UserProfile(
                user_id=user_id,
                user_profile=CharacterProfile(name=""),
                ai_profile=CharacterProfile(name=""),
            )
            self.ai_instances[user_id] = ai_instance
            return ai_instance
        else:
            logger.warning(f"為使用者 {user_id} 初始化 AI 實例失敗（資料庫中可能無記錄）。")
            return None

    @tasks.loop(seconds=240)
    async def connection_watcher(self):
        try:
            await self.bot.wait_until_ready()
            latency = self.bot.latency
            if math.isinf(latency):
                logger.critical("【重大錯誤】與 Discord 的 WebSocket 連線已中斷！")
            else:
                await self.bot.change_presence(activity=discord.Game(name="與你共度時光"))
        except asyncio.CancelledError:
            logger.info("【健康檢查】任務被正常取消。")
            raise
        except Exception as e:
            logger.error(f"【健康檢查】任務中發生未預期的錯誤: {e}", exc_info=True)

    @connection_watcher.before_loop
    async def before_connection_watcher(self):
        await self.bot.wait_until_ready()
        logger.info("【健康檢查 & Keep-Alive】背景任務已啟動。")

    @connection_watcher.after_loop
    async def after_connection_watcher(self):
        if self.connection_watcher.is_being_cancelled():
            logger.info("【健康檢查 & Keep-Alive】背景任務已正常停止。")
        else:
            logger.error(f"【健康檢查 & Keep-Alive】背景任務因未處理的錯誤而意外終止！")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not isinstance(message.channel, discord.DMChannel):
            return
        
        ctx = await self.bot.get_context(message)
        if ctx.valid:
            return
        
        user_id = str(message.author.id)
        
        ai_instance = await self.get_or_create_ai_instance(user_id)
        if not ai_instance:
            await message.channel.send("歡迎！您的設定似乎不完整，請使用 `/start` 指令來開始或重置您的 AI 戀人。")
            return

        logger.info(f"[{user_id}] 接收到訊息 '{message.content[:50]}...'，啟動 LangGraph 對話流程...")
        async with message.channel.typing():
            try:
                if user_id not in ai_instance.session_histories:
                    ai_instance.session_histories[user_id] = ChatMessageHistory()
                
                chat_history_manager = ai_instance.session_histories[user_id]
                current_messages = chat_history_manager.messages.copy()
                current_messages.append(HumanMessage(content=message.content))

                initial_state = ConversationGraphState(
                    user_id=user_id,
                    ai_core=ai_instance,
                    messages=current_messages,
                    input_analysis=None,
                    scene_analysis=None,
                    rag_context="",
                    structured_context={},
                    dynamic_prompt="",
                    llm_response="",
                    final_output="",
                    state_updates={}
                )

                final_state = await self.main_response_graph.ainvoke(initial_state)
                
                response = final_state.get('final_output')
                
                if response and response.strip() and response != "（...）":
                    for i in range(0, len(response), 2000):
                        await message.channel.send(response[i:i+2000])
                else:
                    logger.error(f"LangGraph 為使用者 {user_id} 返回了空的或無效的回應。觸發智能備用回應。")
                    fallback_message = (
                        "（抱歉，我好像突然斷線了，腦袋一片空白... 這可能是因為您的指令觸發了內容安全限制，或者是一個暫時的網絡問題。\n\n"
                        "**您可以嘗試：**\n"
                        "1.  **換一種說法**：嘗試用更委婉或不同的詞語來表達您的意思。\n"
                        "2.  **稍後再試**：如果認為是網絡問題，請稍等片刻再發送一次相同的指令。）"
                    )
                    await message.channel.send(fallback_message)

            except Exception as e:
                logger.error(f"處理使用者 {user_id} 的 LangGraph 聊天流程時發生未捕獲的異常: {e}", exc_info=True)
                await message.channel.send(f"處理您的訊息時發生了一個嚴重的內部錯誤，管理員已收到通知。")

    async def finalize_setup(self, interaction: discord.Interaction, canon_text: Optional[str] = None):
        user_id = str(interaction.user.id)
        
        initial_message = "✅ 設定流程已進入最後階段！\n🚀 **正在為您執行最終創世...**"
        if canon_text:
            initial_message = "✅ 世界聖經已提交！\n🚀 **正在融合您的世界觀並執行最終創世...**"
        
        await interaction.followup.send(initial_message, ephemeral=True)
        
        ai_instance = await self.get_or_create_ai_instance(user_id)
        if not ai_instance or not ai_instance.profile:
            logger.error(f"[{user_id}] 在 finalize_setup 中獲取 AI 核心失敗。")
            await interaction.followup.send("❌ 錯誤：無法從資料庫加載您的基礎設定以進行創世。", ephemeral=True)
            self.setup_locks.discard(user_id)
            return

        try:
            logger.info(f"[{user_id}] /start 流程：正在強制初始化 AI 核心組件...")
            await ai_instance._configure_model_and_chain()
            
            initial_state = SetupGraphState(
                user_id=user_id,
                ai_core=ai_instance,
                canon_text=canon_text,
                genesis_result=None,
                opening_scene=""
            )

            final_state = await self.setup_graph.ainvoke(initial_state)
            opening_scene = final_state.get('opening_scene')
            
            if not opening_scene:
                 opening_scene = (f"在一片柔和的光芒中，你和 {ai_instance.profile.ai_profile.name} 發現自己身處於一個寧靜的空間裡，故事即將從這裡開始。"
                                  "\n\n（系統提示：由於您的設定，AI無法生成更詳細的開場白，但您現在可以開始互動了。）")


            await interaction.followup.send("🎉 您的專屬世界已誕生！正在為您揭開故事的序幕...", ephemeral=True)
            dm_channel = await interaction.user.create_dm()
            
            DISCORD_MSG_LIMIT = 2000
            if len(opening_scene) > DISCORD_MSG_LIMIT:
                for i in range(0, len(opening_scene), DISCORD_MSG_LIMIT):
                    await dm_channel.send(opening_scene[i:i+DISCORD_MSG_LIMIT])
            else:
                await dm_channel.send(opening_scene)

        except Exception as e:
            logger.error(f"[{user_id}] 在 LangGraph 設定流程中發生無法恢復的嚴重錯誤: {e}", exc_info=True)
            await interaction.followup.send(f"❌ **錯誤**：在執行最終設定時發生了未預期的嚴重錯誤: {e}", ephemeral=True)
        finally:
            self.setup_locks.discard(user_id)

    async def parse_and_create_lore_from_canon(self, interaction: discord.Interaction, content_text: str, is_setup_flow: bool = False):
        user_id = str(interaction.user.id)
        try:
            ai_instance = await self.get_or_create_ai_instance(user_id)
            if not ai_instance or not ai_instance.profile:
                if not is_setup_flow:
                    await interaction.followup.send("❌ **錯誤**：無法初始化您的 AI 核心來處理檔案。", ephemeral=True)
                return

            logger.info(f"[{user_id}] 背景任務：開始智能合併世界聖經...")
            
            followup_target = interaction.followup if interaction and not is_setup_flow else None

            await ai_instance.parse_and_create_lore_from_canon(interaction, content_text, is_setup_flow)

            if followup_target:
                await followup_target.send("✅ **智能合併完成！**\nAI 正在學習您的世界觀，相關的 NPC、地點等資訊將在後續對話中體現。", ephemeral=True)

        except Exception as e:
            logger.error(f"[{user_id}] 在背景中解析世界聖經時發生錯誤: {e}", exc_info=True)
            if not is_setup_flow and interaction:
                await interaction.followup.send(f"❌ **錯誤**：在處理您的世界聖經時發生未預期的錯誤。", ephemeral=True)


    
    
    
    
    
    # 函式：開始重置流程 (v39.1 - 健壯性審查)
    # 更新紀錄:
    # v39.1 (2025-09-02): [健壯性] 審查並確認此函式的重試邏輯。此函式現在依賴於 `AILover.shutdown` (v198.1) 中強化的資源回收機制來確保 `shutil.rmtree` 的成功執行。保留其內部的重試循環作為應對極端情況下的最終保障。
    # v37.0 (2025-08-31): [災難性BUG修復] 修復了在 Windows 環境下，因檔案鎖定導致的 `/start` 重置時刪除向量目錄失敗的嚴重問題。
    async def start_reset_flow(self, interaction: discord.Interaction):
        user_id = str(interaction.user.id)
        try:
            logger.info(f"[{self.user_id}] 後台重置任務開始...")
            
            # 步驟 1: 關閉並移除記憶體中的 AI 實例 (現在會調用強化版的 shutdown)
            if user_id in self.ai_instances:
                ai_instance_to_shutdown = self.ai_instances.pop(user_id)
                await ai_instance_to_shutdown.shutdown()
                logger.info(f"[{self.user_id}] 已關閉活躍的 AI 實例並已請求釋放檔案鎖定。")
            
            # 步驟 2: 從共享資料庫中安全地刪除該使用者的所有數據
            async with AsyncSessionLocal() as session:
                await session.execute(delete(MemoryData).where(MemoryData.user_id == user_id))
                await session.execute(delete(Lore).where(Lore.user_id == user_id))
                await session.execute(delete(UserData).where(UserData.user_id == user_id))
                await session.commit()
                logger.info(f"[{self.user_id}] 已從資料庫安全地清除了所有相關記錄。")

            # 步驟 3: 刪除向量數據庫目錄 (帶重試邏輯)
            vector_store_path = Path(f"./data/vector_stores/{user_id}")
            if vector_store_path.exists() and vector_store_path.is_dir():
                max_attempts = 10
                for attempt in range(max_attempts):
                    try:
                        await asyncio.to_thread(shutil.rmtree, vector_store_path)
                        logger.info(f"[{user_id}] (第 {attempt + 1} 次嘗試) 已成功刪除向量數據庫目錄。")
                        break
                    except (PermissionError, OSError, FileNotFoundError) as e:
                        if attempt < max_attempts - 1:
                            logger.warning(f"[{user_id}] /start 重置時刪除向量目錄失敗 (第 {attempt + 1} 次)，將在 1.0 秒後重試。錯誤: {e}")
                            await asyncio.sleep(1.0)
                        else:
                            logger.error(f"[{user_id}] /start 重置時刪除向量目錄失敗，已達最大重試次數: {e}", exc_info=True)
                            error_message = (
                                "❌ **重置失敗**\n"
                                "刪除舊數據時發生檔案鎖定錯誤，這通常是暫時的。\n\n"
                                "**建議：** 請等待約 **10-30 秒**，讓系統完全釋放檔案，然後再次嘗試 `/start` 指令。"
                            )
                            if not interaction.response.is_done():
                                await interaction.response.edit_message(content=error_message, view=None)
                            else:
                                await interaction.followup.send(content=error_message, ephemeral=True)
                            return

            view = StartSetupView(cog=self, user_id=user_id)
            if not interaction.response.is_done():
                await interaction.response.edit_message(content="✅ 重置完成！請點擊下方按鈕開始全新的設定流程。", view=view)
            else:
                 await interaction.followup.send(content="✅ 重置完成！請點擊下方按鈕開始全新的設定流程。", view=view, ephemeral=True)

        except Exception as e:
            logger.error(f"[{user_id}] 後台重置任務失敗: {e}", exc_info=True)
            error_message = f"執行重置時發生未知的嚴重錯誤: {e}"
            if not interaction.response.is_done():
                await interaction.response.edit_message(content=error_message, view=None)
            else:
                await interaction.followup.send(content=error_message, view=None, ephemeral=True)
        finally:
            self.setup_locks.discard(user_id)
    # 函式：開始重置流程 (v39.1 - 健壯性審查)
            







    

    @app_commands.command(name="start", description="開始全新的冒險（這將重置您所有的現有資料）")
    async def start(self, interaction: discord.Interaction):
        user_id = str(interaction.user.id)
        if not isinstance(interaction.channel, discord.DMChannel):
            await interaction.response.send_message("此指令只能在私訊頻道中使用。", ephemeral=True)
            return

        if user_id in self.setup_locks:
            view = ForceRestartView(cog=self)
            view.original_interaction_user_id = interaction.user.id
            await interaction.response.send_message(
                "我們偵測到您有一個尚未完成的設定流程。您想要？",
                view=view,
                ephemeral=True
            )
            return
        
        view = ConfirmStartView(cog=self)
        view.original_interaction_user_id = interaction.user.id
        await interaction.response.send_message(
            "⚠️ **警告** ⚠️\n您確定要開始一段全新的冒險嗎？\n這將會**永久刪除**您當前所有的角色、世界、記憶和進度。此操作無法復原。", 
            view=view, 
            ephemeral=True
        )

    @app_commands.command(name="settings", description="進入設定中心，管理你的角色、AI戀人與世界觀")
    async def settings_command(self, interaction: discord.Interaction):
        if not isinstance(interaction.channel, discord.DMChannel):
            await interaction.response.send_message("此指令只能在私訊頻道中使用。", ephemeral=True)
            return
        user_id = str(interaction.user.id)
        ai_instance = await self.get_or_create_ai_instance(user_id)
        if ai_instance and ai_instance.profile:
            view = SettingsChoiceView(self)
            await interaction.response.send_message("請選擇您想修改的設定項目：", view=view, ephemeral=True)
        else:
            await interaction.response.send_message("請先使用 `/start` 指令進行初始設定。", ephemeral=True)

    @app_commands.command(name="response_style", description="客製化 AI 的回覆風格")
    async def response_style(self, interaction: discord.Interaction):
        if not isinstance(interaction.channel, discord.DMChannel):
            await interaction.response.send_message("此指令只能在私訊頻道中使用。", ephemeral=True)
            return
            
        user_id = str(interaction.user.id)
        logger.info(f"[{user_id}] 使用者觸發 /response_style 指令。")

        ai_instance = await self.get_or_create_ai_instance(user_id)
        if not ai_instance or not ai_instance.profile:
            await interaction.response.send_message("請先使用 `/start` 指令進行初始設定。", ephemeral=True)
            return
        
        current_style = ai_instance.profile.response_style_prompt or """# ==================================================
# AI 回覆風格設定 (對話為主，描述為輔)
# ==================================================
# 說明：
# 1. 直接修改下方程式碼塊中的「參數設定」值。
# 2. 或者，刪除整個參數區塊，直接用白話文描述你想要的風格。
# 3. AI會理解所有內容，包含 # 註解。

# --- 參數設定 ---
# 說明：請從各項的「可選值」中挑選一個填入。

旁白比例: 中等
# 控制場景、動作描述的文字量。
# 推薦值: 低, 中等
# 可選值: 無, 極低, 低, 中等, 高, 極高

對話比例: 極高
# 控制角色發言的文字量和頻率，推薦保持「極高」以鼓勵對話。
# 推薦值: 極高
# 可選值: 無, 極低, 低, 中等, 高, 極高

角色主動性: 極高
# 控制 AI/NPC 主動發起對話或引導話題的傾向。
# 推薦值: 高, 極高
# 可選值: 低, 中等, 高, 極高

# --- (可選) 自然語言風格範例 ---
#
# 範例 (小說風格):
# 我想要非常細膩的描寫，請大量描述角色的內心活動、表情和周圍環境的細節。
# 同時，我也非常鼓勵角色之間的對話，請確保 AI 和 NPC 有足夠的、生動的發言來推進故事。
"""
        modal = ResponseStyleModal(self, current_style)
        await interaction.response.send_modal(modal)

    @app_commands.command(name="edit_profile", description="使用選單或按鈕編輯您或任何角色的個人檔案。")
    async def edit_profile(self, interaction: discord.Interaction):
        view = EditProfileRootView(cog=self, original_user_id=interaction.user.id)
        await interaction.response.send_message("請選擇您想編輯的角色檔案：", view=view, ephemeral=True)
        
    @app_commands.command(name="upload_canon", description="上傳您的個人世界觀設定檔 (txt)")
    @app_commands.describe(file="請上傳一個 .txt 格式的檔案，最大 5MB。")
    async def upload_canon(self, interaction: discord.Interaction, file: discord.Attachment):
        user_id = str(interaction.user.id)
        ai_instance = await self.get_or_create_ai_instance(user_id)
        if not ai_instance:
            await interaction.response.send_message("錯誤：找不到您的使用者資料。", ephemeral=True)
            return
        if not file.filename.lower().endswith('.txt'):
            await interaction.response.send_message("❌ 檔案格式錯誤！請上傳 `.txt` 檔案。", ephemeral=True)
            return
        if file.size > 5 * 1024 * 1024:
            await interaction.response.send_message("❌ 檔案過大！檔案大小不能超過 5MB。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            content_bytes = await file.read()
            content_text = content_bytes.decode('utf-8')
            
            chunk_count = await ai_instance.add_canon_to_vector_store(content_text)
            await interaction.followup.send(f"✅ **核心設定檔上傳成功！**\n檔案 `{file.filename}` 已被分解為 **{chunk_count}** 個知識片段儲存。\n\n🧠 現在將在背景中為您智能合併世界設定，請稍候...", ephemeral=True)

            asyncio.create_task(self.parse_and_create_lore_from_canon(interaction, content_text))

        except UnicodeDecodeError:
            await interaction.followup.send("❌ **檔案編碼錯誤！**\n請將檔案另存為 `UTF-8` 編碼後再試一次。", ephemeral=True)
        except Exception as e:
            logger.error(f"[{user_id}] 處理上傳的核心設定檔時發生錯誤: {e}", exc_info=True)
            await interaction.followup.send(f"處理檔案時發生內部錯誤。")

    @app_commands.command(name="admin_set_affinity", description="[管理員] 設定指定使用者的好感度")
    @app_commands.check(is_admin)
    @app_commands.autocomplete(target_user=user_autocomplete)
    async def admin_set_affinity(self, interaction: discord.Interaction, target_user: str, value: app_commands.Range[int, -1000, 1000]):
        target_user_id = target_user
        async with AsyncSessionLocal() as session:
            user_data = await session.get(UserData, target_user_id)
            if user_data:
                user_data.affinity = value
                await session.commit()
                if target_user_id in self.ai_instances and self.ai_instances[target_user_id].profile:
                    self.ai_instances[target_user_id].profile.affinity = value
                await interaction.response.send_message(f"已將使用者 {user_data.username} ({target_user_id}) 的好感度設定為 {value}。", ephemeral=True)
            else:
                await interaction.response.send_message(f"錯誤：找不到使用者 {target_user_id}。", ephemeral=True)

    @app_commands.command(name="admin_reset", description="[管理員] 清除指定使用者的所有資料")
    @app_commands.check(is_admin)
    @app_commands.autocomplete(target_user=user_autocomplete)
    async def admin_reset(self, interaction: discord.Interaction, target_user: str):
        target_user_id = target_user
        
        user_display_name = f"ID: {target_user_id}"
        async with AsyncSessionLocal() as session:
            user_data = await session.get(UserData, target_user_id)
            if not user_data:
                await interaction.response.send_message(f"錯誤：在資料庫中找不到使用者 {target_user_id}。", ephemeral=True)
                return
            user_display_name = user_data.username or user_display_name
        
        await interaction.response.defer(ephemeral=True, thinking=True)
        logger.info(f"管理員 {interaction.user.id} 正在重置使用者 {target_user_id}...")
        
        if target_user_id in self.ai_instances:
            await self.ai_instances.pop(target_user_id).shutdown()
            logger.info(f"[{target_user_id}] 已為管理員重置關閉活躍的 AI 實例。")
        
        await init_db()
        
        async with AsyncSessionLocal() as session:
            await session.execute(delete(MemoryData).where(MemoryData.user_id == target_user_id))
            await session.execute(delete(Lore).where(Lore.user_id == target_user_id))
            await session.execute(delete(UserData).where(UserData.user_id == target_user_id))
            await session.commit()
            logger.info(f"[{target_user_id}] 已從資料庫清除該使用者的所有相關記錄。")
            
        try:
            vector_store_path = Path(f"./data/vector_stores/{target_user_id}")
            if vector_store_path.exists():
                await asyncio.to_thread(shutil.rmtree, vector_store_path)
                logger.info(f"[{target_user_id}] 已成功刪除該使用者的向量數據庫目錄。")
        except Exception as e:
            logger.error(f"管理員重置使用者 {target_user_id} 時刪除向量目錄失敗: {e}", exc_info=True)
            await interaction.followup.send(f"已成功重置使用者 {user_display_name} 的核心資料庫數據，但刪除其向量目錄時發生錯誤。", ephemeral=True)
            return
        
        await interaction.followup.send(f"已成功重置使用者 {user_display_name} ({target_user_id}) 的所有資料。", ephemeral=True)

    # 函式：管理員強制更新指令 (v39.0 新增)
    @app_commands.command(name="admin_force_update", description="[管理員] 強制從 GitHub 同步最新程式碼並重啟機器人。")
    @app_commands.check(is_admin)
    async def admin_force_update(self, interaction: discord.Interaction):
        """強制從 GitHub 拉取最新程式碼並重啟。"""
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        logger.info(f"管理員 {interaction.user.id} 觸發了強制更新...")
        
        await interaction.followup.send("⏳ **正在強制同步...**\n正在從遠端倉庫 `origin/main` 獲取最新版本...", ephemeral=True)

        try:
            # 使用 asyncio.to_thread 在單獨的線程中運行阻塞的 subprocess 命令
            git_reset_command = ["git", "reset", "--hard", "origin/main"]
            process = await asyncio.to_thread(
                subprocess.run,
                git_reset_command,
                capture_output=True,
                text=True,
                encoding='utf-8',
                check=False
            )

            if process.returncode == 0:
                logger.info("強制同步成功，準備重啟...")
                success_message = (
                    "✅ **同步成功！**\n"
                    "程式碼已強制更新至最新版本。\n\n"
                    "🔄 **機器人將在 3 秒後自動重啟...**"
                )
                await interaction.followup.send(success_message, ephemeral=True)
                await asyncio.sleep(3)
                
                # 使用 os.execv 實現無縫重啟
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                logger.error(f"強制同步失敗: {process.stderr}")
                error_message = (
                    f"🔥 **同步失敗！**\n"
                    f"Git 返回了錯誤，請檢查後台日誌。\n\n"
                    f"```\n{process.stderr.strip()}\n```"
                )
                await interaction.followup.send(error_message, ephemeral=True)

        except FileNotFoundError:
            logger.error("Git 命令未找到，無法執行強制更新。")
            await interaction.followup.send("🔥 **錯誤：`git` 命令未找到！**\n請確保伺服器環境已安裝 Git。", ephemeral=True)
        except Exception as e:
            logger.error(f"執行強制更新時發生未預期錯誤: {e}", exc_info=True)
            await interaction.followup.send(f"🔥 **發生未預期錯誤！**\n執行更新時遇到問題: {e}", ephemeral=True)
    # 函式：管理員強制更新指令 (v39.0 新增)

    @app_commands.command(name="admin_check_status", description="[管理員] 查詢指定使用者的當前狀態")
    @app_commands.check(is_admin)
    @app_commands.autocomplete(target_user=user_autocomplete)
    async def admin_check_status(self, interaction: discord.Interaction, target_user: str):
        target_user_id = target_user
        discord_user = self.bot.get_user(int(target_user_id))
        async with AsyncSessionLocal() as session:
            user_data = await session.get(UserData, target_user_id)
            if user_data:
                game_state = GameState.model_validate(user_data.game_state or {})
                embed = Embed(title=f"📊 使用者狀態查詢: {user_data.username}", description=f"AI 戀人: **{user_data.ai_name}**", color=discord.Color.blue())
                if discord_user: embed.set_thumbnail(url=discord_user.display_avatar.url)
                embed.add_field(name="❤️ AI 好感度", value=f"**{user_data.affinity}** / 1000", inline=True)
                embed.add_field(name="💰 金錢", value=str(game_state.money), inline=True)
                embed.add_field(name="📍 當前地點", value=' > '.join(game_state.location_path), inline=False)
                inventory_text = ", ".join(game_state.inventory) if game_state.inventory else "空"
                embed.add_field(name="🎒 物品欄", value=inventory_text, inline=False)
                embed.set_footer(text=f"User ID: {target_user_id}")
                await interaction.response.send_message(embed=embed, ephemeral=True)
            else:
                await interaction.response.send_message(f"錯誤：找不到使用者 {target_user_id}。", ephemeral=True)
    
    @app_commands.command(name="admin_check_lore", description="[管理員] 查詢指定使用者的 Lore 詳細資料")
    @app_commands.check(is_admin)
    @app_commands.describe(target_user="從列表中選擇要查詢的使用者", category="選擇 Lore 類別", key="輸入文字以搜尋 Lore")
    @app_commands.autocomplete(target_user=user_autocomplete, key=lore_key_autocomplete)
    @app_commands.choices(category=LORE_CATEGORIES)
    async def admin_check_lore(self, interaction: discord.Interaction, target_user: str, category: str, key: str):
        target_user_id = target_user
        lore_entry = await lore_book.get_lore(target_user_id, category, key)
        discord_user = self.bot.get_user(int(target_user_id))
        if lore_entry:
            content_str = json.dumps(lore_entry.content, ensure_ascii=False, indent=2)
            embed = Embed(title=f"📜 Lore 查詢: {key.split(' > ')[-1]}", description=f"**類別**: `{category}`\n**使用者**: {discord_user.name if discord_user else '未知'}", color=discord.Color.green())
            if len(content_str) > 1000: content_str = content_str[:1000] + "\n... (內容過長)"
            embed.add_field(name="詳細資料", value=f"```json\n{content_str}\n```", inline=False)
            embed.set_footer(text=f"完整主鍵: {key}\nUser ID: {target_user_id}")
            await interaction.response.send_message(embed=embed, ephemeral=True)
        else:
            await interaction.response.send_message(f"錯誤：找不到使用者 {target_user_id} 的 `{category}` / `{key}` Lore。", ephemeral=True)

    @admin_set_affinity.error
    @admin_reset.error
    @admin_check_status.error
    @admin_check_lore.error
    @admin_force_update.error
    async def on_admin_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.CheckFailure):
            await interaction.response.send_message("你沒有權限使用此指令。", ephemeral=True)
        else:
            logger.error(f"一個管理員指令發生錯誤: {error}", exc_info=True)
            if not interaction.response.is_done():
                await interaction.response.send_message(f"發生未知錯誤。", ephemeral=True)
# 類別：機器人核心功能集 (Cog)

# 類別：AI 戀人機器人主體
class AILoverBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix='/', intents=intents, activity=discord.Game(name="與你共度時光"))
    
    async def setup_hook(self):
        await self.add_cog(BotCog(self))
        await self.tree.sync()
        logger.info("Discord Bot is ready and commands are synced!")
    
    async def on_ready(self):
        logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
# 類別：AI 戀人機器人主體
