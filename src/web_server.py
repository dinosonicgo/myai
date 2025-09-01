# web_server.py 的中文註釋(v2.0 - 健壯性修正)
# 更新紀錄:
# v2.0 (2025-08-04):
# 1. [健壯性] 在 ConnectionManager.send_message 中增加了連線狀態檢查。現在，在發送任何訊息之前，都會確認客戶端 WebSocket 是否仍然處於連接狀態。
# 2. [BUG修復] 此修改徹底解決了因使用者在 AI 處理期間提前關閉或刷新頁面，導致後端嘗試向已關閉的連接發送訊息而引發的 RuntimeError 和 WebSocketDisconnect 錯誤。
# v1.9 (2050-08-07):
# 1. [架構重構] 採用「延遲導入」模式，將所有 `src` 內部的模組導入語句從頂層移動到需要它們的函式內部。
# 2. [BUG修復] 此修改徹底解決了因 `main.py` -> `web_server.py` -> `ai_core.py` 導入鏈引發的循環導入（Circular Import）問題。

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from pathlib import Path
from sqlalchemy import delete
import logging
import shutil

from src.logger import logger

router = APIRouter()
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

@router.get('/favicon.ico', include_in_schema=False)
async def get_favicon():
    return FileResponse(str(STATIC_DIR / "img" / "favicon.ico"))

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.ai_instances: dict[str, "AILover"] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        from src.ai_core import AILover

        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Connection opened for client_id: {client_id}")

        if client_id not in self.ai_instances:
            ai = AILover(client_id)
            if await ai.initialize():
                self.ai_instances[client_id] = ai
                if ai.profile:
                    await self.send_message(f"歡迎回來！您的 AI 戀人 '{ai.profile.ai_profile.name}' 已上線。", client_id)
            else:
                await self.send_message("歡迎！請使用「開始冒險」按鈕來創建您的 AI 戀人。", client_id)

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.ai_instances:
            del self.ai_instances[client_id]
        logger.info(f"Connection closed for client_id: {client_id}")

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except WebSocketDisconnect:
                logger.warning(f"Attempted to send to a disconnected client (ID: {client_id}). Disconnecting.")
                self.disconnect(client_id)
            except RuntimeError as e:
                logger.warning(f"RuntimeError while sending to client {client_id}: {e}. Disconnecting.")
                self.disconnect(client_id)
        else:
            logger.info(f"Ignored send to non-existent or disconnected client_id: {client_id}")

manager = ConnectionManager()

async def finalize_setup_logic(client_id: str, setup_data: dict):
    from src.ai_core import AILover, CORE_SAFETY_PROMPT
    from src.database import AsyncSessionLocal, UserData
    from src.models import UserProfile, GameState
    from src.schemas import CharacterProfile, WorldGenesisResult
    from src.lore_book import add_or_update_lore

    ai_instance = AILover(client_id)
    manager.ai_instances[client_id] = ai_instance

    user_profile = CharacterProfile(**setup_data.get('user_profile', {}))
    ai_profile = CharacterProfile(**setup_data.get('ai_profile', {}))
    world_settings = setup_data.get('world_settings', '這是一個溫馨的世界')

    new_profile = UserProfile(
        user_id=client_id,
        user_profile=user_profile,
        ai_profile=ai_profile,
        world_settings=world_settings,
        game_state=GameState()
    )
    await ai_instance.update_and_persist_profile(new_profile.model_dump())
    await ai_instance.initialize()

    if not ai_instance.profile:
        await manager.send_message("錯誤：AI 核心初始化失敗。", client_id)
        return

    try:
        logger.info(f"[{client_id}] 開始補完使用者和AI角色檔案...")
        completion_prompt = ai_instance.get_profile_completion_prompt()
        completion_llm = ai_instance.gm_model.with_structured_output(CharacterProfile)
        completion_chain = completion_prompt | completion_llm

        completed_user_profile = await ai_instance.ainvoke_with_rotation(completion_chain, {
            "username": user_profile.name or "使用者",
            "ai_name": ai_profile.name or "AI",
            "profile_json": user_profile.model_dump_json()
        })
        
        completed_ai_profile = await ai_instance.ainvoke_with_rotation(completion_chain, {
            "username": user_profile.name or "使用者",
            "ai_name": ai_profile.name or "AI",
            "profile_json": ai_profile.model_dump_json()
        })

        await ai_instance.update_and_persist_profile({
            'user_profile': completed_user_profile.model_dump(),
            'ai_profile': completed_ai_profile.model_dump()
        })
        logger.info(f"[{client_id}] 角色檔案補完成功。")

        logger.info(f"[{client_id}] 開始動態創世...")
        genesis_chain = ai_instance.get_world_genesis_chain()
        genesis_result: WorldGenesisResult = await ai_instance.ainvoke_with_rotation(genesis_chain, {
            "world_settings": ai_instance.profile.world_settings, "username": completed_user_profile.name, "ai_name": completed_ai_profile.name
        })
        
        gs = ai_instance.profile.game_state
        gs.location_path = genesis_result.location_path
        await ai_instance.update_and_persist_profile({'game_state': gs.model_dump()})

        await add_or_update_lore(client_id, 'location_info', " > ".join(genesis_result.location_path), genesis_result.location_info.model_dump())
        for npc in genesis_result.initial_npcs:
            npc_key = " > ".join(genesis_result.location_path) + f" > {npc.name}"
            await add_or_update_lore(client_id, 'npc_profile', npc_key, npc.model_dump())
        logger.info(f"[{client_id}] 動態創世成功。")
        
        await manager.send_message("🎉 您的專屬世界已誕生！正在為您揭開故事的序幕...", client_id)
        
        safety_bible = CORE_SAFETY_PROMPT.format(username=completed_user_profile.name, ai_name=completed_ai_profile.name)
        
        initial_message = f"""{safety_bible}

[系統指令]：撰寫故事開場白。

【核心要求】
1.  **雙人開場**：開場白【必須】同時描寫使用者角色「{completed_user_profile.name}」和 AI 戀人角色「{completed_ai_profile.name}」。
2.  **狀態還原**：【必須】準確描寫他們在【當前地點】的場景，並讓他們的行為、穿著和姿態完全符合下方提供的【角色檔案】。
3.  **氛圍營造**：營造出符合【世界觀】和【當前地點描述】的氛圍。

---
【絕對禁令】 - 這是最高優先級規則，必須無條件遵守！
1.  **【🚫 禁止翻譯名稱 🚫】**：角色名稱是固定的。你【絕對禁止】將使用者角色名稱「{completed_user_profile.name}」翻譯成任何其他語言或變體（例如，如果名稱是 'DINO'，絕不能寫成 '迪諾'）。必須原樣使用。
2.  **【🚫 禁止扮演使用者 🚫】**：你的職責是描寫場景和 AI 角色。你【絕對禁止】描寫使用者角色「{completed_user_profile.name}」的任何主觀思想、內心感受、或未明確提供的動作。只能根據其角色檔案進行客觀、靜態的描述（例如，他正坐在哪裡，穿著什麼）。
3.  **【🚫 禁止杜撰情節 🚫】**：這是一個**寧靜的**故事開端。你【絕對禁止】在開場白中加入任何極端的、未經使用者觸發的劇情，特別是任何形式的性愛、暴力或衝突場景。開場應是和平、中性的。

---
【世界觀】
{ai_instance.profile.world_settings}
---
【當前地點】: { " > ".join(genesis_result.location_path) }
【地點描述】: {genesis_result.location_info.description}
---
【使用者角色檔案：{completed_user_profile.name}】
{json.dumps(completed_user_profile.model_dump(), indent=2, ensure_ascii=False)}
---
【AI戀人角色檔案：{completed_ai_profile.name}】
{json.dumps(completed_ai_profile.model_dump(), indent=2, ensure_ascii=False)}
---

請嚴格遵守以上所有規則，開始撰寫一個寧靜且符合設定的開場故事。
"""
        
        response_message = await ai_instance.ainvoke_with_rotation(ai_instance.gm_model, {"input": initial_message})
        opening_scene = response_message.content
        await manager.send_message(opening_scene, client_id)

    except Exception as e:
        logger.error(f"[{client_id}] 在最終設定步驟中失敗: {e}", exc_info=True)
        await manager.send_message("在生成詳細設定或初始世界時發生嚴重錯誤。", client_id)

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    from src.ai_core import AILover
    from src.database import AsyncSessionLocal, UserData, MemoryData, Lore

    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()

            if data == '/reset':
                logger.info(f"[{client_id}] 收到重置指令...")
                if client_id in manager.ai_instances:
                    await manager.ai_instances[client_id].shutdown()
                    del manager.ai_instances[client_id]
                async with AsyncSessionLocal() as session:
                    await session.execute(delete(MemoryData).where(MemoryData.user_id == client_id))
                    await session.execute(delete(Lore).where(Lore.user_id == client_id))
                    await session.execute(delete(UserData).where(UserData.user_id == client_id))
                    await session.commit()
                vector_store_path = Path(f"./data/vector_stores/{client_id}")
                if vector_store_path.exists():
                    shutil.rmtree(vector_store_path)
                await manager.send_message("重置完成，請開始新的設定。", client_id)

            elif data.startswith('/finalize_setup '):
                payload = data.replace('/finalize_setup ', '', 1).strip()
                try:
                    setup_data = json.loads(payload)
                    await finalize_setup_logic(client_id, setup_data)
                except json.JSONDecodeError:
                    await manager.send_message("設定資料格式錯誤。", client_id)
                except Exception as e:
                    logger.error(f"Finalize setup failed for {client_id}: {e}", exc_info=True)
                    await manager.send_message("最終設定失敗，發生未知錯誤。", client_id)

            elif data.startswith('/get_character_profile '):
                profile_type = data.replace('/get_character_profile ', '', 1).strip()
                ai_instance = manager.ai_instances.get(client_id)
                if ai_instance and ai_instance.profile:
                    profile_attr = f"{profile_type}_profile"
                    profile_data = getattr(ai_instance.profile, profile_attr, None)
                    if profile_data:
                        await manager.send_message(json.dumps({"type": "current_character_profile", "profile_type": profile_type, "profile": profile_data.model_dump()}), client_id)
                    else:
                        await manager.send_message(f"錯誤：找不到 {profile_type} 的角色設定。", client_id)
                else:
                    await manager.send_message("錯誤：找不到您的 AI 實例或設定檔案。", client_id)

            elif data.startswith('/set_character_profile '):
                payload = data.replace('/set_character_profile ', '', 1).strip()
                try:
                    parts = payload.split(' ', 1)
                    profile_type = parts[0]
                    profile_json = parts[1]
                    profile_data = json.loads(profile_json)
                    
                    ai_instance = manager.ai_instances.get(client_id)
                    if ai_instance and ai_instance.profile:
                        profile_attr = f"{profile_type}_profile"
                        current_profile = getattr(ai_instance.profile, profile_attr)
                        
                        current_profile.name = profile_data.get('name', current_profile.name)
                        current_profile.gender = profile_data.get('gender', current_profile.gender)
                        current_profile.description = profile_data.get('description', current_profile.description)
                        current_profile.appearance = profile_data.get('appearance', current_profile.appearance)

                        success = await ai_instance.update_and_persist_profile({
                            profile_attr: current_profile.model_dump()
                        })
                        if success:
                            await manager.send_message(f"✅ **{current_profile.name}** 的角色設定已成功更新！", client_id)
                        else:
                            await manager.send_message("錯誤：更新角色設定失敗。", client_id)
                    else:
                        await manager.send_message("錯誤：找不到您的 AI 實例或設定檔案。", client_id)
                except (IndexError, json.JSONDecodeError):
                    await manager.send_message("設定角色資料的格式錯誤。", client_id)

            elif data == '/get_settings':
                async with AsyncSessionLocal() as session:
                    user_data = await session.get(UserData, client_id)
                    if user_data:
                        settings = {
                            "type": "current_settings",
                            "world_settings": user_data.world_settings or "",
                            "ai_settings": user_data.ai_settings or ""
                        }
                        await manager.send_message(json.dumps(settings), client_id)
                    else:
                        await manager.send_message("找不到您的資料，請先使用「初始設定」。", client_id)
            
            elif data == '/get_system_settings':
                async with AsyncSessionLocal() as session:
                    user_data = await session.get(UserData, client_id)
                    if user_data and user_data.one_instruction:
                        settings = {
                            "type": "current_system_settings",
                            "one_instruction": user_data.one_instruction
                        }
                        await manager.send_message(json.dumps(settings), client_id)
                    else:
                        await manager.send_message("找不到您的設定資料，請先進行「初始設定」。", client_id)

            elif data.startswith('/set_system_settings '):
                content = data.replace('/set_system_settings ', '', 1).strip()
                async with AsyncSessionLocal() as session:
                    user_data = await session.get(UserData, client_id)
                    if user_data:
                        user_data.one_instruction = content
                        await session.commit()
                        if client_id in manager.ai_instances and manager.ai_instances[client_id].profile:
                            manager.ai_instances[client_id].profile.one_instruction = content
                            await manager.ai_instances[client_id]._configure_model_and_chain()
                        await manager.send_message("系統設定已更新！AI 已重新載入設定。", client_id)
                    else:
                        await manager.send_message("找不到您的資料，請先使用「初始設定」。", client_id)

            elif data.startswith('/set_worldview '):
                content = data.replace('/set_worldview ', '', 1).strip()
                async with AsyncSessionLocal() as session:
                    user_data = await session.get(UserData, client_id)
                    if user_data:
                        user_data.world_settings = content
                        await session.commit()
                        if client_id in manager.ai_instances and manager.ai_instances[client_id].profile:
                            manager.ai_instances[client_id].profile.world_settings = content
                        await manager.send_message("世界觀設定已更新！", client_id)
                    else:
                        await manager.send_message("找不到您的資料，請先使用「初始設定」。", client_id)

            elif data.startswith('/set_aisettings '):
                content = data.replace('/set_aisettings ', '', 1).strip()
                async with AsyncSessionLocal() as session:
                    user_data = await session.get(UserData, client_id)
                    if user_data:
                        user_data.ai_settings = content
                        await session.commit()
                        if client_id in manager.ai_instances and manager.ai_instances[client_id].profile:
                            manager.ai_instances[client_id].profile.ai_profile.description = content
                        await manager.send_message("AI規範已更新！", client_id)
                    else:
                        await manager.send_message("找不到您的資料，請先使用「初始設定」。", client_id)

            elif data == '/max_affinity':
                async with AsyncSessionLocal() as session:
                    user_data = await session.get(UserData, client_id)
                    if user_data:
                        user_data.affinity = 1000
                        await session.commit()
                        if client_id in manager.ai_instances and manager.ai_instances[client_id].profile:
                            manager.ai_instances[client_id].profile.affinity = 1000
                        await manager.send_message("好感度已設定為最大值！", client_id)
                    else:
                        await manager.send_message("找不到使用者資料，請先使用「初始設定」。", client_id)

            elif data == '/clear_history':
                await manager.send_message("此指令已被 /reset 取代，請使用「開始冒險」按鈕重置。", client_id)

            else: 
                if client_id in manager.ai_instances and manager.ai_instances[client_id].profile:
                    await manager.send_message(json.dumps({"type": "typing"}), client_id)
                    response = await manager.ai_instances[client_id].chat(data)
                    await manager.send_message(response, client_id)
                else:
                    await manager.send_message("請先使用「開始冒險」按鈕來創建您的AI戀人。", client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket Error for client {client_id}: {e}", exc_info=True)
        manager.disconnect(client_id)
