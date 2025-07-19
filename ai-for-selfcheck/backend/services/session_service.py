# session_service.py
# 管理会话的保存、导入导出、历史列表等

from db.database import DBUtils
from models.schemas import SessionInfo, SessionListResponse, ChatHistory, Message as MessageSchema
from sqlalchemy.ext.asyncio import AsyncSession
import json

class SessionService:

    @staticmethod
    async def save_session(db: AsyncSession, session_id: str, messages: list):
        """保存会话及其消息（会话不存在则新建，消息为[{role, content}]列表）"""
        await DBUtils.get_session(db, session_id)
        for m in messages:
            await DBUtils.add_message(db, session_id, m.get("role"), m.get("content"))
        return True 
    
    @staticmethod
    async def list_sessions(db: AsyncSession, user_id=None):
        """获取所有会话列表（当前无用户体系，user_id暂未用）"""
        sessions = await DBUtils.get_all_sessions(db)
        result = []
        for s in sessions:
            messages = await DBUtils.get_messages(db, s.id)
            result.append(SessionInfo(
                session_id=s.id,
                created_at=s.created_at,
                updated_at=s.updated_at,
                message_count=len(messages)
            ))
        return SessionListResponse(sessions=result, total=len(result))

    @staticmethod
    async def export_session(db: AsyncSession, session_id: str):
        """导出指定会话的所有消息为 JSON"""
        messages = await DBUtils.get_messages(db, session_id)
        msg_list = [
            dict(role=m.role, content=m.content, created_at=m.created_at.isoformat() if m.created_at else None)
            for m in messages
        ]
        return json.dumps({"session_id": session_id, "messages": msg_list}, ensure_ascii=False)

    @staticmethod
    async def import_session(db: AsyncSession, session_data: dict):
        """导入会话（session_data: dict，包含session_id和messages）"""
        session_id = session_data.get("session_id")
        messages = session_data.get("messages", [])
        await DBUtils.get_session(db, session_id)
        for m in messages:
            await DBUtils.add_message(db, session_id, m.get("role"), m.get("content"))
        return True
