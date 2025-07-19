# session.py
# 会话列表/导入导出接口

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import get_db
from services.session_service import SessionService
import json

router = APIRouter()

@router.get('/sessions')
async def list_sessions(db: AsyncSession = Depends(get_db)):
    """获取历史会话列表接口"""
    return await SessionService.list_sessions(db)

@router.post('/sessions/import')
async def import_session(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """导入会话接口，上传json文件"""
    try:
        content = await file.read()
        session_data = json.loads(content.decode('utf-8'))
        await SessionService.import_session(db, session_data)
        return {"msg": "导入成功"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"导入失败: {e}")

@router.get('/sessions/export')
async def export_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """导出会话接口，返回json字符串"""
    data = await SessionService.export_session(db, session_id)
    return {
        "session_id": session_id,
        "data": json.loads(data)
    } 