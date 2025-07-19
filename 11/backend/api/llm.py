# llm.py
# 与大模型对话/文档上传/网络搜索接口

from fastapi import APIRouter, Depends, Form, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import get_db
from services.llm_service import LLMService
from models.schemas import ChatRequest
from typing import Optional, AsyncGenerator
import uuid

router = APIRouter()
llm_service = LLMService()

@router.post('/chat')
async def chat(
    message: str = Form(...),
    session_id: str = Form(None),
    file: UploadFile = File(None),
    db: AsyncSession = Depends(get_db)
):
    req = ChatRequest(message=message, session_id=session_id)
    file_bytes = None
    filename = None
    if file:
        file_bytes = await file.read()
        filename = file.filename

    async def stream():
        async for chunk in llm_service.chat(req, db, file_bytes, filename):
            yield chunk
    return StreamingResponse(stream(), media_type="text/event-stream")

@router.post('/upload')
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    """文档上传与解析接口，返回解析内容"""
    file_bytes = await file.read()
    doc_content = await llm_service.upload_and_parse_document(file_bytes, file.filename, session_id)
    return {"session_id": session_id, "content": doc_content}

@router.get('/search')
def web_search():
    # TODO: 实现网络搜索接口
    pass 