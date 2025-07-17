# 聊天服务模块

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import os
import uuid
from typing import List, Optional, AsyncGenerator
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from db import get_db, DBUtils, Session as DBSession, Message as DBMessage
from schemas import (
    Message as MessageSchema, 
    ChatHistory, 
    ChatResponse, 
    SessionInfo, 
    SessionListResponse,
    FileUploadResponse
)
from services.document_processor import DocumentProcessor
from config import settings
from dotenv import load_dotenv
load_dotenv()

router = APIRouter()

# 通义千问 API 配置
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
if not QWEN_API_KEY:
    raise RuntimeError("请先设置环境变量 QWEN_API_KEY")

class ChatService:
    """聊天服务类，处理聊天相关的业务逻辑"""
    
    @staticmethod
    async def qwen_streaming_ask(prompt: str) -> AsyncGenerator[str, None]:
        """通过通义千问API获取流式回复"""
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": settings.LLM_MODEL_NAME,
            "input": {"prompt": prompt},
            "parameters": {"result_format": "message", "stream": True}
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", settings.QWEN_API_URL, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise HTTPException(status_code=response.status_code, detail=f"API错误: {error_text}")
                
                async for line in response.aiter_lines():
                    if line.strip():
                        yield line + "\n"
    
    @staticmethod
    async def process_document(file: UploadFile, session_id: str) -> str:
        """处理上传的文档文件"""
        # 保存文件到 uploads/{session_id}/ 目录
        upload_dir = os.path.join(settings.UPLOAD_DIR, session_id)
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        # 写入文件
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 解析文件内容
        processor = DocumentProcessor()
        doc_content = processor.process(file_path)
        return doc_content
    
    @staticmethod
    async def prepare_prompt(question: str, doc_content: str = "") -> str:
        """准备发送给模型的提示词"""
        if doc_content:
            return f"用户问题：{question}\n\n以下是用户上传的文件内容，请结合文件内容回答：\n{doc_content}"
        else:
            return question
    
    @staticmethod
    async def generate_summary(messages: List[DBMessage], prompt: str):
        """使用langchain生成对话摘要"""
        from langchain_community.chat_models.tongyi import ChatTongyi
        from langchain.memory import ConversationSummaryMemory
        from pydantic import SecretStr
        
        # 初始化模型
        api_key = SecretStr(QWEN_API_KEY) if QWEN_API_KEY else None
        llm = ChatTongyi(
            api_key=api_key,
            model_name=settings.LLM_MODEL_NAME
        )
        
        # 使用摘要记忆
        memory = ConversationSummaryMemory(llm=llm, return_messages=True)
        
        # 添加历史消息
        for m in messages:
            memory.save_context(
                {"input": m.content if m.role == "user" else ""}, 
                {"output": m.content if m.role == "assistant" else ""}
            )
        
        # 添加当前问题
        memory.save_context({"input": prompt}, {"output": ""})
        
        # 构造历史prompt（摘要）
        history_prompt = ""
        for msg in memory.chat_memory.messages:
            if msg.type == "system":
                history_prompt += f"摘要：{msg.content}\n"
            elif msg.type == "human":
                history_prompt += f"用户：{msg.content}\n"
            elif msg.type == "ai":
                history_prompt += f"助手：{msg.content}\n"
        
        history_prompt += "助手："
        return history_prompt

# 路由定义
@router.post("/ask")
async def ask(
    question: str = Form(...),
    session_id: str = Form(None),
    file: UploadFile = File(None),
    db: AsyncSession = Depends(get_db)
):
    """处理用户提问"""
    # 生成会话ID
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # 处理上传的文件
    doc_content = ""
    if file:
        doc_content = await ChatService.process_document(file, session_id)
    
    # 准备提示词
    prompt = await ChatService.prepare_prompt(question, doc_content)
    
    # 保存用户消息
    await DBUtils.get_session(db, session_id)
    await DBUtils.add_message(db, session_id, "user", prompt)
    
    # 获取历史消息
    messages = await DBUtils.get_messages(db, session_id)
    
    # 生成摘要
    history_prompt = await ChatService.generate_summary(messages, prompt)
    
    # 流式返回大模型回复，并保存
    async def stream_and_save():
        content = ""
        import json as _json
        
        async for chunk in ChatService.qwen_streaming_ask(history_prompt):
            try:
                data = _json.loads(chunk)
                message = data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                content += message
            except Exception as e:
                print(f"解析异常：{e}")
            yield chunk
        
        # 保存助手回复
        await DBUtils.add_message(db, session_id, "assistant", content)
    
    return StreamingResponse(
        stream_and_save(), 
        media_type="text/event-stream", 
        headers={"X-Session-Id": session_id}
    )

@router.get("/history/{session_id}")
async def get_history(session_id: str, db: AsyncSession = Depends(get_db)):
    """获取会话历史"""
    # 检查会话是否存在
    session = await DBUtils.get_session(db, session_id)
    
    # 获取所有消息
    messages = await DBUtils.get_messages(db, session_id)
    
    return ChatHistory(
        session_id=session_id,
        messages=[
            MessageSchema(
                role=m.role, 
                content=m.content,
                created_at=m.created_at
            ) for m in messages
        ]
    )

@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(db: AsyncSession = Depends(get_db)):
    """获取所有会话列表"""
    sessions = await DBUtils.get_all_sessions(db)
    
    result = []
    for s in sessions:
        # 获取消息数量
        messages = await DBUtils.get_messages(db, s.id)
        
        result.append(SessionInfo(
            session_id=s.id,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=len(messages)
        ))
    
    return SessionListResponse(
        sessions=result,
        total=len(result)
    )

@router.post("/upload_doc", response_model=FileUploadResponse)
async def upload_doc(
    file: UploadFile = File(...), 
    session_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """上传文档文件"""
    # 生成会话ID
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # 处理文档
    doc_content = await ChatService.process_document(file, session_id)
    
    # 保存文档内容作为用户消息
    await DBUtils.get_session(db, session_id)
    await DBUtils.add_message(db, session_id, "user", doc_content)
    
    return FileUploadResponse(
        session_id=session_id,
        file_name=file.filename,
        content_length=len(doc_content)
    )