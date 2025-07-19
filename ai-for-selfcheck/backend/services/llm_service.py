# llm_service.py
# 整合大模型对话、文档解析、网络搜索的 service 入口

import os
import uuid
import httpx
import json
from typing import AsyncGenerator, Optional
from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import DBUtils
from utils.config import settings
from models.schemas import ChatRequest, ChatResponse
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.memory import ConversationSummaryMemory
from pydantic import SecretStr
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

import aiofiles
from io import BytesIO

class LLMService:
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
        api_key = SecretStr(DASHSCOPE_API_KEY) if DASHSCOPE_API_KEY else None
        self.llm = ChatTongyi(api_key=api_key, model_name=settings.LLM_MODEL_NAME)

    @staticmethod
    async def qwen_streaming_ask(prompt: str) -> AsyncGenerator[str, None]:
        """通过通义千问API获取流式回复"""
        DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
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
    async def prepare_prompt(question: str, doc_content: str = "") -> str:
        if doc_content:
            return f"用户问题：{question}\n\n以下是用户上传的文件内容，请结合文件内容回答：\n{doc_content}"
        else:
            return question

    async def generate_summary(self, messages, prompt: str):
        memory = ConversationSummaryMemory(llm=self.llm, return_messages=True)
        for m in messages:
            memory.save_context(
                {"input": m.content if m.role == "user" else ""},
                {"output": m.content if m.role == "assistant" else ""}
            )
        memory.save_context({"input": prompt}, {"output": ""})
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

    async def upload_and_parse_document(self, file_bytes: bytes, filename: str, session_id: str) -> str:
        """上传并解析文档，返回内容字符串"""
        upload_dir = os.path.join(settings.UPLOAD_DIR, session_id)
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_bytes)
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        doc_content = "\n".join([doc.page_content for doc in docs])
        return doc_content

    async def chat(self, request: ChatRequest, db: AsyncSession, file_bytes: Optional[bytes] = None, filename: Optional[str] = None) -> AsyncGenerator[str, None]:
        session_id = request.session_id or str(uuid.uuid4())
        doc_content = ""
        if file_bytes and filename:
            doc_content = await self.upload_and_parse_document(file_bytes, filename, session_id)
        # 1. 只存用户原始输入
        await DBUtils.get_session(db, session_id)
        await DBUtils.add_message(db, session_id, "user", request.message)
        # 2. 获取历史消息
        messages = await DBUtils.get_messages(db, session_id)
        # 3. 生成摘要（只用历史消息+本轮用户输入，不含文档内容）
        history_prompt = await self.generate_summary(messages, request.message)
        # 4. prompt 构造时临时拼接文档内容
        if doc_content:
            prompt = f"{history_prompt}\n\n以下是用户上传的文件内容，请结合文件内容回答：\n{doc_content}\n用户问题：{request.message}"
        else:
            prompt = f"{history_prompt}\n用户问题：{request.message}"
        content = ""
        async for chunk in self.qwen_streaming_ask(prompt):
            try:
                data = json.loads(chunk)
                message = data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                content += message
            except Exception as e:
                print(f"解析异常：{e}")
            yield chunk
        await DBUtils.add_message(db, session_id, "assistant", content)

    def web_search(self, query):
        # TODO: 实现网络搜索
        pass 