from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Message(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="消息角色，如'user'或'assistant'")
    content: str = Field(..., description="消息内容")
    created_at: Optional[datetime] = Field(None, description="消息创建时间")

class ChatHistory(BaseModel):
    """聊天历史记录模型"""
    session_id: str = Field(..., description="会话ID")
    messages: List[Message] = Field(default_factory=list, description="消息列表")

class ChatResponse(BaseModel):
    """聊天回复模型"""
    reply: str = Field(..., description="AI回复内容")
    session_id: Optional[str] = Field(None, description="会话ID")

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户消息")
    session_id: Optional[str] = Field(None, description="会话ID，如果为空则创建新会话")
    
class SessionInfo(BaseModel):
    """会话信息模型"""
    session_id: str = Field(..., description="会话ID")
    created_at: Optional[datetime] = Field(None, description="会话创建时间")
    updated_at: Optional[datetime] = Field(None, description="会话最后更新时间")
    message_count: Optional[int] = Field(0, description="消息数量")

class SessionListResponse(BaseModel):
    """会话列表响应模型"""
    sessions: List[SessionInfo] = Field(default_factory=list, description="会话列表")
    total: int = Field(0, description="会话总数")

class FileUploadRequest(BaseModel):
    """文件上传请求模型"""
    session_id: Optional[str] = Field(None, description="会话ID，如果为空则创建新会话")

class FileUploadResponse(BaseModel):
    """文件上传响应模型"""
    session_id: str = Field(..., description="会话ID")
    file_name: str = Field(..., description="文件名")
    content_length: int = Field(..., description="文件内容长度")

class SearchResult(BaseModel):
    """搜索结果项模型"""
    title: str = Field(..., description="结果标题")
    url: str = Field("", description="结果URL")
    snippet: str = Field(..., description="结果摘要")
    rank: int = Field(..., description="结果排名")

class SearchResponse(BaseModel):
    """搜索响应模型"""
    query: str = Field(..., description="搜索查询")
    results: List[SearchResult] = Field(default_factory=list, description="搜索结果列表")
    total_results: int = Field(0, description="结果总数")
