from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from datetime import datetime
from config import settings
from typing import AsyncGenerator

# 使用配置中的数据库URL
DATABASE_URL = settings.DATABASE_URL

# 创建异步引擎
engine = create_async_engine(DATABASE_URL, echo=True, future=True)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()

# 数据模型定义
class Session(Base):
    """聊天会话模型"""
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    """聊天消息模型"""
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    session = relationship("Session", back_populates="messages")

# 数据库会话上下文管理器
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# 数据库初始化
async def init_db():
    """初始化数据库表结构"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# 数据库操作辅助函数
class DBUtils:
    @staticmethod
    async def get_session(db: AsyncSession, session_id: str):
        """获取会话，如果不存在则创建"""
        from sqlalchemy.future import select
        result = await db.execute(select(Session).where(Session.id == session_id))
        db_session = result.scalars().first()
        if not db_session:
            db_session = Session(id=session_id)
            db.add(db_session)
            await db.commit()
            await db.refresh(db_session)
        return db_session

    @staticmethod
    async def add_message(db: AsyncSession, session_id: str, role: str, content: str):
        """添加消息到会话"""
        message = Message(session_id=session_id, role=role, content=content)
        db.add(message)
        await db.commit()
        await db.refresh(message)
        return message

    @staticmethod
    async def get_messages(db: AsyncSession, session_id: str):
        """获取会话的所有消息"""
        from sqlalchemy.future import select
        result = await db.execute(select(Message).where(Message.session_id == session_id))
        return result.scalars().all()

    @staticmethod
    async def get_all_sessions(db: AsyncSession):
        """获取所有会话"""
        from sqlalchemy.future import select
        result = await db.execute(select(Session))
        return result.scalars().all()