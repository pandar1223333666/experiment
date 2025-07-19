# session_model.py
# 会话ORM模型

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from ..db.database import Base
from sqlalchemy.orm import relationship
import datetime

class Session(Base):
    __tablename__ = 'sessions'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    title = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship('User') 