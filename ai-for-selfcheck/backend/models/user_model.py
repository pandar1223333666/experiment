# user_model.py
# 用户ORM模型

from sqlalchemy import Column, Integer, String
from ..db.database import Base

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String) 