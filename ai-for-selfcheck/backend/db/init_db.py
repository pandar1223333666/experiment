# init_db.py
# 创建表结构，初始化数据（可选）

from .database import Base, engine
from ..models.user_model import User
from ..models.session_model import Session

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == '__main__':
    init_db() 