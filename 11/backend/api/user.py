# user.py
# 注册/登录/认证接口

from fastapi import APIRouter

router = APIRouter()

@router.post('/register')
def register():
    # TODO: 用户注册接口
    pass

@router.post('/login')
def login():
    # TODO: 用户登录接口
    pass

@router.get('/me')
def get_me():
    # TODO: 获取当前用户信息接口
    pass 