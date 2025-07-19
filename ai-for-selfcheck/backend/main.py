# main.py
# FastAPI 入口

from fastapi import FastAPI
from api.llm import router as llm_router
from api.session import router as session_router
from api.user import router as user_router
from db.database import init_db
import asyncio
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from services.llm_service import LLMService
from utils.config import settings

@asynccontextmanager
async def lifespan(app):
    await init_db()
    yield


app = FastAPI(lifespan=lifespan, debug=settings.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建LLM服务实例
llm_service = LLMService()

app.include_router(llm_router, prefix='/api/llm', tags=['llm'])
app.include_router(session_router, prefix='/api/session', tags=['session'])
app.include_router(user_router, prefix='/api/user', tags=['user'])

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000) 