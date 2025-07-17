from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from services.memory_manager import router as chat_router
from db import init_db, get_db
from contextlib import asynccontextmanager
from services.llm_service import LLMService
from schemas import ChatResponse, ChatRequest
from sqlalchemy.ext.asyncio import AsyncSession
from config import settings
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("应用启动中...")
    # 初始化数据库
    await init_db()
    logger.info("数据库初始化完成")
    yield
    logger.info("应用关闭中...")

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    description="AI自检系统API",
    version="0.1.0",
    lifespan=lifespan,
    debug=settings.DEBUG
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router, prefix="/chat", tags=["聊天"])

# 创建LLM服务实例
llm_service = LLMService()

@app.get("/")
def read_root():
    """API根路径"""
    return {"message": f"欢迎使用 {settings.APP_NAME} 后端API!"} 

@app.post("/chat", response_model=ChatResponse, tags=["聊天"])
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """处理聊天请求"""
    ai_reply = await llm_service.get_response(request.message.strip())
    return ChatResponse(reply=ai_reply, session_id=request.session_id)

@app.get("/health", tags=["系统"])
def health_check():
    """健康检查接口"""
    return {"status": "ok", "version": "0.1.0"}

