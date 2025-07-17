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
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="[%(levelname)s] %(asctime)s / %(name)s / %(message)s",
)
logger = logging.getLogger(__name__)

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("应用启动中...")
    # 初始化数据库
    await init_db()
    logger.info("数据库初始化完成")
    yield
    logger.info("应用关闭中...")

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
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

# 创建LLM服务实例
llm_service = LLMService()

# 注册路由
@app.post("/chat", response_model=ChatResponse, tags=["聊天"])
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """处理聊天请求"""
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"

    ai_reply = await llm_service.get_response(request.message.strip(),session_id=session_id)

    return ChatResponse(reply=ai_reply, session_id=request.session_id)

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    import os

    # 加载环境变量
    load_dotenv()

    # 检查API密钥
    if not os.getenv("QWEN_API_KEY"):
        print("警告: 未设置QWEN_API_KEY环境变量，请在.env文件中配置或手动设置环境变量")


    print("正在启动AI自检系统后端...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
    print("服务已启动，请访问 http://127.0.0.1:8000/docs 查看API文档") 