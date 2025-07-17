from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from utils.logger import setup_logging, get_logger
from services.memory_manager import router as chat_router
from db import init_db, get_db
from services.llm_service import LLMService
from schemas import ChatResponse, ChatRequest


# 初始化自定义日志配置
setup_logging()

# 创建FastAPI应用
app = FastAPI(
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

# 注册路由(把功能添加到这)
@app.post("/chat", response_model=ChatResponse, tags=["聊天"])
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """处理聊天请求"""
    ai_reply = await llm_service.get_response(request.message.strip())
    return ChatResponse(reply=ai_reply, session_id=request.session_id)

# 启动后端
if __name__ == "__main__":
    import uvicorn


    # 自定义日志使用例子
    logger = get_logger(__name__) # 创建文件内日志，需要from utils.logger import get_logger，不能在全局区使用
    logger.debug("调用logger.debug") # 调试信息，用于开发阶段详细查看内部行为
    logger.info("调用logger.info") # 一般信息，比如程序启动、处理流程说明等
    logger.warning("调用logger.warning") # 警告，不一定是错误，但可能导致问题或需要注意
    logger.error("调用logger.error") # 错误事件，程序虽然还能继续运行，但某些功能已失败

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True) # 待理解

