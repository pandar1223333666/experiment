# 这里集成LangChain相关逻辑

from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from config import settings
import logging
import os
from typing import Dict, Any, List, Sequence, cast
from pydantic import SecretStr

logger = logging.getLogger(__name__)

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")

class LLMService:
    def __init__(self):
        self._init_model()
        self._init_app()
        self._init_config()
    
    def _init_model(self):
        """初始化通义千问模型"""
        # 将API密钥转换为SecretStr类型
        api_key = SecretStr(QWEN_API_KEY) if QWEN_API_KEY else None
        
        self.model = ChatTongyi(
            api_key=api_key,
            streaming=settings.LLM_STREAMING,
            name=settings.LLM_MODEL_NAME
        )
    
    def _init_app(self):
        """初始化LangGraph应用"""
        workflow = StateGraph(MessagesState)
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一位乐于助人的AI助手。请遵循以下准则：
            1. 用中文回答问题
            2. 保持友好和专业的态度
            3. 提供准确、有用的信息
            4. 如果不确定答案，请诚实说明"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        def call_model(state: MessagesState) -> Dict[str, Any]:
            """调用模型获取回复"""
            messages = state.get("messages", [])
            prompt = prompt_template.invoke({"messages": messages})
            response = self.model.invoke(prompt)
            return {"messages": messages + [response]}
        
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)
        
        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)    
    
    def _init_config(self):
        """初始化配置"""
        # 创建符合RunnableConfig类型的配置
        self.config: RunnableConfig = {
            "configurable": {
                "thread_id": settings.DEFAULT_SESSION_ID
            }
        }
    
    async def get_response(self, message: str) -> str:
        """获取AI回复
        
        Args:
            message: 用户输入的消息
            
        Returns:
            AI回复的内容
        """
        # 创建消息状态对象
        state = MessagesState(messages=[HumanMessage(content=message)])
        
        # 调用AI应用获取回复
        result = await self.app.ainvoke(
            state,
            config=self.config
        )
        
        # 提取AI回复内容
        ai_message = result["messages"][-1]
        reply = ai_message.content
            
        return reply
    
    async def get_streaming_response(self, message: str):
        """获取流式AI回复
        
        Args:
            message: 用户输入的消息
            
        Yields:
            AI回复的内容片段
        """
        # 创建消息状态对象
        state = MessagesState(messages=[HumanMessage(content=message)])
        
        # 调用AI应用获取流式回复
        async for chunk in self.app.astream(
            state,
            config=self.config
        ):
            if "messages" in chunk and len(chunk["messages"]) > 0:
                ai_message = chunk["messages"][-1]
                if isinstance(ai_message, AIMessage):
                    yield ai_message.content
         
    


