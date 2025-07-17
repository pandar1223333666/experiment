class Settings:
    # 应用基础配置
    APP_NAME = "AI SelfCheck"
    DEBUG = True
    
    # 数据库配置
    DATABASE_URL = "sqlite+aiosqlite:///./chat.db"
    
    # 搜索引擎配置
    search_engine: str = "baidu"
    search_country: str = "cn"
    search_language: str = "zh-cn"
    max_search_results: int = 10
    serpapi_api_key: str = ""
    
    # LLM模型配置
    DEFAULT_SESSION_ID = "default_session"
    LLM_MODEL_NAME = "qwen-turbo"
    LLM_STREAMING = True
    
    # API配置
    QWEN_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    # 文件上传配置
    UPLOAD_DIR = "uploads"
    
    # 认证配置
    AUTH_ENABLED = False
    DEFAULT_API_KEY = "valid-token"

settings = Settings() 